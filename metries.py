import os
import numpy as np
from PIL import Image
import torch
from torchvision.models import inception_v3
from torchvision import transforms
from scipy.stats import entropy
from tqdm import tqdm
from pytorch_fid import fid_score
from torch_fidelity import calculate_metrics
import clip
from skimage.metrics import structural_similarity as ssim

# ----- PATHS -----
real_folder = './test_real'
output_folder = './test_output'

# ----- FID -----
def compute_fid():
    fid_value = fid_score.calculate_fid_given_paths(
        [real_folder, output_folder],
        batch_size=1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dims=2048
    )
    print(f"\n[FID] Score: {fid_value:.4f}")
    return fid_value

# ----- Inception Score -----
def get_inception_features(img_folder, device='cuda'):
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    features = []
    for filename in tqdm(os.listdir(img_folder), desc="Inception Features"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(img_folder, filename)).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = model(img)
                preds = torch.nn.functional.softmax(preds, dim=1)
                features.append(preds.cpu().numpy())
    return np.concatenate(features, axis=0)

def compute_inception_score(img_folder, device='cuda', splits=2):
    preds = get_inception_features(img_folder, device)
    N = preds.shape[0]
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = [entropy(part[i], py) for i in range(part.shape[0])]
        split_scores.append(np.exp(np.mean(scores)))
    mean_is, std_is = float(np.mean(split_scores)), float(np.std(split_scores))
    print(f"\n[Inception Score] {mean_is:.4f} ± {std_is:.4f}")
    return mean_is, std_is

# ----- CLIPScore -----
def compute_clip_score(captions):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    clip_scores = []
    image_files = sorted([f for f in os.listdir(output_folder) if f.endswith((".png", ".jpg"))])
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(output_folder, img_file)
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        text = clip.tokenize([captions[i]]).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).item()
            clip_scores.append(similarity)
    clip_scores = np.array(clip_scores)
    print(f"\n[CLIPScore] Mean: {clip_scores.mean():.4f}, Std: {clip_scores.std():.4f}")
    return clip_scores.mean(), clip_scores.std()

# ----- Torch-Fidelity (FID, IS, KID, Precision/Recall) -----
def compute_torch_fidelity_metrics():
    metrics = calculate_metrics(
        input1=real_folder,
        input2=output_folder,
        cuda=True,
        isc=True,
        fid=True,
        kid=True,
        pr=True,
        pr_subset_size=5,
        kid_subset_size=5
    )
    print("\n[Torch-Fidelity Metrics]")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    return metrics

# ----- PSNR -----
def psnr(img1_path, img2_path):
    img1 = np.array(Image.open(img1_path).convert('RGB'))
    img2 = np.array(Image.open(img2_path).convert('RGB'))
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 10 * np.log10((PIXEL_MAX ** 2) / mse)

def compute_psnr():
    real_images = sorted([f for f in os.listdir(real_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    output_images = sorted([f for f in os.listdir(output_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    psnr_values = []
    for real_img, output_img in zip(real_images, output_images):
        real_path = os.path.join(real_folder, real_img)
        output_path = os.path.join(output_folder, output_img)
        psnr_val = psnr(real_path, output_path)
        psnr_values.append(psnr_val)
    avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0
    print(f"\n[PSNR] Average: {avg_psnr:.2f} dB")
    return avg_psnr

# ----- SSIM -----
def calculate_ssim(img1_path, img2_path):
    img1 = np.array(Image.open(img1_path).convert('RGB'))
    img2 = np.array(Image.open(img2_path).convert('RGB'))
    img1_gray = np.mean(img1, axis=2)
    img2_gray = np.mean(img2, axis=2)
    return ssim(img1_gray, img2_gray, data_range=img2_gray.max() - img2_gray.min())

def compute_ssim():
    real_images = sorted([f for f in os.listdir(real_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    output_images = sorted([f for f in os.listdir(output_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    ssim_values = []
    for real_img, output_img in zip(real_images, output_images):
        real_path = os.path.join(real_folder, real_img)
        output_path = os.path.join(output_folder, output_img)
        ssim_val = calculate_ssim(real_path, output_path)
        ssim_values.append(ssim_val)
    avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0
    print(f"\n[SSIM] Average: {avg_ssim:.4f}")
    return avg_ssim

# ---------- MAIN ----------
if __name__ == "__main__":
    # Replace this with your actual list of captions corresponding to each image in ./test_output
    captions = [
        "A bottle of 17 cm height", "A bottle of 14 cm height",
        "A bottle of 17 cm height", "A bottle of 18 cm height", "A bottle of 19.2 cm height"
    ]

    print("Running All Evaluation Metrics...\n")

    compute_fid()
    compute_inception_score(output_folder)
    compute_clip_score(captions)
    compute_torch_fidelity_metrics()
    compute_psnr()
    compute_ssim()

    print("\nAll metrics computed successfully.")
