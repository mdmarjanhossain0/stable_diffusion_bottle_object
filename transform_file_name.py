import os
from pathlib import Path

folder = "./training_samples"

for filename in os.listdir(folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
        if "[" in filename or "]" in filename:
            old_path = os.path.join(folder, filename)
            new_filename = filename.replace("[", "").replace("]", "")
            new_path = os.path.join(folder, new_filename)

            # If file exists, add a suffix like _1, _2, etc.
            if os.path.exists(new_path):
                stem = Path(new_filename).stem
                ext = Path(new_filename).suffix
                counter = 1
                while True:
                    candidate = f"{stem}_{counter}{ext}"
                    candidate_path = os.path.join(folder, candidate)
                    if not os.path.exists(candidate_path):
                        new_path = candidate_path
                        break
                    counter += 1

            os.rename(old_path, new_path)
            print(f"Renamed: {filename} → {os.path.basename(new_path)}")
