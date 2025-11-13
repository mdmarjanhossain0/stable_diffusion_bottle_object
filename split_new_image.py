import os
import shutil
from pathlib import Path

# Set your paths
source_folder = "./new_original"
destination_folder = "./new_generated"

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Loop through each file in the source folder
for filename in os.listdir(source_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
        # Get full source path
        src_path = os.path.join(source_folder, filename)
        
        # Remove extension and extract base name
        name_without_ext = Path(filename).stem
        ext = Path(filename).suffix

        # Look for " and " to split
        if " and " in name_without_ext:
            try:
                prefix = "A bottle of "
                details = name_without_ext.replace(prefix, "")
                part1, part2 = details.split(" and ")

                # Create full new filenames with prefix added back
                new_name1 = prefix + part1.strip() + ext
                new_name2 = prefix + part2.strip() + ext

                dest_path1 = os.path.join(destination_folder, new_name1)
                dest_path2 = os.path.join(destination_folder, new_name2)

                # Copy image with new names
                shutil.copy(src_path, dest_path1)
                shutil.copy(src_path, dest_path2)

                print(f"Copied to:\n - {dest_path1}\n - {dest_path2}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"Skipping (no 'and' in filename): {filename}")
