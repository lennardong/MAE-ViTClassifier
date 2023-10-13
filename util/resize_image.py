'''
Resizes all images to be the same dimensions
'''

from PIL import Image
import os
from tqdm import tqdm

# Source folder containing the images
src_folder = "/users/Lenn/Desktop/MAETest/Test"

# Destination folder to save resized images
dst_folder = "/users/Lenn/Desktop/MAETest/Test"

# Create the destination folder if it doesn't exist
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

# Loop through each file in the source folder
for filename in tqdm(os.listdir(src_folder)):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # Add or remove file extensions as needed
        img_path = os.path.join(src_folder, filename)
        
        # Open an image file
        img = Image.open(img_path)
        
        # Resize the image
        img_resized = img.resize((500,500))
        
        # Save the image to the destination folder
        img_resized.save(os.path.join(dst_folder, filename))

print("Resizing completed.")
