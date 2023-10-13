'''
Crops an image into 4 equal parts
'''

from PIL import Image
from tqdm import tqdm
import os

# Define the folder containing the images
image_folder = "./data/pRCC_nolabel/"
output_folder = "./data/pRCC_cropped/"


# Loop through each image in the folder
for image_name in tqdm(os.listdir(image_folder)):
    if image_name.endswith(('.jpg', '.png', '.jpeg')):  # Add or modify the file extensions as needed
        image_path = os.path.join(image_folder, image_name)
        
        # Open the image
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Calculate the size of each cropped region
            cropped_width = width // 4
            cropped_height = height // 4
            
            # Loop through each set of coordinates to perform the cropping
            for i in range(4):
                for j in range(4):
                    left = i * cropped_width
                    upper = j * cropped_height
                    right = (i + 1) * cropped_width
                    lower = (j + 1) * cropped_height
                    
                    cropped_img = img.crop((left, upper, right, lower))
                    
                    # Save the cropped image
                    cropped_img_path = os.path.join(output_folder, f"{image_name.split('.')[0]}_cropped_{i}_{j}.jpg")
                    cropped_img.save(cropped_img_path)