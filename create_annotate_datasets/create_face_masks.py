import os
import cv2
import face_recognition
import numpy as np

def create_face_mask(input_image_path, output_mask_path):
    """
    Detect faces in input_image_path, create a mask with white
    bounding-box regions for each face, and save to output_mask_path.
    """
    # Load the image using face_recognition
    image = face_recognition.load_image_file(input_image_path)
    
    # Convert image to RGB (face_recognition.load_image_file already loads as RGB)
    # But if you were using cv2.imread, you'd need to convert BGR -> RGB.
    # image = cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB)
    
    # Detect face locations
    face_locations = face_recognition.face_locations(image)
    
    # If no faces found, just create an all-black mask
    if not face_locations:
        # Create a black mask with same height/width as original
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.imwrite(output_mask_path, mask)
        return

    # Create an all-black mask (single channel) with same height/width
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    # Each face location is a tuple: (top, right, bottom, left)
    for (top, right, bottom, left) in face_locations:
        # Draw white rectangles (value = 255) on the mask
        cv2.rectangle(mask, (left, top), (right, bottom), (255), thickness=-1)
    
    # Save the mask to disk
    cv2.imwrite(output_mask_path, mask)


def process_images(input_folder, output_folder):
    """
    Loop through images in input_folder, detect faces, create bounding-box masks,
    and save them to output_folder with the same base filename.
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop over every file in the input folder
    for file_name in os.listdir(input_folder):
        # Construct full path to the image file
        input_path = os.path.join(input_folder, file_name)
        
        # Check if it's an image (you can add more checks if needed)
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            print(f"Skipping non-image file: {file_name}")
            continue
        
        # Construct output mask path
        base_name, ext = os.path.splitext(file_name)
        output_mask_path = os.path.join(output_folder, base_name + "_mask.png")

        try:
            create_face_mask(input_path, output_mask_path)
            print(f"Saved mask: {output_mask_path}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")



input_folder_path = "images"
output_folder_path = "face_masks"

process_images(input_folder_path, output_folder_path)
