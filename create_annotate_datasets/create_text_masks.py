import os
import cv2
import easyocr
import numpy as np

def create_text_mask_easyocr(input_image_path, output_mask_path, 
                             reader, conf_threshold=0.3):
    """
    Detect text in input_image_path using EasyOCR, create a mask 
    with white bounding-box regions for each text area (above conf_threshold),
    and save to output_mask_path.
    """
    # Read image with OpenCV (BGR format)
    image_bgr = cv2.imread(input_image_path)
    if image_bgr is None:
        print(f"Error reading {input_image_path}")
        return
    
    # Convert BGR to RGB (EasyOCR expects RGB or grayscale)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Use EasyOCR to detect text
    # Returns list of [ [ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ], text, confidence ]
    results = reader.readtext(image_rgb, detail=1)
    
    # Prepare a blank mask (single-channel) with the same dimensions
    height, width = image_rgb.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Loop through each detection
    for (coords, detected_text, confidence) in results:
        # coords = [ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ]
        if confidence >= conf_threshold:
            # Extract the bounding box in an axis-aligned manner
            xs = [int(point[0]) for point in coords]
            ys = [int(point[1]) for point in coords]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            # Draw a filled white rectangle on the mask
            cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, thickness=-1)
    
    # Save the mask
    cv2.imwrite(output_mask_path, mask)
    print(f"Saved text mask to {output_mask_path}")


def process_images_easyocr(input_folder, output_folder, languages=['en'], conf_threshold=0.3):
    """
    Loop through images in input_folder, detect text with EasyOCR, 
    create bounding-box masks, and save them to output_folder 
    with the same base filename.
    """
    # Initialize the EasyOCR reader once (for performance)
    reader = easyocr.Reader(languages, gpu=True)  
    # Set gpu=True if you have a GPU and want faster inference

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop over every file in the input folder
    for file_name in os.listdir(input_folder):
        # Construct full path to the image file
        input_path = os.path.join(input_folder, file_name)
        
        # Check if it's an image by extension
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            print(f"Skipping non-image file: {file_name}")
            continue
        
        # Construct output mask path
        base_name, ext = os.path.splitext(file_name)
        output_mask_path = os.path.join(output_folder, base_name + "_text_mask.png")

        # Create the text mask
        try:
            create_text_mask_easyocr(input_path, output_mask_path, 
                                     reader, conf_threshold=conf_threshold)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")



input_folder_path = "images"
output_folder_path = "text_masks"

# Process all images in input_images/ and save masks to output_masks/
process_images_easyocr(input_folder_path, output_folder_path, 
                       languages=['en'], conf_threshold=0.3)
