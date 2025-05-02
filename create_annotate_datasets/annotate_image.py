import os
import cv2
import numpy as np

def create_masks_with_manual_rois(input_folder, output_folder):
    """
    For each image in `input_folder`, this function opens an interactive 
    OpenCV window, lets you draw multiple bounding boxes (ROIs), then 
    creates a mask with white boxes for each ROI and saves it in `output_folder`.
    """

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of files in the input folder
    file_list = os.listdir(input_folder)

    for file_name in file_list:
        # Skip non-image files (simple check by extension)
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            print(f"Skipping non-image file: {file_name}")
            continue

        # Build full input path
        input_path = os.path.join(input_folder, file_name)
        # Read the image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Could not read image {file_name}, skipping.")
            continue

        # Create a copy for display (optional)
        display_image = image.copy()

        # Use OpenCV's selectROIs to get multiple bounding boxes
        #    - Instructions: 
        #       - Draw a box with mouse, press ENTER to confirm that box
        #       - Repeat for multiple boxes
        #       - Press ESC or close the window to finish selecting
        rois = cv2.selectROIs("Select ROIs (Press ESC/Close when done)", display_image, 
                              showCrosshair=True, fromCenter=False)

        # Once the selection window is closed, rois is a list of bounding boxes
        # Each ROI is (x, y, w, h)
        cv2.destroyWindow("Select ROIs (Press ESC/Close when done)")

        if len(rois) == 0:
            print(f"No ROIs selected for {file_name}, creating empty mask.")
            # Create an empty (black) mask
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
        else:
            # Create an empty (black) mask
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)

            # For each selected ROI, draw a filled white rectangle on the mask
            for roi in rois:
                x, y, w, h = roi
                # Draw the rectangle in white
                cv2.rectangle(mask, (x, y), (x + w, y + h), (255), thickness=-1)

        # Build output mask path
        base_name, ext = os.path.splitext(file_name)
        output_mask_path = os.path.join(output_folder, base_name + "_mask.png")

        # Save the mask
        cv2.imwrite(output_mask_path, mask)
        print(f"Saved mask for {file_name} -> {output_mask_path}")


if __name__ == "__main__":
    # Example usage:
    input_folder = "images_folder"    # Replace with your input folder
    output_folder = "logo_masks"      # Folder to save the generated masks

    create_masks_with_manual_rois(input_folder, output_folder)
