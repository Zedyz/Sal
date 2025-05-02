import os
import cv2
import numpy as np

def create_masks_with_manual_rois(input_folder, output_folder):
    """
    For each image in `input_folder`, this function opens an interactive 
    OpenCV window, lets you draw multiple bounding boxes (ROIs), then 
    creates a mask with white rectangles for each ROI and saves it in `output_folder`.
    Usage notes:
      - Draw a box with mouse, then press ENTER to confirm each box.
      - Repeat for multiple boxes.
      - Press ESC or close the window to finish selection for this image.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in input_folder
    file_list = os.listdir(input_folder)

    for file_name in file_list:
        # Skip non-image files (quick extension check)
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            print(f"Skipping non-image file: {file_name}")
            continue

        input_path = os.path.join(input_folder, file_name)
        image = cv2.imread(input_path)
        if image is None:
            print(f"Could not read image {file_name}, skipping.")
            continue

        # Optional: show instructions/usage
        print("\n---------------------------------")
        print(f"Image: {file_name}")
        print("[INSTRUCTIONS] Drag a box around the region of interest, then press ENTER.")
        print("Press ENTER again to draw another box, or press ESC/close window to finish.\n")

        # Let user select multiple ROIs
        rois = cv2.selectROIs(
            "Select ROIs (Press ESC or close window when done)",
            image, 
            showCrosshair=True,
            fromCenter=False
        )

        # Close the ROI selection window
        cv2.destroyWindow("Select ROIs (Press ESC or close window when done)")

        # Create a blank (black) mask, same size as the image
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        if len(rois) == 0:
            print(f"No ROIs selected for {file_name}, creating empty mask.")
        else:
            # For each ROI => (x, y, w, h)
            for (x, y, w, h) in rois:
                cv2.rectangle(mask, (x, y), (x + w, y + h), (255), thickness=-1)

        # Save mask
        base_name, _ = os.path.splitext(file_name)
        mask_filename = f"{base_name}_mask.png"
        output_mask_path = os.path.join(output_folder, mask_filename)
        cv2.imwrite(output_mask_path, mask)
        print(f"Saved mask for {file_name} -> {output_mask_path}")


if __name__ == "__main__":
    input_folder = "images_folder"  # Replace with your folder containing images
    output_folder = "logo_masks"    # Folder to save the generated masks

    create_masks_with_manual_rois(input_folder, output_folder)
