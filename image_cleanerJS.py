# -----------------------------------------------------------
# IMAGE CLEANER (Connected Pixels Version)
# ------------------------------------------------
#this version ignores tiny specks/hairs by only keeping
# large consecutive groups of non-white pixels.
# -----------------------------
import cv2      # OpenCV library for image processing
import numpy as np  # NumPy library for array/matrix operations
import os       # OS library for working with folders/files

# ---- 1. Folder paths ----
input_folder = "input_images"
output_folder = "output_images"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

#  2. Parameters you can change ----
threshold_value = 240    # Anything brighter than this is "white"
min_blob_size = 5000     # Ignore blobs smaller than this many pixels
extra_crop = 18          # Safety crop from all sides after main crop

#  3. Process each image ---
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        file_path = os.path.join(input_folder, filename)

        # Read the image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Skipping unreadable file: {filename}")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold to find non-white areas
        _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

        # 4. Connected components analysis ----
        #This labels every group of connected white pixels in the mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # stats[i] gives [x, y, width, height, area] for each blob
        # label 0 is background, so skip it
        largest_label = None
        largest_area = 0

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_blob_size and area > largest_area:
                largest_area = area
                largest_label = label

        if largest_label is None:
            print(f"No large object found in {filename}, skipping.")
            continue

        # Create a mask for the largest blob
        main_blob_mask = np.uint8(labels == largest_label) * 255



        #find bounding box of the main blob
        coords = cv2.findNonZero(main_blob_mask)
        x, y, w, h = cv2.boundingRect(coords)

        #crop the main object
        cropped = image[y:y+h, x:x+w]

      

        #Save the cleaned image
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, cropped)
        print(f"Cleaned and saved: {filename}")

