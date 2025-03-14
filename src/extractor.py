import os
import cv2 as cv

class_map = {
    0: "skier",
    1: "snowboarder",
}

# Function to extract objects from images
def extract_cnn_data(image_dir, labels_dir, output_dir):
    '''
        Function that extracts objects from images
        and saves them in class directories
    '''
    # Make sure the output directory exists
    # And create the class directories
    try:
        for class_name in class_map.values():
            os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

        # Loop through labels and extract objects
        for label_file in os.listdir(labels_dir):
            img_file = label_file.replace(".txt", ".jpg")
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(labels_dir, label_file)

            # Load the image
            if os.path.exists(img_path):
                image = cv.imread(img_path)
                if image is None or image.size == 0:
                    print(f"Warning: Image {img_path} is empty or could not be loaded.")
                    continue

                h, w, _ = image.shape

                # Read the YOLO label file
                with open(label_path, "r") as f:
                    lines = f.readlines()

                for i, line in enumerate(lines):
                    parts = line.strip().split(" ")
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * w
                    y_center = float(parts[2]) * h
                    width = float(parts[3]) * w
                    height = float(parts[4]) * h

                    # Calculate the bounding box coordinates
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

                    # Ensure the bounding box coordinates are within the image dimensions
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)

                    # Extract the object
                    object_img = image[y1:y2, x1:x2]

                    # Check if the extracted object is not empty
                    if object_img.size == 0:
                        print(f"Warning: Extracted object from {img_path} is empty.")
                        continue

                    # Save the object
                    class_name = class_map[class_id]
                    output_path = os.path.join(output_dir, class_name, f"{img_file}_{i}.jpg")

                    cv.imwrite(output_path, object_img)
    except Exception as e:
        print(f"Warning: Could not save object {output_path}; Error: {e}")