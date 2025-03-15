import os
import cv2 as cv
from ultralytics import YOLO
import shutil
import torch

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
        label_dirs = {}
        for class_name in class_map.values():
            os.makedirs(os.path.join(output_dir, class_name, "images"), exist_ok=True)
            label_dir = os.path.join(output_dir, class_name, "labels")
            label_dirs[class_name] = label_dir
            os.makedirs(label_dir, exist_ok=True)

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

                # Read the YOLO label file
                with open(label_path, "r") as f:
                    lines = f.readlines()

                    for i, line in enumerate(lines):
                        parts = line.strip().split(" ")
                        class_id = int(parts[0])

                        # Check if the extracted object is not empty
                        if image.size == 0:
                            print(f"Warning: Extracted object from {img_path} is empty.")
                            continue

                        # Save the object
                        class_name = class_map[class_id]
                        output_path = os.path.join(output_dir, class_name, "images", f"{img_file}_{i}.jpg")

                        # Save the cropped image and label
                        cv.imwrite(output_path, image)
                        shutil.copy(label_path, os.path.join(label_dirs[class_name], f"{label_file}_{i}.txt"))
    except Exception as e:
        print(f"Warning: Could not save object {output_path}; Error: {e}")
        

# Function to crop objects from images
def crop_data(image_dir, labels_dir, output_dir):
    '''
        Function that crop objects from images
        and saves them in images and labels directories
    '''
    # Make sure the output directory exists
    # And create the class directories
    output_img_dir = os.path.join(output_dir, "images")
    output_label_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Loop through labels and extract objects
    idx = 0;
    for label_file in os.listdir(labels_dir):
        idx += 1
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
                
                if len(parts) != 5:
                    print(f"Warning: Invalid label format in {label_path}")
                    continue
                
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
                
                # Save the cropped image
                cropped_img_name = f"{idx}_{i}.jpg"
                copped_img_path = os.path.join(output_img_dir, cropped_img_name)
                cv.imwrite(copped_img_path, object_img)

                # Save the cropped label
                cropped_label_name = f"{idx}_{i}.txt"
                cropped_label_path = os.path.join(output_label_dir, cropped_label_name)
                with open(cropped_label_path, "w") as f:
                    f.write(f"{class_id} {0.5} {0.5} {1.0} {1.0}\n")

def correct_label(model_path, image_dir, label_dir, output_dir):
    '''
        Function that corrects the label files
        to match the new extracted objects
        but keeps original bounding boxes and updates only the class if necessary.
        Writes corrected labels to a new file in the output directory.
    '''
    
    # Load the YOLO model
    model = YOLO(model_path)
    
    # Make sure the output directory exists
    output_img = os.path.join(output_dir, "images")
    output_label = os.path.join(output_dir, "labels")
    os.makedirs(output_img, exist_ok=True)
    os.makedirs(output_label, exist_ok=True)
    
    for image_name in os.listdir(image_dir):
        if not image_name.endswith((".jpg",".png",".jpeg",".JPG",".PNG",".JPEG")):
            # Skip non-image files
            continue
        
        img_path = os.path.join(image_dir, image_name)
        img_end = os.path.splitext(image_name)[1]
        label_path = os.path.join(label_dir, image_name.replace(img_end, ".txt"))
        
        if not os.path.exists(label_path):
            print(f"Label not found for {image_name}, skipping.")
            continue
        
        # Load the image and get predictions from the model
        results = model(img_path, conf=0.8)
        
        best_prediction = None
        best_confidence = -1
        img_width, img_height = results[0].orig_img.shape[1], results[0].orig_img.shape[0]
        
        for result in results:
            for box in result.boxes:
                confidence = box.conf.item()  # Extract the confidence score
                
                # Update if this box has a higher confidence
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_prediction = box
        
        # Read original label file
        with open(label_path, "r") as f:
            original_lines = f.readlines()
        
        updated_lines = []
        
        # Process the best prediction if found
        if best_prediction is not None:
            # Extract the bounding box from the best prediction
            predicted_class = int(best_prediction.cls)
            
            if box.xywh.shape != torch.Size([1,4]):  # Ensure it's a 1D tensor
                print(f"Invalid box shape for {image_name}, skipping label update.")
                continue
            
            # Limit the width to 64 pixels
            min_width = 64
            min_width_normalized = min_width / img_width  # Normalize max width to [0, 1]
            if img_width < min_width_normalized:
                print(f"Image width too small for {image_name}, skipping label update.")
                continue
            
            # Check if the predicted class is different from the original
            for line in original_lines:
                parts = line.strip().split()
                
                if len(parts) != 5:
                    print(f"Invalid label format for {image_name}, skipping label update.")
                    continue
                
                original_class = int(parts[0])
                original_x_center = float(parts[1])
                original_y_center = float(parts[2])
                original_width = float(parts[3])
                original_height = float(parts[4])
                
                # If the predicted class is different, update it
                if original_class != predicted_class:
                    updated_line = f"{predicted_class} {original_x_center} {original_y_center} {original_width} {original_height}\n"
                else:
                    updated_line = line  # Keep the original annotation if the class is the same
                
                updated_lines.append(updated_line)

            # Write the updated label to the new file in the output directory
            new_label_path = os.path.join(output_label, image_name.replace(img_end, ".txt"))
            with open(new_label_path, "w") as f:
                f.writelines(updated_lines)
            print(f"Updated label for {image_name}, saved to {new_label_path}")

            # Copy the image and label to the output directories
            shutil.copy(img_path, os.path.join(output_img, image_name))
        else:
            print(f"No high-confidence predictions for {image_name}, skipping label update.")
        