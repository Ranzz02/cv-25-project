import cv2 as cv
from yt_dlp import YoutubeDL
from ultralytics import YOLO
import torch

def count_classes(video_url, model_path, line_y):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Load YOLO model with FP16 precision on GPU
    model = YOLO(model_path, task="detect").to(device)

    # Video streaming setup
    ydl_opts = {"format": "best"}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        best_stream_url = info['url']

    cap = cv.VideoCapture(best_stream_url)
    assert cap.isOpened()

    skier_count, snowboarder_count = 0, 0
    counted_ids = set()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video frame is empty or processing is complete")
            break

        # Run YOLO model with optimized settings
        results = model.track(frame, persist=True, conf=0.5, iou=0.4, device=device)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                obj_id = box.id.item() if box.id is not None else -1
                if cls not in [0,1]:
                    continue
                
                # Filter out tiny detections (noise)
                if (x2 - x1) < 30 or (y2 - y1) < 30:
                    continue

                # Assign colors and class names
                if cls == 0:  
                    color, class_name, count = (255, 0, 0), "Skier", skier_count
                elif cls == 1:  
                    color, class_name, count = (0, 0, 255), "Snowboarder", snowboarder_count
                else:
                    continue

                # Prevent duplicate counting by checking motion
                if obj_id not in counted_ids and ((y1 + y2) // 2) > line_y:
                    counted_ids.add(obj_id)
                    if cls == 0:
                        skier_count += 1
                    elif cls == 1:
                        snowboarder_count += 1

                # Draw bounding box
                cv.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # Draw class label and count
                label = f"{class_name} {count}"
                cv.putText(frame, label, (x1, y2 + 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw the counting line
        cv.line(frame, (0, line_y), (frame.shape[1], line_y), (107, 107, 107), 3)

        # Display the frame
        cv.imshow("Tracker", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
