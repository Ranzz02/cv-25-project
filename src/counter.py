import cv2 as cv
import time
from yt_dlp import YoutubeDL
from ultralytics.solutions import ObjectCounter

def count_classes(video_url, model_path, line_points):
    best_strea_url = None
    
    # Get the best stream url
    ydl_opts = {
        "format": "best",
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        best_strea_url = info['url']
        
    # Create a new VideoCapture object and read the video from the best stream url
    cap = cv.VideoCapture(best_strea_url)
    assert cap.isOpened()
    
    counter = ObjectCounter(show=True, line=line_points, model=model_path, classes=[0,1])

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video frame is empty or processing is complete")
            break
        
        results = counter.process(frame)
        frame = results.annotator.result()
        
        cv.line(frame, [0, line_points], [1920, line_points], (0,255,0), 3)
        
        # Render the frame
        cv.imshow("Skier & Snowboarder counter", frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()