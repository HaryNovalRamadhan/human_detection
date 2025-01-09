import cv2
import mediapipe as mp
from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponse
from django.views.decorators import gzip
import torch

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Move model initialization inside the function where it's needed
model = None

class VideoCamera:
    def __init__(self):
        print("Initializing camera")
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            raise ValueError("Could not open video device")

    def __del__(self):
        print("Releasing camera")
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        if not success:
            return None
        return image

def index(request):
    print("Index view called")
    x = torch.rand(5, 3)
    print(x)
    return render(request, 'detector/index.html')

def mediapipe_detection(request):
    print("MediaPipe detection view called")
    return render(request, 'detector/mediapipe_detection.html')

def yolo_detection(request):
    print("YOLO detection view called")
    return render(request, 'detector/yolo_detection.html')

def combined_detection(request):
    print("Combined detection view called")
    return render(request, 'detector/combined_detection.html')

def gen_mediapipe(camera):
    print("Starting MediaPipe stream")
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
            
        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )
            
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

def gen_yolo(camera):
    global model
    if model is None:
        print("Loading YOLO model")
        import yolov5
        model = yolov5.load('yolov5s.pt')
    
    print("Starting YOLO stream")
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
            
        # YOLOv5 detection
        results = model(frame)
        
        # Draw detections
        for det in results.xyxy[0]:
            if det[5] == 0:  # Class 0 is person in COCO dataset
                x1, y1, x2, y2 = map(int, det[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

def gen_combined(camera):
    global model
    if model is None:
        print("Loading YOLO model")
        import yolov5
        model = yolov5.load('yolov5s.pt')
    
    print("Starting combined stream")
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
            
        # MediaPipe detection
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(image_rgb)
        
        # Draw pose landmarks
        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results_pose.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
        # YOLOv5 detection
        results_yolo = model(frame)
        
        # Draw YOLO detections
        for det in results_yolo.xyxy[0]:
            if det[5] == 0:  # Class 0 is person in COCO dataset
                conf = float(det[4])
                if conf > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = map(int, det[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Add confidence score
                    cv2.putText(frame, f'Person {conf:.2f}', 
                              (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 2)
                
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@gzip.gzip_page
def video_feed_combined(request):
    print("Starting combined video feed")
    try:
        return StreamingHttpResponse(
            gen_combined(VideoCamera()),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        print(f"Error in combined video feed: {e}")
        return HttpResponse(f"Error: {str(e)}")

@gzip.gzip_page
def video_feed_mediapipe(request):
    print("Starting MediaPipe video feed")
    try:
        return StreamingHttpResponse(
            gen_mediapipe(VideoCamera()),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        print(f"Error in MediaPipe video feed: {e}")
        return HttpResponse(f"Error: {str(e)}")

@gzip.gzip_page
def video_feed_yolo(request):
    print("Starting YOLO video feed")
    try:
        return StreamingHttpResponse(
            gen_yolo(VideoCamera()),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        print(f"Error in YOLO video feed: {e}")
        return HttpResponse(f"Error: {str(e)}")