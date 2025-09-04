import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO
import uvicorn
import logging
import threading
import time
import os

# ---------------- INIT ----------------
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Load YOLO model
model_path = os.path.join(os.path.dirname(__file__), "ViolenceDetection.pt")
model = YOLO(model_path)

# Camera setup
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
logging.info(f"Camera initialized at {int(actual_width)}x{int(actual_height)} resolution")

# Control flag
detection_active = True

# ---------------- VIDEO STREAM ----------------
def video_stream():
    global detection_active
    while True:
        success, frame = camera.read()
        if not success:
            break
        if detection_active:
            results = model.predict(frame, verbose=False)
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame

        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ret:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

# ---------------- HTML UI ----------------
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
    <head>
        <title>Violence Detection</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { background-color: #121212; color: white; text-align: center; font-family: Arial, sans-serif; margin: 0; }
            header { padding: 15px; background-color: #1f1f1f; box-shadow: 0 2px 10px rgba(0,0,0,0.5); }
            h1 { color: #ff4d4d; margin: 0; }
            p { font-size: 16px; color: #ccc; margin-top: 5px; }
            img { border-radius: 10px; width: 95%; max-width: 1000px; margin-top: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
            #toggleBtn { position: fixed; bottom: 20px; right: 20px; padding: 15px 20px; font-size: 16px; border: none; border-radius: 50px; background-color: #ff4d4d; color: white; cursor: pointer; box-shadow: 0 4px 10px rgba(0,0,0,0.3); }
            #toggleBtn.paused { background-color: #4caf50; }
            footer { font-size: 14px; padding: 10px; color: #777; }
        </style>
    </head>
    <body>
        <header>
            <h1>Violence Detection Live Feed</h1>
            <p>Plug in your camera anytime — stream starts automatically</p>
        </header>
        <main>
            <img src="/video_feed" alt="Live Feed" id="feed"/>
        </main>
        <button id="toggleBtn" onclick="toggleDetection()">Pause Detection</button>
        <footer>
            &copy; 2025 - AI Violence Detection System
        </footer>
        <script>
            function toggleDetection() {
                fetch('/toggle').then(res => res.text()).then(status => {
                    let btn = document.getElementById('toggleBtn');
                    if (status === 'paused') {
                        btn.textContent = 'Resume Detection';
                        btn.classList.add('paused');
                    } else {
                        btn.textContent = 'Pause Detection';
                        btn.classList.remove('paused');
                    }
                });
            }
        </script>
    </body>
    </html>
    """

# ---------------- TOGGLE ENDPOINT ----------------
@app.get("/toggle")
def toggle_detection():
    global detection_active
    detection_active = not detection_active
    return "running" if detection_active else "paused"

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_config=None)
