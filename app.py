import os, sys, base64, json
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import cv2, numpy as np

app = FastAPI()

# Resolve files when frozen by PyInstaller
BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))

# Bundle your model next to the exe via --add-data
MODEL_PATH = os.path.join(BASE_DIR, "ViolenceDetection.pt")

# Load YOLO once
model = YOLO(MODEL_PATH)

# Inlined UI (overlay + siren + smooth camera)
HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"/><title>Violence Detection</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<script src="https://cdn.tailwindcss.com"></script></head>
<body class="bg-gray-900 text-white flex flex-col items-center p-5">
<h1 class="text-3xl font-bold mb-4">Violence Detection</h1>
<div id="sr-live" class="sr-only" aria-live="assertive" aria-atomic="true"></div>
<div id="alert" class="hidden text-center p-3 rounded-lg bg-red-600 text-white text-xl font-bold mb-4 animate-pulse">🚨 Violence Detected! 🚨</div>
<audio id="siren" src="https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg" preload="auto"></audio>
<video id="video" autoplay playsinline muted class="absolute opacity-0"></video>
<canvas id="canvas" class="rounded-lg shadow-lg w-full max-w-5xl aspect-video"></canvas>
<div id="overlay" class="hidden fixed inset-0 z-50 bg-red-700/70 backdrop-blur-sm flex items-center justify-center">
  <div class="text-center">
    <div class="text-5xl font-extrabold text-white drop-shadow mb-4 animate-pulse">🚨 VIOLENCE DETECTED 🚨</div>
    <p class="text-white/90 text-lg">Press <kbd class="px-2 py-1 bg-white/20 rounded">M</kbd> to mute/unmute siren.</p>
  </div>
</div>
<script>
const W=640,H=480;
const video=document.getElementById("video");
const canvas=document.getElementById("canvas");
const ctx=canvas.getContext("2d");
const alertBox=document.getElementById("alert");
const overlay=document.getElementById("overlay");
const siren=document.getElementById("siren");
const srLive=document.getElementById("sr-live");
let was=false, muted=false;

navigator.mediaDevices.getUserMedia({video:{width:W,height:H}})
  .then(s=>video.srcObject=s)
  .catch(e=>{srLive.textContent="Camera blocked/unavailable."; console.error(e);});

const ws=new WebSocket(`ws://${location.host}/ws`);
ws.onmessage=(ev)=>{
  const d=JSON.parse(ev.data); const v=!!d.violence;
  if(v){alertBox.classList.remove("hidden"); overlay.classList.remove("hidden");
    if(!was){ if(!muted){ try{siren.currentTime=0; siren.play();}catch{} }
      if(navigator.vibrate) navigator.vibrate([150,80,150]);
      srLive.textContent="Alert: Violence detected."; }
    was=true;
  } else {
    alertBox.classList.add("hidden"); overlay.classList.add("hidden");
    if(was) srLive.textContent="Status: Clear."; was=false;
  }
  const img=new Image(); img.src=d.image; img.onload=()=>{canvas.width=W; canvas.height=H; ctx.drawImage(img,0,0,W,H);};
};
setInterval(()=>{const t=document.createElement("canvas"),c=t.getContext("2d"); t.width=W; t.height=H;
  c.drawImage(video,0,0,W,H); ws.send(t.toDataURL("image/jpeg",0.5));},100);
addEventListener("keydown",(e)=>{if(e.key.toLowerCase()==="m"){muted=!muted; if(muted){try{siren.pause();}catch{}} srLive.textContent=muted?"Siren muted.":"Siren unmuted.";}});
</script></body></html>"""

@app.get("/")
async def root():
    return HTMLResponse(HTML)

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    while True:
        data_url = await websocket.receive_text()
        img_bytes = base64.b64decode(data_url.split(",")[1])
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Inference at fixed size for speed/stability
        results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
        violence = False
        r = results[0]
        if r.boxes is not None and r.boxes.cls is not None:
            names = r.names
            for c in r.boxes.cls:
                try:
                    if names[int(c)] in ("Violence","violence","fight","fighting"):
                        violence = True
                        break
                except Exception:
                    pass

        # (Optional) send annotated frame: r.plot()
        ok, buf = cv2.imencode(".jpg", frame)
        await websocket.send_json({
            "image": "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8"),
            "violence": violence
        })
