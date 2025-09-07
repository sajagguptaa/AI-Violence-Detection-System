
# AI Violence Detection System 🔍

A real-time **violence detection system** built using **YOLOv8, OpenCV, and FastAPI**.  
The project uses a webcam as input, performs inference through a trained YOLO model, and streams live annotated video in a browser-based interface.  

---

## 🚀 Features
- Real-time violence detection using YOLOv8.
- Webcam-based live video feed with bounding box annotations.
- Accessible web interface (FastAPI + HTML + JavaScript).
- Pause/Resume detection with a single click.
- Optimized for low-latency and smooth streaming.
- Easy packaging into a `.exe` using PyInstaller.

---

## 📂 Project Structure


AI-Violence-Detection-System/
├─ launcher.py          # Main FastAPI + YOLO application
├─ requirements.txt     # Dependencies (install Torch separately)
├─ ViolenceDetection.pt # Trained YOLO model (not included in repo)
├─ scripts/             # Build scripts for Windows
│   ├─ build\_windows.ps1
│   └─ build\_windows\_cmd.bat
└─ README.md            # Project documentation

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

git clone https://github.com/sajagguptaa/AI-Violence-Detection-System.git
cd AI-Violence-Detection-System


### 2. Create Virtual Environment (Recommended)


python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate


### 3. Install Dependencies


pip install -r requirements.txt


> ⚠️ Install PyTorch manually (CPU or CUDA version) from [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Add YOLO Model

Place your trained model in the root directory as:


AI-Violence-Detection-System/ViolenceDetection.pt

---

## ▶️ Run the Application

python launcher.py

Then open your browser at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 📦 Build Executable (Windows)

Using PowerShell:
./scripts/build_windows.ps1

Using CMD:
scripts\build_windows_cmd.bat

The executable will be created in the `dist/` folder as `launcher.exe`.

---

## 🛠 Troubleshooting

* **Black Screen:** Close other apps using the webcam or try `CAM_INDEX=1 python launcher.py`.
* **Model Not Found:** Ensure `ViolenceDetection.pt` exists in the root folder.
* **High CPU Usage:** Pause detection using the floating button or lower inference size.

---

## 📖 Tech Stack

* **YOLOv8** (Ultralytics)
* **OpenCV**
* **FastAPI**
* **Uvicorn**
* **PyInstaller** (for packaging)

---

## 📜 License

This project is licensed under the **MIT License** – free to use and modify.

---

## ✨ Credits

* Internship Project by **Sajag Gupta**
* YOLOv8 by [Ultralytics](https://github.com/ultralytics/ultralytics)
* FastAPI by Sebastián Ramírez & contributors

---
