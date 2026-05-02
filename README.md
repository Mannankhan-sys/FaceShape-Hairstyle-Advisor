# AI Face Shape Detection & Hairstyle Advisor

A real-time Computer Vision application that detects a user's face shape and provides personalized hairstyle and beard recommendations using geometric feature extraction and AI-driven landmark analysis.

## 🚀 Features
- **Real-time Detection:** Live camera feed with 30fps landmark tracking.
- **AI-Powered Analysis:** Uses MediaPipe's 468-point 3D Face Mesh for high-precision feature extraction.
- **Smart Face Tracking:** Automatically identifies and focuses on the primary user (the largest face found in the frame).
- **Geometric Classification:** Rule-based expert system that calculates facial ratios (Height, Width, Jawline, Forehead) to classify shapes:
  - Oval, Round, Square, Oblong, Diamond, Heart.
- **Dynamic Recommendations:** Gender-aware suggestions for both hairstyles and beard styles that update instantly.
- **Modern UI:** Premium dark-mode interface built with `CustomTkinter`.

## 🛠️ Technical Stack
- **Python 3.10+**
- **MediaPipe:** For 3D facial landmark detection.
- **OpenCV:** For video processing and camera handling.
- **CustomTkinter:** For a modern, high-tech user interface.
- **PIL (Pillow):** For image processing and display.

## 📐 How it Works (AI Methodology)
The project follows a classic AI pipeline:
1.  **Data Acquisition:** Raw frames are captured via OpenCV and mirrored.
2.  **Feature Extraction:** MediaPipe Face Mesh extracts 468 landmarks.
3.  **Feature Engineering:** We calculate Euclidean distances between specific landmarks (e.g., forehead edges, cheekbones, chin).
4.  **Classification:** The system calculates specialized ratios (e.g., `Face Height / Cheekbone Width`) and passes them through a rule-based classifier to determine the face shape.
5.  **Heuristics Engine:** Based on the classified shape and selected gender, a curated dictionary of recommendations is displayed.

## 📦 Installation

1. **Clone the repository** (or navigate to the folder).
2. **Setup Virtual Environment:**
   ```bash
   python -m venv .venv310
   source .venv310/Scripts/activate  # On Windows
   ```
3. **Install Dependencies:**
   ```bash
   pip install customtkinter mediapipe opencv-python pillow numpy
   ```

## 🖥️ Usage

Run the modernized GUI application:
```bash
python gui_face_shape_modern.py
```

1. Select your **Gender** (Male/Female).
2. Click **Start Analysis**.
3. Align your face within the camera preview.
4. View your detected **Face Shape**, **Confidence Score**, and **Recommendations** in the sidebar.

## 📁 Project Structure
- `gui_face_shape_modern.py`: The main entry point (Modern GUI).
- `face_shape.py`: The core AI classification logic.
- `recommendations.py`: Data mappings for styles.
- `gui_face_shape_images/`: (Optional) Folder for thumbnail images.
- `imp.py`: Original terminal-based demo.

---
*Created for AI Class Project Presentation.*
