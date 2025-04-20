# 🏋️‍♂️ Realtime Exercise Tracker

**A real-time pose-based fitness tracker using YOLO for detecting and counting squats and push-ups.**

---

## 📌 Overview

`AIGym_Modified` is an extended solution based on [Ultralytics YOLO Pose](https://docs.ultralytics.com/tasks/pose/) that automatically:

- Detects multiple people in a video stream or from a webcam
- Classifies the performed exercise as **squat** or **push-up**
- Tracks each person's repetitions and stage in real-time
- Uses joint angles to ensure accurate repetition counting

> **Credits:** This project builds on top of the YOLO Pose estimation system provided by [Ultralytics](https://github.com/ultralytics/ultralytics), 
and specifically modifies their [ai_gym.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/ai_gym.py) class. 
The original `BaseSolution` structure has been extended to include multi-person exercise 
classification and per-person repetition tracking logic.

---

## 📸 Example Use Case

A webcam is pointed at the user.  
The system detects poses, identifies whether each person is doing squats or push-ups, and tracks their reps.  
Realtime feedback is displayed directly on the screen, including:
- Person ID
- Exercise type
- Joint angle
- Stage (up/down)
- Repetition count

**⚠️ Important Input Requirements:**
For the system to function accurately, the following body parts must be clearly visible in the input feed:
- **Shoulders**
- **Hips**
- **Arms**
- **Feet**

These key body parts are essential for **detecting joint angles and classifying the exercises correctly**. Ensure that the camera is positioned in a way that these areas are not obstructed, as this will help the system provide precise tracking and repetition counting.

---

## ⚙️ Features

- 🔄 **Automatic exercise classification** based on keypoint positions (squat or push-up)
- 🔢 **Per-person repetition counter** with live update
- 👥 **Multi-person tracking** with dynamic initialization
- 🎯 **Pose-specific keypoint analysis** for accurate stage transitions

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/lucasoricetti/Realtime-Exercise-Tracker.git
cd Realtime-Exercise-Tracker
```

### 2. Virtual Environment
I recommend to create a virtual environment to isolate project dependencies:
```bash
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate
```

### 3. Install Requirements
Make sure you have Python 3.8+ and the required packages:
```bash
pip install -r requirements.txt
```
You also need the appropriate YOLO pose weights from Ultralytics.
> 💡 You can download models such as yolo11m-pose.pt from [Ultralytics Models](https://docs.ultralytics.com/models/yolo11/), 
otherwise the model you choose will be installed before the process starts.

---

## 🏃 Running the Tracker
❓You can use the `help` command to see a detailed list of the customizable parameters:
```bash
python main.py --help
```
### Examples
▶️ Webcam Input:
```bash
python main.py --input webcam --model s
```
📼 Video File Input:
```bash
python main.py --input video --video_path ./videos/your_video.mp4 --model l --save
```

---

## 🧠 How It Works
The `AIGym_Modified` class extends YOLO's `BaseSolution` to perform the following steps:
1. **Detect Exercise Type from Keypoint Posture**  
    - **Upright Torso** → Squat  
    - **Horizontal Torso** → Push-up
2. **Track Joint Angles**  
    - **Squats** → Hip, Knee, Ankle  
    - **Push-ups** → Shoulder, Elbow, Wrist
3. **Determine stage transitions**
    - **Down** → when angle is below down_angle
    - **Up** → when angle exceeds up_angle
4. **Count reps when transitioning from up to down**
5. **Draw overlays with angles, counters, and exercise types**

---

## 📎 Acknowledgements
Special thanks to Ultralytics for their powerful **YOLOv11 pose estimation framework**.

## 📜 License
This project is open-source and follows the licensing of the underlying YOLO model you integrate.

## 🤝 Contributing
Found a bug or have an improvement idea? Feel free to open an issue or pull request!