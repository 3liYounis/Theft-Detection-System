# 🚨 AI Theft Detection System

**Real-time Behavioral Analysis | Computer Vision | Deep Learning**

> *Transforming surveillance from passive recording to active prevention.*

This project implements an intelligent security system that doesn't just "see" people—it **understands** their actions. By combining state-of-the-art Computer Vision with Deep Learning, the system detects suspicious behavior (like theft) in real-time, drastically reducing reaction times for security personnel.

---

## 🎥 Demo
Check out the system in action:

[**▶️ Watch the Main Pipeline Output**](https://drive.google.com/drive/folders/1hJR54wdDWOx8r9qLvAeAMcHG5Bm1VWUS?usp=sharing)

---
## 📃 Model Results
![ALT](<Source Code/Behaveioral/training_results/DEMO PIC.jpg>)
![ALT](<Source Code/Behaveioral/training_results/Summary.png>)

## 🛠️ The Tech Stack

Built with a robust pipeline of modern libraries:

*   **Logic & Processing**: `Python`, `NumPy`, `Pandas`
*   **Computer Vision (The Eyes)**: `OpenCV`, `MediaPipe`, `YOLOv8` (Ultralytics)
*   **Deep Learning (The Brain)**: `TensorFlow` (Keras), `LSTM Networks`
*   **Machine Learning**: `Scikit-learn` (for data preprocessing)

---

## 🧠 Behind the Scenes: How It Works

This system mimics human cognition to identify threats. Here's the "HR-Loveable" breakdown of the architecture:

### 1. The Eyes: Detection & Tracking 👀
We use **YOLOv8** to instantly detect people entering the frame and assign them unique IDs. Once tracked, **MediaPipe** extracts their skeletal pose landmarks (shoulders, elbows, wrists). This is like giving the computer the ability to see body language, not just pixels.

### 2. The Brain: Behavioral Analysis 🤖
This is where the magic happens. We don't just look at a single frame; we analyze **time**.
*   An **LSTM (Long Short-Term Memory)** Neural Network processes sequences of movements over time.
*   It looks for specific patterns indicative of theft (e.g., reaching, concealing, swift erratic movements).
*   If the probability crosses a threshold (e.g., 90%), the system flags the behavior as "Shoplifting".

### 3. The Memory: Identity Persistence 👤
If a theft is detected, the system:
*   Captures a video clip of the incident.
*   Uses **Face Recognition** to extract facial features.
*   Logs the suspect's identity (or marks them as "Unknown") and timestamps the event for security alerts.

---

## 📂 Project Structure

A clean, modular architecture designed for scalability:

```text
28. Final Project/
├── Data/                   # Raw videos and training datasets
├── Source Code/
│   ├── main.py             # 🚀 Entry point of the application
│   ├── Behaveioral/        # LSTM model, training scripts, and scalers
│   ├── Detection/          # Object detection logic
│   ├── Recognition/        # Face recognition modules
│   ├── Alert/              # Alerting system configuration
│   └── suspect_clips/      # Auto-generated evidence clips
├── requirements.txt        # Project dependencies
└── README.md               # You are here!
```

---

## 🔐 Configuration & Secrets

This project relies on sensitive credentials that are **excluded** from the repository for security. You must provide them manually to run the full pipeline.

### 1. Firebase Credentials
The system uses Firebase for real-time alerts.
*   **Missing File**: `Source Code/Alert/serviceAccountKey.json`
*   **How to get it**:
    1.  Go to your Firebase Console > Project Settings > Service Accounts.
    2.  Generate a new private key.
    3.  Save the downloaded JSON file as `serviceAccountKey.json` in `Source Code/Alert/`.

### 2. Firebase Service Script
*   **Missing File**: `Source Code/Alert/firbaseService.py`
*   **Reason**: Contains custom initialization logic or API keys.
*   **Structure**: This file should export an initialized `alerts_db` object.

### 3. Face Database
*   **Excluded File**: `portrait_database.json`
*   **Reason**: Auto-generated local database of face embeddings.
*   **Action**: This file will be automatically created when you run the face recognition module for the first time by scanning the `Data/Faces` directory.

---

## 🚀 Getting Started

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Pipeline**:
    ```bash
    python "Source Code/main.py"
    ```
