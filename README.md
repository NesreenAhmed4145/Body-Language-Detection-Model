# 🧠 AI Body Language Recognition System

## 📌 Project Overview
This project utilizes Computer Vision and Machine Learning to detect and analyze human body language in real-time. It uses **MediaPipe** for pose estimation to extract skeletal landmarks and a trained **Random Forest** classifier to predict behavioral states.

Additionally, the system implements **geometric analysis** (calculating neck inclination angles) to detect "Low Confidence" posture (slouching) regardless of the specific gesture being performed.

---

## 📂 Dataset Statistics
The model was trained on a custom dataset containing approximately **5,500 images**, balanced across five distinct classes to ensure unbiased performance.

| Class Name | Image Count | Description |
| :--- | :--- | :--- |
| **Explaining** | 1,200 | Hand gestures indicating conversation |
| **Touching Face** | 1,161 | Hand touching or covering parts of the face |
| **Raised Hand** | 1,070 | Hand raised above shoulder level |
| **Straight** | 1,060 | Neutral standing/sitting posture |
| **Crossed Arms** | 990 | Defensive or closed posture |
| **TOTAL** | **~5,500** | |

---

## ⚙️ Methodology

### 1. Feature Extraction
We utilized **MediaPipe Pose** to detect 33 3D landmarks on the human body. The raw coordinates were processed into relative coordinates (normalized to the nose position) to make the model robust against camera distance and subject position.

### 2. Hybrid Analysis Approach
* **Machine Learning:** Classifies specific gestures (Crossed Arms, Raised Hand, etc.).
* **Geometric Logic:** Calculates the angle between the **Ear**, **Shoulder**, and a **Vertical Reference Point**. If the neck inclination exceeds a threshold (e.g., 25°), the system overrides the ML prediction to flag the posture as **"Low Confidence" (Slouching)**.

---

## 📊 Model Performance
Two algorithms were trained and evaluated using the extracted features. The **Random Forest Classifier** was selected for deployment due to its superior accuracy.

| Model | Accuracy | Status |
| :--- | :--- | :--- |
| **Random Forest** | **98.08%** | 🏆 **Selected** |
| Support Vector Machine (SVM) | 97.50% | |

**Conclusion:** The high accuracy confirms that the skeletal landmarks provide clear, separable patterns for these specific behavioral cues.

---

## 🚀 How to Run

### Prerequisites
Install the required libraries:
```bash
pip install opencv-python mediapipe pandas scikit-learn numpy
