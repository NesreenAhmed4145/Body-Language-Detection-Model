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

## 📈 Analysis & Feedback Metrics
The system provides a comprehensive JSON report after analyzing the video. Below are the key metrics and how they are calculated:

### 1. Dominant Movement
* **Definition:** The behavioral class that appeared most frequently throughout the video.
* **Why it matters:** Identifies the user's baseline behavior (e.g., were they mostly "Straight" or frequently "Touching Face"?).

### 2. Engagement Score (0.0 - 1.0)
* **Definition:** The percentage of time the user spent performing active gestures.
* **Calculation:** `(Count("Explaining") + Count("Raised Hand")) / Total Frames`.
* **Interpretation:** A higher score indicates high energy and active participation. Low scores may indicate passiveness.

### 3. Face Touch Count
* **Definition:** The total number of distinct times the user touched their face.
* **Logic:** Filters out fleeting detections to count intentional movements.
* **Insight:** Frequent face touching is often a psychological indicator of nervousness or cognitive load.

### 4. Crossed Arms Detection
* **Definition:** A boolean flag (`True`/`False`) indicating if defensive posture was detected.
* **Threshold:** Triggers only if "Crossed Arms" is detected with **high confidence (>75%)** for a sustained period.
* **Insight:** Crossing arms can create a barrier and signal defensiveness or resistance.

### 5. Posture Balance (%)
* **Definition:** The percentage of time the user's shoulders were horizontally aligned.
* **Logic:** Checks the vertical difference between left and right shoulder coordinates.
* **Insight:** High balance percentages (>80%) indicate a stable, professional sitting posture. Frequent leaning suggests low energy or casualness.

### 6. Body Openness
* **Definition:** Categorizes the user's posture as **Open**, **Normal**, or **Closed**.
* **Method:** Calculates the ratio between **Elbow Spread** and **Shoulder Width**.
    * **Open:** Elbows are wide (Power Pose).
    * **Closed:** Elbows are tucked in tightly (Defensive/Small).
* **Insight:** "Open" postures are generally correlated with confidence and authority.

---

## 🚀 How to Run

### Prerequisites
Install the required libraries:
```bash
pip install opencv-python mediapipe pandas scikit-learn numpy
