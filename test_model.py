import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
from collections import Counter # Import Counter to calculate the most frequent action

# 1. Load the trained model
print("⏳ Loading model...")
with open('body_language_model.pkl', 'rb') as f:
    model = pickle.load(f)
print("✅ Model loaded successfully.")

# 2. Setup MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 3. Initialize Video
video_path = 'C:/Users/anesr/Downloads/Video_Interview_Example_A_Step-by-Step_Guide_for_Virtual_Success_Indeed_720P.mp4' 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Cannot open video. Check the path!")
    exit()

# --- NEW: List to store all predictions ---
all_predictions = [] 



# 4. Processing Loop

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Convert colors (BGR to RGB)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Extract landmarks
    results = pose.process(image)
    
    # Revert colors for display (RGB to BGR)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Draw landmarks on the body (optional)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    try:
        # Extract coordinates
        landmarks = results.pose_landmarks.landmark
        upper_body = landmarks[:23] 
        
        row = []
        nose = upper_body[0]
        
        # Calculate Relative Coordinates
        for lm in upper_body:
            rel_x = lm.x - nose.x
            rel_y = lm.y - nose.y
            rel_z = lm.z - nose.z
            row.extend([rel_x, rel_y, rel_z, lm.visibility])
            
        col_names = []
        for i in range(23):
            col_names += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']
            
        X = pd.DataFrame([row], columns=col_names)

        # 🔮 Make Prediction
        body_language_class = model.predict(X)[0] 
        body_language_prob = model.predict_proba(X)[0] 
        prob_value = round(body_language_prob[np.argmax(body_language_prob)], 2) 

        # --- NEW: Save this prediction to our list ---
        all_predictions.append(body_language_class)

        # Draw result on screen
        cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
        
        cv2.putText(image, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, body_language_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(image, 'PROB', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(prob_value), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    except Exception as e:
        pass

    cv2.imshow('AI Body Language Analysis', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
# ... (Previous code remains the same) ...

cap.release()
cv2.destroyAllWindows()

# --- NEW: Detailed Final Analysis Report ---
print("\n" + "="*40)
print("📊 FINAL VIDEO ANALYSIS RESULT")
print("="*40)

if len(all_predictions) > 0:
    # Count the occurrence of each movement
    count = Counter(all_predictions)
    total_frames = len(all_predictions)
    
    # Sort from most frequent to least frequent
    sorted_counts = count.most_common() 
    
    # Get the winner (the first one in the sorted list)
    most_common_movement, frequency = sorted_counts[0]
    
    print(f"🏆 DOMINANT MOVEMENT: {most_common_movement}")
    print("-" * 40)
    print(f"{'MOVEMENT':<20} | {'PERCENTAGE':<10} | {'FRAMES'}")
    print("-" * 40)

    # Loop through all classes and calculate percentage
    for movement, freq in sorted_counts:
        percentage = (freq / total_frames) * 100
        print(f"{movement:<20} | {percentage:6.1f}%    | {freq}")

else:
    print("⚠️ No movements were detected in the video.")
print("="*40)