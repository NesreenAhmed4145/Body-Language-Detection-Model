import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
from collections import Counter

# ---------------------------------------------------------
# 1. HELPER FUNCTIONS
# ---------------------------------------------------------

# (Removed calculate_angle function as it is no longer needed)

def calculate_distance(point1, point2, image_width, image_height):
    """
    Calculate Euclidean distance between two points in pixels.
    """
    x1, y1 = point1.x * image_width, point1.y * image_height
    x2, y2 = point2.x * image_width, point2.y * image_height
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return int(distance)

# ---------------------------------------------------------
# 2. SETUP & LOAD MODEL
# ---------------------------------------------------------

print("⏳ Loading model...")
try:
    with open('body_language_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'body_language_model.pkl' not found.")
    exit()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ---------------------------------------------------------
# 3. VIDEO CAPTURE
# ---------------------------------------------------------

video_path = 'C:/users/anesr/OneDrive/Pictures/Camera Roll/WIN_20260129_02_57_00_Pro.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Cannot open video.")
    exit()

# Lists for history
all_ml_predictions = []      
# all_slouch_results removed
all_openness_results = []    
all_alignment_results = []   

# ---------------------------------------------------------
# 4. MAIN LOOP
# ---------------------------------------------------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    try:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = image.shape
        
        # (PART A: NECK Removed completely)

        # =========================================================
        # --- PART B: BODY OPENNESS (Normalized Ratio) ---
        # =========================================================
        
        sh_l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        sh_r = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        el_l = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        el_r = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

        # Calculate Distances
        shoulder_width = calculate_distance(sh_l, sh_r, w, h)
        elbow_spread = calculate_distance(el_l, el_r, w, h)

        # Calculate Ratio
        openness_ratio = elbow_spread / (shoulder_width + 1e-6)

        # Apply Thresholds
        if openness_ratio > 1.6:
            open_status = "Open (Power Pose)"
            open_color = (0, 255, 0) # Green
        elif openness_ratio >= 1.2:
            open_status = "Normal (Neutral)"
            open_color = (255, 255, 0) # Yellow
        else:
            open_status = "Closed (Defensive)"
            open_color = (0, 0, 255) # Red

        all_openness_results.append(open_status)
        

        # =========================================================
        # --- PART C: AI MODEL (High Confidence) ---
        # =========================================================
        upper_body = landmarks[:23]
        row = []
        nose = upper_body[0]
        for lm in upper_body:
            rel_x = lm.x - nose.x
            rel_y = lm.y - nose.y
            rel_z = lm.z - nose.z
            row.extend([rel_x, rel_y, rel_z, lm.visibility])
            
        col_names = []
        for i in range(23):
            col_names += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']
        X = pd.DataFrame([row], columns=col_names)

        ml_class = model.predict(X)[0] 
        ml_probabilities = model.predict_proba(X)[0] 
        current_prob = ml_probabilities[np.argmax(ml_probabilities)] 

        final_action = ml_class 
        if ml_class == "Touching Face":
            if current_prob < 0.75: 
                final_action = "Straight" 
        all_ml_predictions.append(final_action)


        # =========================================================
        # --- PART D: SHOULDER ALIGNMENT ---
        # =========================================================
        left_sh_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        right_sh_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y

        shoulder_diff = abs(left_sh_y - right_sh_y)

        if shoulder_diff < 0.04: 
            align_status = "Balanced (Straight)"
            align_color = (0, 255, 0) # Green
        else:
            align_status = "Leaning (Tilted)"
            align_color = (0, 0, 255) # Red
        
        all_alignment_results.append(align_status)


        # =========================================================
        # VISUALIZATION
        # =========================================================
        # Reduced box height (was 200)
        cv2.rectangle(image, (0,0), (350, 170), (245, 117, 16), -1)
        
        # Action (At y=30)
        cv2.putText(image, 'ACTION:', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, final_action.split(' ')[0], (100,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"({current_prob:.2f})", (250,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Neck Removed

        # Body (Openness) - Moved up to y=70
        cv2.putText(image, 'BODY:', (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, open_status.split(' ')[0], (100,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, open_color, 2, cv2.LINE_AA)
        cv2.putText(image, f"Ratio: {openness_ratio:.2f}", (250, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

        # Alignment - Moved up to y=110
        cv2.putText(image, 'ALIGN:', (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, align_status.split(' ')[0], (100,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, align_color, 2, cv2.LINE_AA)
        
        # Draw line connecting shoulders
        ls_px = tuple(np.multiply([landmarks[11].x, landmarks[11].y], [w, h]).astype(int))
        rs_px = tuple(np.multiply([landmarks[12].x, landmarks[12].y], [w, h]).astype(int))
        cv2.line(image, ls_px, rs_px, align_color, 2)

    except Exception as e:
        pass

    cv2.imshow('AI Interview Coach', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------------------------------------------------------
# 6. ADVANCED ANALYTICS (FINAL REPORT)
# ---------------------------------------------------------

print("\n" + "="*60)
print("📊 DETAILED BEHAVIORAL INSIGHTS")
print("="*60)

if len(all_ml_predictions) > 0:
    total_frames = len(all_ml_predictions)
    counts = Counter(all_ml_predictions)

    # --- 1. Face Touching ---
    face_touch_events = 0
    is_touching_now = False
    for action in all_ml_predictions:
        clean_action = action.strip()
        if clean_action == "Touching Face":
            if not is_touching_now:
                face_touch_events += 1 
                is_touching_now = True
        else:
            is_touching_now = False 

    # --- 2. Crossed Arms ---
    crossed_arms_frames = counts.get("Crossed Arms", 0)
    did_cross_arms = "YES" if crossed_arms_frames > 10 else "NO"

    # --- 3. Engagement Score ---
    engagement_frames = counts.get("Explaining", 0) + counts.get("Raised Hand", 0)
    engagement_score = engagement_frames / total_frames
    if engagement_score > 0.3: engagement_label = "High (Active)"
    elif engagement_score > 0.1: engagement_label = "Moderate"
    else: engagement_label = "Low (Passive)"

    # --- PRINT RESULTS ---
    print(f"✋ High-Confidence Face Touches: {face_touch_events} times")
    print(f"🙅 Did Candidate Cross Arms?:    {did_cross_arms}")
    print(f"🔥 ENGAGEMENT SCORE:  {engagement_score:.2f} ({engagement_label})")
    print("-" * 60)

    most_common, freq = counts.most_common(1)[0]
    print(f"🏆 DOMINANT MOVEMENT: {most_common} ({(freq/total_frames)*100:.1f}%)")

    # --- Openness Report ---
    print("\n📊 BODY OPENNESS STATS")
    open_counts = Counter(all_openness_results)
    for k, v in open_counts.most_common():
        print(f"{k}: {(v/total_frames)*100:.1f}%")

    # --- Alignment Report ---
    print("\n⚖️ SHOULDER ALIGNMENT STATS (Posture)")
    align_counts = Counter(all_alignment_results)
    
    straight_frames = align_counts.get("Balanced (Straight)", 0)
    straight_pct = (straight_frames / total_frames) * 100
    
    print(f"Balanced (Straight): {straight_pct:.1f}%")
    print(f"Leaning (Tilted):    {100 - straight_pct:.1f}%")

    if straight_pct > 80:
        print(">> VERDICT: Excellent Posture Stability ✅")
    else:
        print(">> VERDICT: Frequent Leaning Detected (Try to sit straighter) ⚠️")

else:
    print("⚠️ No data collected.")

print("="*60)