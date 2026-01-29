import cv2
import mediapipe as mp
import numpy as np

# 1. Math Functions
def calculate_distance(point1, point2, image_width, image_height):
    """
    Calculate Euclidean distance between two points in pixels.
    """
    x1, y1 = point1.x * image_width, point1.y * image_height
    x2, y2 = point2.x * image_width, point2.y * image_height
    
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return int(distance)

def calculate_angle(a, b, c):
    """
    Calculate angle between three points (a, b, c).
    """
    a = np.array(a) # Point 1
    b = np.array(b) # Vertex
    c = np.array(c) # Point 3
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return int(angle)

# 2. Setup MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ---------------------------------------------------------
# 3. VIDEO PATH CONFIGURATION
# ---------------------------------------------------------
# Put your video path here
video_path = 'C:/Users/anesr/Downloads/Recording 2026-01-29 021610.mp4'

# Open the video file instead of camera (0)
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"❌ Error: Cannot open video at {video_path}")
    print("Please check the file path and try again.")
    exit()

print(f"✅ Video loaded: {video_path}")

# 4. Main Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # Resize frame if it's too big (Optional, helps with speed)
    # frame = cv2.resize(frame, (800, 600))

    # Color conversion
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get dimensions
    h, w, _ = image.shape

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # ---------------------------------------------------------
        # (A) Extract Points
        # ---------------------------------------------------------
        # Right Side
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Left Side
        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

        # ---------------------------------------------------------
        # (B) Calculate Distances (Elbow <-> Hip)
        # ---------------------------------------------------------
        dist_r_elbow_hip = calculate_distance(r_elbow, r_hip, w, h)
        dist_l_elbow_hip = calculate_distance(l_elbow, l_hip, w, h)

        # ---------------------------------------------------------
        # (C) Calculate Angles (Shoulder -> Elbow -> Wrist)
        # ---------------------------------------------------------
        r_angle = calculate_angle(
            [r_shoulder.x, r_shoulder.y], 
            [r_elbow.x, r_elbow.y], 
            [r_wrist.x, r_wrist.y]
        )
        
        l_angle = calculate_angle(
            [l_shoulder.x, l_shoulder.y], 
            [l_elbow.x, l_elbow.y], 
            [l_wrist.x, l_wrist.y]
        )

        # ---------------------------------------------------------
        # (D) Visualization
        # ---------------------------------------------------------
        
        # Draw Skeleton
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # --- LEFT SIDE VISUALS ---
        # Draw line between Elbow and Hip
        cv2.line(image, 
                 (int(l_elbow.x * w), int(l_elbow.y * h)), 
                 (int(l_hip.x * w), int(l_hip.y * h)), 
                 (255, 0, 0), 2) 
        
        # Text for Distance
        cv2.putText(image, f"Dist: {dist_l_elbow_hip}", 
                    (int(l_elbow.x * w) - 80, int(l_elbow.y * h)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Text for Angle
        cv2.putText(image, f"Ang: {l_angle}", 
                    (int(l_elbow.x * w) - 80, int(l_elbow.y * h) - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)


        # --- RIGHT SIDE VISUALS ---
        cv2.line(image, 
                 (int(r_elbow.x * w), int(r_elbow.y * h)), 
                 (int(r_hip.x * w), int(r_hip.y * h)), 
                 (255, 0, 0), 2) 
        
        cv2.putText(image, f"Dist: {dist_r_elbow_hip}", 
                    (int(r_elbow.x * w) + 10, int(r_elbow.y * h)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.putText(image, f"Ang: {r_angle}", 
                    (int(r_elbow.x * w) + 10, int(r_elbow.y * h) - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # --- Top Info Bar ---
        cv2.rectangle(image, (0,0), (600, 40), (245, 117, 16), -1)
        cv2.putText(image, f"Left Elbow-Hip: {dist_l_elbow_hip} px | Right Elbow-Hip: {dist_r_elbow_hip} px", 
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Display Frame
    cv2.imshow('Distance & Angle Calculator', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()