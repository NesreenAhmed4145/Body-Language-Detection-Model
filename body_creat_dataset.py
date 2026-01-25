import cv2
import mediapipe as mp
import csv
import numpy as np
import os

# إعداد MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# إعداد ملف الـ CSV
csv_file = 'body_language_dataset.csv'
header = ['label']
# إضافة أسماء الـ 23 نقطة (x, y, z, v) لكل نقطة
for i in range(23): 
    header += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']

# إنشاء الملف وكتابة العناوين
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)

def process_video(video_path, label_name):
    cap = cv2.VideoCapture(video_path)
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # تحويل الألوان لـ RGB لأن MediaPipe يحتاجها
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            # استخراج النقاط (Landmarks)
            landmarks = results.pose_landmarks.landmark
            
            # --- تطبيق منطق المستند ---
            # 1. أخذ أول 23 نقطة فقط (Upper Body)
            upper_body = landmarks[:23]
            
            # 2. تحويل البيانات لصف واحد (Flatten)
            row = [label_name]
            
            # نقطة الأنف (لحساب الـ Relative Coordinates كما في المستند)
            nose = upper_body[0] 
            
            for lm in upper_body:
                # Normalization: طرح إحداثيات الأنف لتصبح الأرقام نسبية
                # (هذا تبسيط لمنطق الـ Scaling المذكور)
                rel_x = lm.x - nose.x
                rel_y = lm.y - nose.y
                rel_z = lm.z - nose.z
                
                row.extend([rel_x, rel_y, rel_z, lm.visibility])
            
            # الحفظ في CSV
            with open(csv_file, mode='a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(row)
            
            count += 1
            
    cap.release()
    print(f"✅ تم استخراج {count} إطار للفئة: {label_name}")


process_video('C:/Users/anesr/OneDrive/Documents/Body Posture and Movement Analysis Model/straight Video.mp4', 'Straight')
process_video('C:/Users/anesr/OneDrive/Documents/Body Posture and Movement Analysis Model/hand_raised_video (1).mp4', 'Raised Hand')
process_video('C:/Users/anesr/OneDrive/Documents/Body Posture and Movement Analysis Model/Explaining Video .mp4', 'Explaining')
process_video('C:/Users/anesr/OneDrive/Documents/Body Posture and Movement Analysis Model/Touching Face_video (1).mp4', 'Touching Face')
process_video('C:/Users/anesr/OneDrive/Documents/Body Posture and Movement Analysis Model/crossed_arms video.mp4', 'Crossed Arms')


