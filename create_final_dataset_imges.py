import cv2
import mediapipe as mp
import csv
import os
import glob

# 1. Setup MediaPipe
mp_pose = mp.solutions.pose
# static_image_mode=True is very important for higher accuracy with images
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)

# 2. New Dataset Filename
csv_file = 'body_language_dataset_imges.csv'

# 3. Setup Column Headers
header = ['label']
for i in range(23): # 23 points for the upper body
    header += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']

# Create the file and write the headers
with open(csv_file, mode='w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(header)

def create_dataset_from_folders(class_folders):
    """
    class_folders: Dictionary containing the class label and its folder path
    Example: {'Straight': 'path/to/straight_images', ...}
    """
    
    total_count = 0
    
    for class_name, folder_path in class_folders.items():
        print(f"🔄 Reading folder: {class_name}...")
        
        # Read all image types
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
            
        print(f"   📂 Found {len(image_paths)} images.")
        
        class_count = 0
        
        with open(csv_file, mode='a', newline='') as f:
            csv_writer = csv.writer(f)
            
            for img_path in image_paths:
                # Read the image
                image = cv2.imread(img_path)
                if image is None: continue
                
                # Convert colors and extract landmarks
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    upper_body = landmarks[:23] # Take upper body only
                    
                    # Prepare the row
                    row = [class_name]
                    nose = upper_body[0] # Reference point
                    
                    for lm in upper_body:
                        # Calculate Relative Coordinates
                        rel_x = lm.x - nose.x
                        rel_y = lm.y - nose.y
                        rel_z = lm.z - nose.z
                        
                        row.extend([rel_x, rel_y, rel_z, lm.visibility])
                    
                    csv_writer.writerow(row)
                    class_count += 1
                    
        print(f"   ✅ Successfully extracted data for {class_count} images for {class_name}")
        total_count += class_count

    print(f"\n🎉 Done! Total rows in dataset: {total_count}")
    print(f"📁 File saved as: {csv_file}")

# --- 🚀 Modify paths here ---
# Put the class name (as you want it in the result) and the path to the folder containing its images (augmented)
my_folders = {
    'Straight': r'C:/Users/anesr/Downloads/bodyModel/dataset_final/Straight',
    'Crossed Arms': r'C:/Users/anesr/Downloads\bodyModel/dataset_final/Crossed Arm',
    'Touching Face': r'C:/Users/anesr/Downloads\bodyModel/dataset_final/Touching Face',
    'Explaining': r'C:/Users/anesr/Downloads\bodyModel/dataset_final/Explaining',
    'Raised Hand': r'C:/Users/anesr/Downloads\bodyModel/dataset_final/Raised Hand'
}

# Run the function
create_dataset_from_folders(my_folders)