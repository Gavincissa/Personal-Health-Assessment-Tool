import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def estimate_height_from_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if not results.pose_landmarks:
        return None
    
    landmarks = results.pose_landmarks.landmark
    head_to_ankle = abs(landmarks[mp_pose.PoseLandmark.NOSE].y - landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y)
    return round(head_to_ankle * 250)  # crude estimate assuming frame is normalized to ~2.5m

# Example
img_path = 'example.jpg'
estimated_height_cm = estimate_height_from_image(img_path)
print("Estimated height (cm):", estimated_height_cm)
