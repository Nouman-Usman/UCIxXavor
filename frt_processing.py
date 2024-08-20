import cv2
import mediapipe as mp
import numpy as np
import time
import json

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
initial_position = None
final_position = None
pose_correct = False
pose_start_time = None
measurement_started = False
initial_hip_position = None
initial_knee_position = None
initial_ankle_position = None
initial_foot_height = None
initial_right_arm_position = None
initial_ratios = None
max_distance_cm = 0

video_path = "t5.mp4"

uncertainty = 0.1
scene_width_cm = 100

desired_width = 520
desired_height = 750

# Define the function to calculate the ratio of distances between key landmarks
def calculate_ratios(landmarks):
    left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
    right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
    left_knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y])
    right_knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])
    left_ankle = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])
    right_ankle = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])
    
    hip_to_knee_left = np.linalg.norm(left_hip - left_knee)
    knee_to_ankle_left = np.linalg.norm(left_knee - left_ankle)
    hip_to_knee_right = np.linalg.norm(right_hip - right_knee)
    knee_to_ankle_right = np.linalg.norm(right_knee - right_ankle)
    
    ratio_left = hip_to_knee_left / knee_to_ankle_left if knee_to_ankle_left != 0 else 0
    ratio_right = hip_to_knee_right / knee_to_ankle_right if knee_to_ankle_right != 0 else 0
    
    return ratio_left, ratio_right

def calculate_angle(a, b, c):
    ab = np.array(b) - np.array(a)
    bc = np.array(c) - np.array(b)
    cos_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle


# Define the function to validate the initial posture
def validate_posture(landmarks):
    issues = []

    # Extract required landmarks
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    # Calculate midpoints
    shoulder_midpoint = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]
    hip_midpoint = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]

    # Validate feet position
    if not np.isclose(left_ankle[1], right_ankle[1], atol=0.05):
        issues.append("Feet should be flat on the floor and aligned.")

    # Validate upright posture
    if shoulder_midpoint[0] < hip_midpoint[0] - 0.05 or shoulder_midpoint[0] > hip_midpoint[0] + 0.05:
        issues.append("Shoulders should be aligned with the hips.")

    # Validate arm position (assuming right arm is used for the reach)
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

    if not np.isclose(right_elbow[1], shoulder_midpoint[1], atol=0.1 * (1 + uncertainty)) or right_wrist[1] > right_elbow[1]:
        issues.append("Right arm should be extended forward at shoulder height.")

    return issues

# Define the function to get the right arm position
def get_right_arm_position(landmarks):
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    
    return right_shoulder, right_elbow, right_wrist

def validate_feet_touching_ground(landmarks, initial_foot_height):
    issues = []

    # Extract required landmarks
    left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    left_foot_index_y = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y
    right_foot_index_y = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y

    # Tolerance for considering the feet to be touching the ground
    foot_tolerance = 0.05

    # Validate feet touching the ground
    if abs(left_foot_index_y - left_ankle_y) > foot_tolerance:
        print("Left Foot: ------------ ",abs(left_foot_index_y - left_ankle_y))
        issues.append("Left foot should be touching the ground.")

    if abs(right_foot_index_y - right_ankle_y) > foot_tolerance:
        print("Right Foot: +++++++++++", abs(right_foot_index_y - right_ankle_y))
        issues.append("Right foot should be touching the ground.")

    return issues

# Define the function to validate the lower body posture
def validate_lower_body_posture(landmarks, initial_ratios):
    issues = []

    # # Extract required landmarks
    # left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
    #             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    # right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
    #              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
    right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
    right_shoulder, right_elbow, right_wrist = get_right_arm_position(landmarks)

    # # Validate feet position
    # if not np.isclose(left_ankle[1], right_ankle[1], atol=0.05):
    #     issues.append("Feet should be flat on the floor and aligned.")

    # # Check for feet being raised
    # if left_foot_index[1] < left_ankle[1] - initial_foot_height or right_foot_index[1] < right_ankle[1] - initial_foot_height:
    #     issues.append("Feet should not be raised off the floor.")

    # Validate lower body ratios
    current_ratios = calculate_ratios(landmarks)
    for initial, current in zip(initial_ratios, current_ratios):
        if not np.isclose(initial, current, atol=0.075):
            print("Difference in Ratios: ", (initial - current))
            issues.append("Lower body alignment has changed.")

    # Validate arm position
    shoulder_midpoint = [(right_shoulder[0] + right_elbow[0]) / 2, (right_shoulder[1] + right_elbow[1]) / 2]
    wrist_midpoint = right_wrist  # Right wrist is the target position for validation

    # Check if the elbow is at a similar height as the shoulder
    if not np.isclose(shoulder_midpoint[1], wrist_midpoint[1], atol=0.04):
        issues.append("Arm should be extended forward at shoulder height.")

     # Validate that feet are touching the ground
    feet_issues = validate_feet_touching_ground(landmarks, initial_foot_height)
    issues.extend(feet_issues)


    return issues

# Define the function to calculate distance in cm
def calculate_distance(initial, final, image_width):
    if initial and final:
        distance = np.sqrt((final[0] - initial[0])**2 + (final[1] - initial[1])**2)
        distance_cm = distance * image_width * (scene_width_cm / image_width)
        print(distance_cm)
        return distance_cm
    return 0

def process_frt(video_path,
initial_position = None,
final_position = None,
pose_correct = False,
pose_start_time = None,
measurement_started = False,
initial_hip_position = None,
initial_knee_position = None,
initial_ankle_position = None,
initial_foot_height = None,
initial_right_arm_position = None,
initial_ratios = None,
max_distance_cm = 0,
uncertainty = 0.1,
scene_width_cm = 100,

desired_width = 520,
desired_height = 750):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_height, image_width, _ = frame.shape  # Get image dimensions

        # Resize the frame to the desired width and height
        frame = cv2.resize(frame, (desired_width, desired_height))

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image and detect the pose
        results = pose.process(image)

        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the pose annotation on the image
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Validate posture
            if not pose_correct:
                issues = validate_posture(results.pose_landmarks.landmark)
                if issues:
                    pose_correct = False
                    pose_start_time = None
                    measurement_started = False
                    for i, issue in enumerate(issues):
                        # print(issue)
                        cv2.putText(image, issue, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, "Posture: Incorrect", (10, 50 + len(issues) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    # print("Posture Incorrect +++++++++++++++++++++++++++++++++++++++++")
                else:
                    if pose_start_time is None:
                        pose_start_time = time.time()
                    elif time.time() - pose_start_time >= 2:
                        pose_correct = True
                        initial_position = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        initial_hip_position = [(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x +
                                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2,
                                                (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y +
                                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2]
                        initial_ankle_position = [(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x +
                                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x) / 2,
                                                (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y +
                                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y) / 2]
                        initial_foot_height = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y +
                                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y) / 2
                        initial_right_arm_position = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        initial_ratios = calculate_ratios(results.pose_landmarks.landmark)
                        # print("Initial Ratios: ", initial_ratios)
                        cv2.putText(image, "Posture: Correct", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                        # print("Posture Correct ---------------------------------")

            if pose_correct and initial_hip_position is not None and initial_ankle_position is not None and initial_foot_height is not None:
                lower_body_issues = validate_lower_body_posture(results.pose_landmarks.landmark, initial_ratios)
                if lower_body_issues:
                    pose_correct = False
                    pose_start_time = None
                    measurement_started = False
                    for i, issue in enumerate(lower_body_issues):
                        cv2.putText(image, issue, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                        print(issue)
                    cv2.putText(image, "Posture: Incorrect", (10, 50 + len(lower_body_issues) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Max Distance: {max_distance_cm:.2f} cm", (10, 120 + len(lower_body_issues) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                    # print("Max Distance covered before incorrect lower body posture:", max_distance_cm)
                else:
                    final_position = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    if not measurement_started:
                        measurement_started = True

                    # Calculate and display the distance covered
                    distance_cm = calculate_distance(initial_position, final_position, image_width)
                    if distance_cm > max_distance_cm:
                        max_distance_cm = distance_cm - 1
                    
                    cv2.putText(image, f"Distance: {distance_cm:.2f} cm", (10, 90 + len(lower_body_issues) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Max Distance: {max_distance_cm:.2f} cm", (10, 120 + len(lower_body_issues) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Max Distance: {max_distance_cm:.2f} cm", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            if max_distance_cm >= 25:
                return "Low risk of falls"
            elif 15 <= max_distance_cm < 25:
                return "Risk of falling is 2x greater than normal"
            elif max_distance_cm < 15 and max_distance_cm > 0:
                return "Risk of falling is 4x greater than normal"

        # Display the image
        cv2.imshow('Functional Reach Test', image)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(10) & 0xFF == 27:
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

def live_frt(initial_position = None,
final_position = None,
pose_correct = False,
pose_start_time = None,
measurement_started = False,
initial_hip_position = None,
initial_knee_position = None,
initial_ankle_position = None,
initial_foot_height = None,
initial_right_arm_position = None,
initial_ratios = None,
max_distance_cm = 0,
uncertainty = 0.1,
scene_width_cm = 100,

desired_width = 520,
desired_height = 750):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_height, image_width, _ = frame.shape  # Get image dimensions

        # Resize the frame to the desired width and height
        frame = cv2.resize(frame, (desired_width, desired_height))

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image and detect the pose
        results = pose.process(image)

        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the pose annotation on the image
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Validate posture
            if not pose_correct:
                issues = validate_posture(results.pose_landmarks.landmark)
                if issues:
                    pose_correct = False
                    pose_start_time = None
                    measurement_started = False
                    for i, issue in enumerate(issues):
                        # print(issue)
                        cv2.putText(image, issue, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, "Posture: Incorrect", (10, 50 + len(issues) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    # print("Posture Incorrect +++++++++++++++++++++++++++++++++++++++++")
                else:
                    if pose_start_time is None:
                        pose_start_time = time.time()
                    elif time.time() - pose_start_time >= 2:
                        pose_correct = True
                        initial_position = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        initial_hip_position = [(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x +
                                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2,
                                                (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y +
                                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2]
                        initial_ankle_position = [(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x +
                                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x) / 2,
                                                (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y +
                                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y) / 2]
                        initial_foot_height = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y +
                                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y) / 2
                        initial_right_arm_position = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        initial_ratios = calculate_ratios(results.pose_landmarks.landmark)
                        # print("Initial Ratios: ", initial_ratios)
                        cv2.putText(image, "Posture: Correct", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                        # print("Posture Correct ---------------------------------")

            if pose_correct and initial_hip_position is not None and initial_ankle_position is not None and initial_foot_height is not None:
                lower_body_issues = validate_lower_body_posture(results.pose_landmarks.landmark, initial_ratios)
                if lower_body_issues:
                    pose_correct = False
                    pose_start_time = None
                    measurement_started = False
                    for i, issue in enumerate(lower_body_issues):
                        cv2.putText(image, issue, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                        print(issue)
                    cv2.putText(image, "Posture: Incorrect", (10, 50 + len(lower_body_issues) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Max Distance: {max_distance_cm:.2f} cm", (10, 120 + len(lower_body_issues) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                    # print("Max Distance covered before incorrect lower body posture:", max_distance_cm)
                else:
                    final_position = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    if not measurement_started:
                        measurement_started = True

                    # Calculate and display the distance covered
                    distance_cm = calculate_distance(initial_position, final_position, image_width)
                    if distance_cm > max_distance_cm:
                        max_distance_cm = distance_cm - 1
                    
                    cv2.putText(image, f"Distance: {distance_cm:.2f} cm", (10, 90 + len(lower_body_issues) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Max Distance: {max_distance_cm:.2f} cm", (10, 120 + len(lower_body_issues) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Max Distance: {max_distance_cm:.2f} cm", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            if max_distance_cm >= 25:
                return "Low risk of falls"
            elif 15 <= max_distance_cm < 25:
                return "Risk of falling is 2x greater than normal"
            elif max_distance_cm < 15 and max_distance_cm > 0:
                return "Risk of falling is 4x greater than normal"
            # elif max_distance_cm == 0:  # Assuming 0 indicates unwillingness to reach
            #     return "Unwilling to reach: Risk of falling is 8x greater than normal"
            # else:
            #     return "Invalid distance"

        # Display the image
        cv2.imshow('Functional Reach Test', image)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(10) & 0xFF == 27:
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    results_data = {"Max Distance (cm)": max_distance_cm, "Date and Time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open("results.json", "w") as f:
        json.dump(results_data, f)


def check_frt_need(symptom_responses, symptom_criteria):
    for symptom in symptom_criteria:
        question = symptom['question']
        if symptom_responses.get(question, False):
            return True
    return False

