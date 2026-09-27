import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def extract_landmarks_from_video(video_path, pose_model):
    """Run pose detection on every frame of a video, returning a list of landmark sets."""
    cap = cv2.VideoCapture(video_path)
    all_landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_model.process(frame_rgb)

        if results.pose_landmarks:
            frame_landmarks = [(lm.x, lm.y, lm.z, lm.visibility) for lm in results.pose_landmarks.landmark]
        else:
            frame_landmarks = None

        all_landmarks.append(frame_landmarks)

    cap.release()
    return all_landmarks


def landmarks_to_array(landmarks_sequence, num_landmarks=33):
    """Convert a list of per-frame landmark tuples into a single (frames, landmarks, 4) array."""
    num_frames = len(landmarks_sequence)
    arr = np.zeros((num_frames, num_landmarks, 4))

    for i, frame_landmarks in enumerate(landmarks_sequence):
        if frame_landmarks is not None:
            arr[i] = np.array(frame_landmarks)

    return arr


def process_video_with_drawing(video_path, pose_model):
    """Run pose detection on every frame and return frames with the skeleton drawn on."""
    cap = cv2.VideoCapture(video_path)
    frames_with_landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_model.process(frame_rgb)

        annotated = frame_rgb.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        frames_with_landmarks.append(annotated)

    cap.release()
    return frames_with_landmarks


def create_comparison_video(frames_v1, frames_v2, overall_similarity, joint_similarities, output_path="comparison_output.mp4"):
    """Combine two sets of annotated frames side by side, with similarity scores overlaid."""
    num_frames = min(len(frames_v1), len(frames_v2))
    h, w = frames_v1[0].shape[:2]

    combined_width = w * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (combined_width, h))

    for i in range(num_frames):
        frame1 = cv2.cvtColor(frames_v1[i], cv2.COLOR_RGB2BGR)
        frame2 = cv2.cvtColor(frames_v2[i], cv2.COLOR_RGB2BGR)

        combined = np.hstack([frame1, frame2])

        cv2.putText(combined, f"Overall Similarity: {overall_similarity:.1f}%",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        y_offset = 80
        for joint_name, score in joint_similarities.items():
            cv2.putText(combined, f"{joint_name}: {score:.1f}%",
                        (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25

        out.write(combined)

    out.release()
    return output_path