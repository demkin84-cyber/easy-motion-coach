import numpy as np
from fastdtw import fastdtw

JOINT_ANGLE_DEFINITIONS = {
    "left_elbow": (11, 13, 15),
    "right_elbow": (12, 14, 16),
    "left_knee": (23, 25, 27),
    "right_knee": (24, 26, 28),
    "left_shoulder": (13, 11, 23),
    "right_shoulder": (14, 12, 24),
    "left_hip": (11, 23, 25),
    "right_hip": (12, 24, 26),
}


def calculate_angle(a, b, c):
    """Calculate the angle at point b, formed by points a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle


def get_joint_angle_sequence(landmarks_array, point_a_idx, point_b_idx, point_c_idx):
    """Calculate a joint's angle across every frame in a landmarks array."""
    angles = []
    for frame in landmarks_array:
        a = frame[point_a_idx][:2]
        b = frame[point_b_idx][:2]
        c = frame[point_c_idx][:2]
        angles.append(calculate_angle(a, b, c))
    return np.array(angles)


def remove_angle_outliers(angle_sequence, max_change_per_frame=15):
    """Hold the previous value whenever a frame-to-frame jump is implausibly large."""
    cleaned = angle_sequence.copy()
    for i in range(1, len(cleaned)):
        if abs(cleaned[i] - cleaned[i - 1]) > max_change_per_frame:
            cleaned[i] = cleaned[i - 1]
    return cleaned


def compute_all_joint_angles(landmarks_array, joint_definitions=None):
    """Compute cleaned angle sequences for every defined joint."""
    if joint_definitions is None:
        joint_definitions = JOINT_ANGLE_DEFINITIONS

    joint_angles = {}
    for joint_name, (a_idx, b_idx, c_idx) in joint_definitions.items():
        raw_angles = get_joint_angle_sequence(landmarks_array, a_idx, b_idx, c_idx)
        joint_angles[joint_name] = remove_angle_outliers(raw_angles)
    return joint_angles


def simple_dist(a, b):
    return abs(a - b)


def compute_joint_similarity(seq1, seq2, max_possible_diff=180):
    """Compare two angle sequences via DTW, returning a 0-100 similarity score."""
    distance, path = fastdtw(seq1, seq2, dist=simple_dist)
    normalized_distance = distance / len(path)
    similarity = max(0, 100 - (normalized_distance / max_possible_diff * 100))
    return similarity


def compute_all_similarities(joint_angles_1, joint_angles_2):
    """Compare all matching joints between two sets of joint angle sequences."""
    joint_similarities = {}
    for joint_name in joint_angles_1:
        if joint_name in joint_angles_2:
            joint_similarities[joint_name] = compute_joint_similarity(
                joint_angles_1[joint_name], joint_angles_2[joint_name]
            )
    overall_similarity = np.mean(list(joint_similarities.values()))
    return overall_similarity, joint_similarities