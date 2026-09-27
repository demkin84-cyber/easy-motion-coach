import sys
import os
import argparse

import mediapipe as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from easymotion.video_utils import (
    extract_landmarks_from_video,
    landmarks_to_array,
    process_video_with_drawing,
    create_comparison_video,
)
from easymotion.pose_utils import (
    compute_all_joint_angles,
    compute_all_similarities,
)

mp_pose = mp.solutions.pose


def compare_videos(video_path_1, video_path_2, output_path="comparison_output.mp4"):
    """Compare two videos' movements and produce a side-by-side output with similarity scores."""

    print(f"Processing video 1: {video_path_1}")
    pose_1 = mp_pose.Pose(static_image_mode=False)
    landmarks_seq_1 = extract_landmarks_from_video(video_path_1, pose_1)
    landmarks_array_1 = landmarks_to_array(landmarks_seq_1)

    print(f"Processing video 2: {video_path_2}")
    pose_2 = mp_pose.Pose(static_image_mode=False)
    landmarks_seq_2 = extract_landmarks_from_video(video_path_2, pose_2)
    landmarks_array_2 = landmarks_to_array(landmarks_seq_2)

    print("Computing joint angles...")
    joint_angles_1 = compute_all_joint_angles(landmarks_array_1)
    joint_angles_2 = compute_all_joint_angles(landmarks_array_2)

    print("Computing similarity scores...")
    overall_similarity, joint_similarities = compute_all_similarities(joint_angles_1, joint_angles_2)

    print(f"\nOverall similarity: {overall_similarity:.1f}%")
    for joint_name, score in joint_similarities.items():
        print(f"  {joint_name}: {score:.1f}%")

    print("\nGenerating comparison video...")
    pose_draw_1 = mp_pose.Pose(static_image_mode=False)
    pose_draw_2 = mp_pose.Pose(static_image_mode=False)
    frames_1 = process_video_with_drawing(video_path_1, pose_draw_1)
    frames_2 = process_video_with_drawing(video_path_2, pose_draw_2)

    saved_path = create_comparison_video(frames_1, frames_2, overall_similarity, joint_similarities, output_path)
    print(f"Saved comparison video to: {saved_path}")

    return overall_similarity, joint_similarities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two exercise videos using pose estimation.")
    parser.add_argument("video1", help="Path to the reference video")
    parser.add_argument("video2", help="Path to the video to compare against the reference")
    parser.add_argument("--output", default="comparison_output.mp4", help="Path to save the comparison video")
    args = parser.parse_args()

    compare_videos(args.video1, args.video2, args.output)