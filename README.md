# Easy Motion — Virtual Movement Coach

A pose-estimation-based tool that compares a user's exercise movement against a
reference video, scoring similarity per joint and displaying the result as an
annotated side-by-side video.

## Goal

**Business framing:** a virtual coach that lets users check their exercise form
against a reference movement without needing a live trainer, driving engagement
and retention in a fitness app.

**Technical metric:** per-joint similarity score (0-100%), derived from Dynamic
Time Warping distance between joint-angle sequences, averaged into an overall
similarity score.

## Input data format

- Input: two `.mp4` video files (a reference movement and a comparison movement).
- No specific resolution/length is required, but both videos should show the
  full body, reasonably well-lit, with the camera relatively still.
- Output: an annotated `.mp4` video (skeleton overlay + similarity scores) and
  a printed per-joint similarity breakdown.

## Pipeline / data transformations

1. **Pose extraction** — each video frame is processed with MediaPipe Pose
   (`static_image_mode=False`), extracting 33 body landmarks
   (x, y, z, visibility) per frame.
2. **Joint angle calculation** — for 8 key joints (elbows, knees, shoulders,
   hips), the angle formed by three adjacent landmarks is calculated per frame,
   turning raw coordinates into a rotation-invariant, camera-distance-invariant
   signal.
3. **Outlier cleaning** — frame-to-frame angle changes larger than a threshold
   (15°/frame) are treated as tracking glitches and replaced with the previous
   valid value, since a real joint cannot move that fast between frames.
4. **Similarity scoring** — each joint's cleaned angle sequence is compared
   between the two videos using Dynamic Time Warping (DTW), which aligns
   sequences of different lengths/speeds before comparing them. The resulting
   distance is normalized and converted to a 0-100% similarity score.
5. **Visualization** — both videos are re-processed with the MediaPipe skeleton
   drawn on each frame, combined side by side, with the similarity scores
   overlaid as on-screen text.

## Validation

Validated manually by recording two short videos of a similar arm-raise
movement and confirming:
- A video compared against itself produces 0 DTW distance (100% similarity),
  confirming the metric is well-behaved.
- Outlier cleaning correctly removes brief tracking glitches (verified by
  cross-checking against the MediaPipe visibility score) while preserving
  genuine, sustained movement differences between videos.

## Experiments and results

| Joint          | Similarity |
|----------------|------------|
| left_elbow     | 96.8%      |
| right_elbow    | 89.2%      |
| left_knee      | 86.0%      |
| right_knee     | 99.0%      |
| left_shoulder  | 97.6%      |
| right_shoulder | 99.0%      |
| left_hip       | 98.3%      |
| right_hip      | 98.8%      |
| **Overall**    | **95.6%**  |

Model used: MediaPipe Pose (pretrained, `mediapipe==0.10.14`), no custom
training required — pose estimation is performed via the pretrained model,
and the comparison logic (angles + DTW) is the custom component of this
project.

## Project structure