import cv2
import os
import numpy as np
from moviepy.editor import VideoFileClip


def compute_optical_flow(video_path, save_folder):
    # Computes the mean optical flow for a rendered video
    # Used for the mean motion comparisons on the DaMM dataset
    clip = VideoFileClip(video_path)
    frames = clip.iter_frames()
    prev_frame = next(frames)
    hsv = np.zeros_like(prev_frame)
    hsv[..., 1] = 255
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_frame = np.reshape(prev_frame, prev_frame.shape + (1,))

    current_index = 0
    mean_face = np.zeros(prev_frame[..., 0].shape)
    for frame in frames:
        curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_frame = np.reshape(curr_frame, curr_frame.shape + (1,))

        flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        rgb = hsv[..., 2]
        mean_face += rgb

        prev_frame = curr_frame
        current_index += 1

    mean_face /= current_index
    cv2.imwrite(os.path.join(save_folder, f'mean_diff.png'), mean_face)