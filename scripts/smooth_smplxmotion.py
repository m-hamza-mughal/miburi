# import logging
from loguru import logger
import sys
import time
import warnings
import os
import random
import torch
import numpy as np
import torch

import trainers
import trainers.utils.rotation_conversions as rc

def smoothen_smplx_motion(motion_file):
    # load the npz data
    npz_data = np.load(motion_file)
    motion=npz_data['poses']
    # breakpoint()

    motion = torch.tensor(motion, dtype=torch.float32)

    # convert the motion to 6d
    motion = rc.axis_angle_to_matrix(motion.reshape(-1, 55, 3))
    motion = rc.matrix_to_rotation_6d(motion) # convert to 6d representation # frames, joints, 6

    # define a filt to smooth the motion
    filt = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    filt /= np.sum(filt)
    filt = torch.tensor(filt, dtype=torch.float32).view(1, 1, 5)  # shape (1, 1, 5)

    frames, joints, dim = motion.shape
    motion_reshaped = motion.permute(1, 2, 0).reshape(joints * dim, 1, frames)  # (joints*6, 1, frames)
    motion_smoothed = torch.nn.functional.conv1d(motion_reshaped, filt, padding=filt.shape[-1] // 2)
    motion = motion_smoothed.reshape(joints, dim, frames).permute(2, 0, 1)  # back to (frames, joints, 6)
    motion = rc.rotation_6d_to_matrix(motion)  # convert back to matrix representation
    motion = rc.matrix_to_axis_angle(motion).reshape(-1, 55, 3)  # convert back to axis-angle

    # save the smoothed motion
    np.savez(motion_file.replace('.npz', '_smoothed.npz'),
        betas=np.zeros(300,),
        poses=motion,
        expressions=np.zeros(300,),
        trans=np.zeros((motion.shape[0], 3)),
        model='smplx2020',
        gender='NEUTRAL_2020',
        mocap_frame_rate = 30, #self.args.motion_fps ,
    )


def select_upper_body_joints(motion_file):
    # load the npz data
    npz_data = np.load(motion_file)
    motion = npz_data['poses'][:822*2]

    upper_mask =[0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    upper_mask = np.array(upper_mask, dtype=np.float32)
    upper_mask = torch.tensor(upper_mask, dtype=torch.float32)
    upper_mask = upper_mask.repeat_interleave(3)  # repeat each joint's mask 3 times for 3D representation
    # breakpoint()
    motion = torch.tensor(motion, dtype=torch.float32)
    motion = motion * upper_mask  # apply the mask to the motion

    # save the masked motion
    np.savez(motion_file.replace('.npz', '_upper.npz'),
        betas=np.zeros(300,),
        poses=motion,
        expressions=np.zeros(300,),
        trans=np.zeros((motion.shape[0], 3)),
        model='smplx2020',
        gender='NEUTRAL_2020',
        mocap_frame_rate = 30, #self.args.motion_fps ,
    )

if __name__ == "__main__":
    # motion_file = "/CT/GestureSynth1/work/GestureMoshi/moshi/moshi/results/vis/codebook_motion_225.npz"
    # smoothen_smplx_motion(motion_file)

    motion_file = "/CT/GestureSynth1/work/GestureMoshi/moshi/moshi/results/vis/2_scott_0_100_100.npz"
    select_upper_body_joints(motion_file)
