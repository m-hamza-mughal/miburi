import librosa
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.pyplot import figure
import math
import smplx
from loguru import logger
from scipy import linalg
from scipy.signal import argrelextrema
from collections import OrderedDict
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp

from .tools import load_checkpoints
from .metric_utils.motion_representation import VAESKConv

from .rotation_conversions import *

class L1div(object):
    def __init__(self):
        self.counter = 0
        self.sum = 0
    def run(self, results):
        self.counter += results.shape[0]
        mean = np.mean(results, 0)
        for i in range(results.shape[0]):
            results[i, :] = abs(results[i, :] - mean)
        sum_l1 = np.sum(results)
        self.sum += sum_l1
    def avg(self):
        return self.sum/self.counter
    def reset(self):
        self.counter = 0
        self.sum = 0
        

class SRGR(object):
    def __init__(self, threshold=0.1, joints=47):
        self.threshold = threshold
        self.pose_dimes = joints
        self.counter = 0
        self.sum = 0
        
    def run(self, results, targets, semantic):
        results = results.reshape(-1, self.pose_dimes, 3)
        targets = targets.reshape(-1, self.pose_dimes, 3)
        semantic = semantic.reshape(-1)
        diff = np.sum(abs(results-targets),2)
        success = np.where(diff<self.threshold, 1.0, 0.0)
        for i in range(success.shape[0]):
            # srgr == 0.165 when all success, scale range to [0, 1]
            success[i, :] *= semantic[i] * (1/0.165) 
        rate = np.sum(success)/(success.shape[0]*success.shape[1])
        self.counter += success.shape[0]
        self.sum += (rate*success.shape[0])
        return rate
    
    def avg(self):
        return self.sum/self.counter

class alignment(object):
    def __init__(self, sigma, order, mmae=None, upper_body=[3,6,9,12,13,14,15,16,17,18,19,20,21]):
        self.sigma = sigma
        self.order = order
        self.upper_body= upper_body
        # self.times = self.oenv = self.S = self.rms = None
        self.pose_data = []
        self.mmae = mmae
        self.threshold = 0.3
    
    def load_audio(self, wave, t_start=None, t_end=None, without_file=False, sr_audio=16000):
        hop_length = 512
        if without_file:
            y = wave
            sr = sr_audio
        else: y, sr = librosa.load(wave)
        if t_start is None:
            short_y = y
        else:
            short_y = y[t_start:t_end]
        # print(short_y.shape)
        onset_t = librosa.onset.onset_detect(y=short_y, sr=sr_audio, hop_length=hop_length, units='time')
        return onset_t

    def load_pose(self, pose, t_start, t_end, pose_fps, without_file=False):
        data_each_file = []
        if without_file:
            for line_data_np in pose: #,args.pre_frames, args.pose_length
                data_each_file.append(line_data_np)
                    #data_each_file.append(np.concatenate([line_data_np[9:18], line_data_np[75:84], ],0))
        else: 
            with open(pose, "r") as f:
                for i, line_data in enumerate(f.readlines()):
                    if i < 432: continue
                    line_data_np = np.fromstring(line_data, sep=" ",)
                    if pose_fps == 15:
                        if i % 2 == 0:
                            continue
                    data_each_file.append(np.concatenate([line_data_np[30:39], line_data_np[112:121], ],0))
                    
        data_each_file = np.array(data_each_file)
        #print(data_each_file.shape)
        
        joints = data_each_file.transpose(1, 0)
        dt = 1/pose_fps
        # first steps is forward diff (t+1 - t) / dt
        init_vel = (joints[:, 1:2] - joints[:, :1]) / dt
        # middle steps are second order (t+1 - t-1) / 2dt
        middle_vel = (joints[:, 2:] - joints[:, 0:-2]) / (2 * dt)
        # last step is backward diff (t - t-1) / dt
        final_vel = (joints[:, -1:] - joints[:, -2:-1]) / dt
        #print(joints.shape, init_vel.shape, middle_vel.shape, final_vel.shape)
        vel = np.concatenate([init_vel, middle_vel, final_vel], 1).transpose(1, 0).reshape(data_each_file.shape[0], -1, 3)
        #print(vel.shape)
        #vel = data_each_file.reshape(data_each_file.shape[0], -1, 3)[1:] - data_each_file.reshape(data_each_file.shape[0], -1, 3)[:-1]
        vel = np.linalg.norm(vel, axis=2) / self.mmae
        
        beat_vel_all = []
        for i in range(vel.shape[1]):
            vel_mask = np.where(vel[:, i]>self.threshold)
            #print(vel.shape)
            #t_end = 80
            #vel[::2, :] -= 0.000001
            #print(vel[t_start:t_end, i], vel[t_start:t_end, i].shape)
            beat_vel = argrelextrema(vel[t_start:t_end, i], np.less, order=self.order) # n*47
            #print(beat_vel, t_start, t_end)
            beat_vel_list = []
            for j in beat_vel[0]:
                if j in vel_mask[0]:
                    beat_vel_list.append(j)
            beat_vel = np.array(beat_vel_list)
            beat_vel_all.append(beat_vel)
        #print(beat_vel_all)
        return beat_vel_all #beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist
    
    
    def load_data(self, wave, pose, t_start, t_end, pose_fps):
        onset_raw, onset_bt, onset_bt_rms = self.load_audio(wave, t_start, t_end)
        beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = self.load_pose(pose, t_start, t_end, pose_fps)
        return onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist 

    def eval_random_pose(self, wave, pose, t_start, t_end, pose_fps, num_random=60):
        onset_raw, onset_bt, onset_bt_rms = self.load_audio(wave, t_start, t_end)
        dur = t_end - t_start
        for i in range(num_random):
            beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = self.load_pose(pose, i, i+dur, pose_fps)
            dis_all_b2a= self.calculate_align(onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist)
            print(f"{i}s: ",dis_all_b2a)


    @staticmethod
    def plot_onsets(audio, sr, onset_times_1, onset_times_2):
        import librosa
        import librosa.display
        import matplotlib.pyplot as plt
        # Plot audio waveform
        fig, axarr = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Plot audio waveform in both subplots
        librosa.display.waveshow(audio, sr=sr, alpha=0.7, ax=axarr[0])
        librosa.display.waveshow(audio, sr=sr, alpha=0.7, ax=axarr[1])
        
        # Plot onsets from first method on the first subplot
        for onset in onset_times_1:
            axarr[0].axvline(onset, color='r', linestyle='--', alpha=0.9, label='Onset Method 1')
        axarr[0].legend()
        axarr[0].set(title='Onset Method 1', xlabel='', ylabel='Amplitude')
        
        # Plot onsets from second method on the second subplot
        for onset in onset_times_2:
            axarr[1].axvline(onset, color='b', linestyle='-', alpha=0.7, label='Onset Method 2')
        axarr[1].legend()
        axarr[1].set(title='Onset Method 2', xlabel='Time (s)', ylabel='Amplitude')
    
        
        # Add legend (eliminate duplicate labels)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        # Show plot
        plt.title("Audio waveform with Onsets")
        plt.savefig("./onset.png", dpi=500)
    
    def audio_beat_vis(self, onset_raw, onset_bt, onset_bt_rms):
        figure(figsize=(24, 6), dpi=80)
        fig, ax = plt.subplots(nrows=4, sharex=True)
        librosa.display.specshow(librosa.amplitude_to_db(self.S, ref=np.max),
                                y_axis='log', x_axis='time', ax=ax[0])
        ax[0].label_outer()
        ax[1].plot(self.times, self.oenv, label='Onset strength')
        ax[1].vlines(librosa.frames_to_time(onset_raw), 0, self.oenv.max(), label='Raw onsets', color='r')
        ax[1].legend()
        ax[1].label_outer()

        ax[2].plot(self.times, self.oenv, label='Onset strength')
        ax[2].vlines(librosa.frames_to_time(onset_bt), 0, self.oenv.max(), label='Backtracked', color='r')
        ax[2].legend()
        ax[2].label_outer()

        ax[3].plot(self.times, self.rms[0], label='RMS')
        ax[3].vlines(librosa.frames_to_time(onset_bt_rms), 0, self.oenv.max(), label='Backtracked (RMS)', color='r')
        ax[3].legend()
        fig.savefig("./onset.png", dpi=500)
    
    @staticmethod
    def motion_frames2time(vel, offset, pose_fps):
        time_vel = vel/pose_fps + offset 
        return time_vel    
    
    @staticmethod
    def GAHR(a, b, sigma):
        dis_all_a2b = 0
        dis_all_b2a = 0
        for b_each in b:
            l2_min = np.inf
            for a_each in a:
                l2_dis = abs(a_each - b_each)
                if l2_dis < l2_min:
                    l2_min = l2_dis
            dis_all_b2a += math.exp(-(l2_min**2)/(2*sigma**2))
        dis_all_b2a /= len(b)
        return dis_all_b2a 
    
    @staticmethod
    def fix_directed_GAHR(a, b, sigma):
        a = alignment.motion_frames2time(a, 0, 30)
        b = alignment.motion_frames2time(b, 0, 30)
        t = len(a)/30
        a = [0] + a + [t]
        b = [0] + b + [t]
        dis_a2b = alignment.GAHR(a, b, sigma)
        return dis_a2b

    def calculate_align(self, onset_bt_rms, beat_vel, pose_fps=30):
        audio_bt = onset_bt_rms
        avg_dis_all_b2a_list = []
        for its, beat_vel_each in enumerate(beat_vel):
            if its not in self.upper_body:
                continue
            #print(beat_vel_each)
            #print(audio_bt.shape, beat_vel_each.shape)
            pose_bt = self.motion_frames2time(beat_vel_each, 0, pose_fps)
            #print(pose_bt)
            avg_dis_all_b2a_list.append(self.GAHR(pose_bt, audio_bt, self.sigma))
        # avg_dis_all_b2a = max(avg_dis_all_b2a_list)
        avg_dis_all_b2a = sum(avg_dis_all_b2a_list)/len(avg_dis_all_b2a_list) #max(avg_dis_all_b2a_list)
        #print(avg_dis_all_b2a, sum(avg_dis_all_b2a_list)/47)
        return avg_dis_all_b2a  
    

class FIDCalculator(object):

    @staticmethod
    def frechet_distance(samples_A, samples_B):
        A_mu = np.mean(samples_A, axis=0)
        A_sigma = np.cov(samples_A, rowvar=False)
        B_mu = np.mean(samples_B, axis=0)
        B_sigma = np.cov(samples_B, rowvar=False)
        try:
            frechet_dist = FIDCalculator.calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
        except ValueError:
            frechet_dist = 1e+10
        return frechet_dist


    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                    inception net (like returned by the function 'get_predictions')
                    for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                    representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                    representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        #print(mu1[0], mu2[0])
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        #print(sigma1[0], sigma2[0])
        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        #print(diff, covmean[0])
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                    'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)


def load_fid_checkpoints(model, save_path, load_name='model'):
    states = torch.load(save_path)
    new_weights = OrderedDict()
    flag=False
    for k, v in states['model_state'].items():
        #print(k)
        if "module" not in k:
            break
        else:
            new_weights[k[7:]]=v
            flag=True
    if flag: 
        try:
            model.load_state_dict(new_weights)
        except:
            try:
                model.load_state_dict(states['model_state'])
            except:
                try:
                    model.load_state_dict(new_weights, strict=False)
                    print(f"Loaded {load_name} with strict=False")
                    return model
                except Exception as e:
                    print(f"Failed to load checkpoint: {str(e)}")
                    raise e
    else:
        model.load_state_dict(states['model_state'])
    logger.info(f"load self-pretrained checkpoints for {load_name}")


def match_motion_fps(data, source_fps=25, target_fps=30):
    """
    data: (n_totalframes, num_joints, 3) - numpy array
    return: (new_num_frames, num_joints, 3) - numpy array
    """
    n_totalframes, num_joints, _ = data.shape
    duration = n_totalframes / source_fps  # Total motion duration in seconds
    new_num_frames = int(np.round(duration * target_fps))  # Compute new frame count

    # Original and new time indices
    original_times = np.arange(0, n_totalframes)
    new_times = np.arange(0, n_totalframes, source_fps/target_fps)
    n_totaljoints = num_joints

    if new_times[-1] > original_times[-1]:
        new_times = new_times[:-1]

    # breakpoint()
    pose_data_aa = torch.from_numpy(data)# .reshape(n_totalframes, n_totaljoints, 3)
    pose_data_rot = axis_angle_to_matrix(pose_data_aa).numpy()

    pose_data_interp_perjoint = []
    for j in range(n_totaljoints):
        pose_data_rot_j = R.from_matrix(pose_data_rot[:, j])
        slerp = Slerp(original_times, pose_data_rot_j)
        pose_data_rot_j_interp = slerp(new_times)
        pose_data_rot_j_interp = torch.from_numpy(pose_data_rot_j_interp.as_matrix())
        pose_data_interp_perjoint.append(pose_data_rot_j_interp)

    # breakpoint()
    pose_data_rot_interp = torch.stack(pose_data_interp_perjoint, dim=1)
    pose_data_aa_interp = matrix_to_axis_angle(pose_data_rot_interp)
    smplx_fullbodypose_interp = pose_data_aa_interp.numpy()

    return smplx_fullbodypose_interp
    

def match_fps_linear(data, source_fps=25, target_fps=30):
    n_totalframes = data.shape[0]
    duration = n_totalframes / source_fps  # Total motion duration in seconds
    new_num_frames = int(np.round(duration * target_fps))  # Compute new frame count

    # Original and new time indices
    original_times = np.arange(0, n_totalframes)
    new_times = np.arange(0, n_totalframes, source_fps/target_fps)
    # n_totaljoints = num_joints

    if new_times[-1] > original_times[-1]:
        new_times = new_times[:-1]

    lerp_trans = interp1d(original_times, data, axis=0)
    trans_interp = lerp_trans(new_times)
    new_data = trans_interp

    return new_data


def _smplx_forward_chunked(
    smplx_model,
    batched_kwargs: dict,
    *,
    return_verts: bool = False,
    return_joints: bool = False,
    chunk_size: int = 64,
) -> dict:
    """Run SMPL-X forward in chunks along the batch dim.

    The shape-blendshape einsum inside SMPL-X's forward materialises a
    `B * V * 3 * (num_betas + num_expression)` intermediate (~50 MB per
    frame at num_betas=300, num_expression=100). At B=300 that's ~15 GB,
    which OOMs a 24 GB card. Chunking the batch dim keeps each
    sub-forward inside a sensible memory budget regardless of how long
    the test clip is.

    `batched_kwargs` is a mapping of name -> tensor; each tensor is
    sliced as `v[start:end]` per chunk. The returned dict contains the
    requested output fields (`"vertices"` and/or `"joints"`) stacked
    back along dim 0 as `.detach()`ed tensors.
    """
    # Discover total batch size from any batched kwarg.
    n_total = None
    for v in batched_kwargs.values():
        if torch.is_tensor(v) and v.ndim >= 1:
            n_total = v.shape[0]
            break
    if n_total is None or n_total == 0:
        out = smplx_model(
            **batched_kwargs,
            return_verts=return_verts,
            return_joints=return_joints,
        )
        result = {}
        if return_verts:
            result["vertices"] = out["vertices"].detach()
        if return_joints:
            result["joints"] = out["joints"].detach()
        return result

    vert_pieces, joint_pieces = [], []
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        sub = {k: v[start:end] for k, v in batched_kwargs.items()}
        out = smplx_model(
            **sub,
            return_verts=return_verts,
            return_joints=return_joints,
        )
        if return_verts:
            vert_pieces.append(out["vertices"].detach())
        if return_joints:
            joint_pieces.append(out["joints"].detach())

    result = {}
    if return_verts:
        result["vertices"] = torch.cat(vert_pieces, dim=0)
    if return_joints:
        result["joints"] = torch.cat(joint_pieces, dim=0)
    return result


def _smplx_face_vertices_chunked(
    smplx_model,
    betas,
    expression,
    jaw_pose,
    leye_pose,
    reye_pose,
    chunk_size: int = 64,
):
    """Face-only SMPL-X forward (body / hands / global pose zeroed),
    returning `(B, V, 3)` vertex tensors. Thin wrapper around
    `_smplx_forward_chunked` for the metric facial L2 / L-Vel paths."""
    B = betas.shape[0]
    device = betas.device
    dtype = betas.dtype
    zero3 = torch.zeros((B, 3), device=device, dtype=dtype)
    zero45 = torch.zeros((B, 45), device=device, dtype=dtype)   # 15 joints * 3
    zero63 = torch.zeros((B, 63), device=device, dtype=dtype)   # 21 joints * 3
    return _smplx_forward_chunked(
        smplx_model,
        dict(
            betas=betas,
            transl=zero3,
            expression=expression,
            jaw_pose=jaw_pose,
            global_orient=zero3,
            body_pose=zero63,
            left_hand_pose=zero45,
            right_hand_pose=zero45,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
        ),
        return_verts=True,
        chunk_size=chunk_size,
    )["vertices"]


class GestureMetrics:
    def __init__(self, trainer_args):

        # ---- BEATX vs Embody3D eval-config toggle ----
        dataset_ratio = (getattr(trainer_args, "dataset_ratio", "") or "").lower()
        if "beatx" in dataset_ratio:
            eval_source = "BEATX"
        elif "embody" in dataset_ratio:
            eval_source = "Embody3D"
        else:
            eval_source = "BEATX"
        logger.info(
            f"GestureMetrics: using {eval_source} eval config "
            f"(dataset_ratio={dataset_ratio!r})"
        )

        eval_args = type('Args', (), {})()
        eval_args.vae_length = 240
        eval_args.vae_test_dim = 330
        eval_args.variational = False
        eval_args.data_path_1 = "assets_dep/smplx_2020/"

        if eval_source == "BEATX":
            self.avg_vel = np.load("datasets/beat_english_v2.0.0/weights/mean_vel_smplxflame_30.npy")
            eval_args.vae_layer = 4
            eval_args.vae_grow = [1, 1, 2, 1]
            self.vae_test_len = 32
            eval_weight_path = "datasets/beat_english_v2.0.0/weights/AESKConv_240_100.bin"
        else:
            self.avg_vel = np.load("datasets/embody_3d_aiagent/beatxweights/mean_vel_smplx_30.npy")
            eval_args.vae_layer = 2
            eval_args.vae_grow = [1, 2]
            self.vae_test_len = 64
            eval_weight_path = "datasets/embody_3d_aiagent/beatxweights/last_13.bin"

        self.alignmenter = alignment(0.3, 7, self.avg_vel, upper_body=[3,6,9,12,13,14,15,16,17,18,19,20,21])
        self.align_mask = 60
        self.gt_alignmenter = alignment(0.3, 7, self.avg_vel, upper_body=[3,6,9,12,13,14,15,16,17,18,19,20,21])
        self.l1_calculator = L1div()
        self.gt_l1_calculator = L1div()

        self.vae_length = eval_args.vae_length

        eval_copy = VAESKConv(eval_args).cuda()
        load_fid_checkpoints(
            eval_copy,
            eval_weight_path,
            'VAESKConv',
        )
        self.eval_copy = eval_copy

        self.trainer_args = trainer_args

        self.reclatent_loss = torch.nn.MSELoss()
        self.vel_loss = torch.nn.L1Loss(reduction="mean")


        self.pred_latents = []
        self.gt_latents = []

        self.facial_l2_all = 0.0
        self.facial_lvel = 0.0

        self.align = 0.0
        self.gt_align = 0.0

        self.dataset_framelen = 0
        self.dataset_len = 0


        self.eval_audio_sr = 16000
        self.eval_posefps = 30


        self.smplx = smplx.create(
            "assets_dep/smplx_2020/",
            model_type="smplx",
            gender="NEUTRAL_2020",
            flat_hand_mean=True,
            num_betas=300,
            num_expression_coeffs=100,
            use_pca=False,
            ext="npz",
        ).cuda().eval()

    def reset(self):
        self.pred_latents = []
        self.gt_latents = []

        self.facial_l2_all = 0.0
        self.facial_lvel = 0.0

        self.align = 0.0
        self.gt_align = 0.0

        self.dataset_framelen = 0
        self.dataset_len = 0

        self.l1_calculator.reset()
        self.gt_l1_calculator.reset()


    def update(self, input_dict):
        """
        dict is of:
        {
            "rec_pose": rec_pose,
            "rec_exps": rec_exps,
            "rec_trans": rec_trans,
            "tar_pose": tar_pose,
            "tar_exps": tar_exps,
            "tar_beta": tar_beta,
            "tar_trans": tar_trans,
            "file_id": file_id,
        }

        """
        rec_pose_aa = input_dict["rec_pose"]
        tar_pose_aa = input_dict["tar_pose"]
        rec_exps = input_dict["rec_exps"]
        tar_exps = input_dict["tar_exps"]
        rec_trans = input_dict["rec_trans"]
        tar_trans = input_dict["tar_trans"]
        tar_beta = input_dict["tar_beta"]
        file_id = input_dict["file_id"]

        bs = 1
        n, j3 = rec_pose_aa.shape
        j = j3 // 3 
        assert j == 55, "Expected 55 joints in SMPLX pose"
        rec_pose = rec_pose_aa
        tar_pose = tar_pose_aa

        

        rec_pose = rec_pose.reshape(bs*n, j, 3).cpu().numpy()
        tar_pose = tar_pose.reshape(bs*n, j, 3).cpu().numpy()
        rec_pose = match_motion_fps(rec_pose, source_fps=25, target_fps=30)
        tar_pose = match_motion_fps(tar_pose, source_fps=25, target_fps=30)   
        new_n = rec_pose.shape[0] 
        rec_pose = torch.from_numpy(rec_pose).reshape(new_n, j*3)
        tar_pose = torch.from_numpy(tar_pose).reshape(new_n, j*3)

        tar_exps = tar_exps.reshape(bs*n, 100).cpu().numpy()
        tar_exps = match_fps_linear(tar_exps, source_fps=25, target_fps=30)
        rec_exps = rec_exps.reshape(bs*n, 100).cpu().numpy()
        rec_exps = match_fps_linear(rec_exps, source_fps=25, target_fps=30)
        assert rec_exps.shape[0] == new_n and tar_exps.shape[0] == new_n, "Expression FPS matching error"
        rec_exps = torch.from_numpy(rec_exps).reshape(new_n, 100)
        tar_exps = torch.from_numpy(tar_exps).reshape(new_n, 100)

        tar_trans = tar_trans.reshape(bs*n, 3).cpu().numpy()
        tar_trans = match_fps_linear(tar_trans, source_fps=25, target_fps=30)
        rec_trans = rec_trans.reshape(bs*n, 3).cpu().numpy()
        rec_trans = match_fps_linear(rec_trans, source_fps=25, target_fps=30)
        assert rec_trans.shape[0] == new_n and tar_trans.shape[0] == new_n, "Translation FPS matching error"
        rec_trans = torch.from_numpy(rec_trans).reshape(new_n, 3)
        tar_trans = torch.from_numpy(tar_trans).reshape(new_n, 3)
        
        # breakpoint()
        # for numpy array
        # tar_beta = tar_beta[None, :].repeat(new_n, 0)
        # # breakpoint()
        # tar_beta = torch.from_numpy(tar_beta).cuda().float()
        
        tar_beta = tar_beta[None, :].expand(new_n, -1)
        # breakpoint()
        tar_beta = tar_beta.cuda().float()
        

        n = new_n


        rec_pose[:, 66:69] = tar_pose.reshape(bs * n, 55 * 3)[:, 66:69]

        rec_pose = axis_angle_to_matrix(rec_pose.reshape(bs * n, j, 3))
        rec_pose = matrix_to_rotation_6d(rec_pose).reshape(bs, n, j * 6)
        tar_pose = axis_angle_to_matrix(tar_pose.reshape(bs * n, j, 3))
        tar_pose = matrix_to_rotation_6d(tar_pose).reshape(bs, n, j * 6)

        # breakpoint()

        rec_pose = rec_pose.cuda().float()
        tar_pose = tar_pose.cuda().float()
        tar_exps = tar_exps.cuda().float()
        rec_exps = rec_exps.cuda().float()
        tar_beta = tar_beta.cuda().float()
        rec_trans = rec_trans.cuda().float()
        tar_trans = tar_trans.cuda().float()

        remain = n % self.vae_test_len
        latent_out = self.eval_copy.map2latent(rec_pose[:, : n - remain]).reshape(-1, self.vae_length).detach().cpu().numpy()
        
        latent_ori = self.eval_copy.map2latent(tar_pose[:, : n - remain]).reshape(-1, self.vae_length).detach().cpu().numpy()
        
        self.pred_latents.append(latent_out)
        self.gt_latents.append(latent_ori)

        rec_pose = rotation_6d_to_matrix(rec_pose.reshape(bs * n, j, 6))
        rec_pose = matrix_to_axis_angle(rec_pose).reshape(bs * n, j * 3)
        tar_pose = rotation_6d_to_matrix(tar_pose.reshape(bs * n, j, 6))
        tar_pose = matrix_to_axis_angle(tar_pose).reshape(bs * n, j * 3)

        # Body-pose forward via the chunked helper: shape-blendshape
        # intermediates would OOM a 24 GB card at full n; chunking the
        # batch dim keeps each sub-forward inside ~3-4 GB.
        zero3_b = torch.zeros((bs * n, 3), device=tar_beta.device, dtype=tar_beta.dtype)
        zero100_b = torch.zeros((bs * n, 100), device=tar_beta.device, dtype=tar_beta.dtype)
        joints_rec = _smplx_forward_chunked(
            self.smplx,
            dict(
                betas=tar_beta.reshape(bs * n, 300),
                transl=zero3_b,
                expression=zero100_b,
                jaw_pose=rec_pose[:, 66:69],
                global_orient=rec_pose[:, :3],
                body_pose=rec_pose[:, 3 : 21 * 3 + 3],
                left_hand_pose=rec_pose[:, 25 * 3 : 40 * 3],
                right_hand_pose=rec_pose[:, 40 * 3 : 55 * 3],
                leye_pose=rec_pose[:, 69:72],
                reye_pose=rec_pose[:, 72:75],
            ),
            return_joints=True,
        )["joints"].cpu().numpy().reshape(bs, n, 127 * 3)[0, :n, : 55 * 3]

        joints_tar = _smplx_forward_chunked(
            self.smplx,
            dict(
                betas=tar_beta.reshape(bs * n, 300),
                transl=zero3_b,
                expression=zero100_b,
                jaw_pose=tar_pose[:, 66:69],
                global_orient=tar_pose[:, :3],
                body_pose=tar_pose[:, 3 : 21 * 3 + 3],
                left_hand_pose=tar_pose[:, 25 * 3 : 40 * 3],
                right_hand_pose=tar_pose[:, 40 * 3 : 55 * 3],
                leye_pose=tar_pose[:, 69:72],
                reye_pose=tar_pose[:, 72:75],
            ),
            return_joints=True,
        )["joints"].cpu().numpy().reshape(bs, n, 127 * 3)[0, :n, : 55 * 3]

        _ = self.l1_calculator.run(joints_rec)
        _ = self.gt_l1_calculator.run(joints_tar)
        
        # Audio path layout depends on which dataset the sample came from.
        # Embody3D dyadic ids look like "<smpid>+<spkid>" and live at
        #   <embody3d_path>/<smpid>/<spkid>/audio_separated/<smpid>.wav
        # BEATX ids are "<speaker>_<name>_<...>" (no "+") and live at
        #   <beatx_data_path>/wave16k/<file_id>.wav
        # build_hdf5_beatx.py stores file_id as the BASE id (no _C<chunk>
        # suffix), which is what the trainer's dict_data["file_id"]
        # carries through, so it's a direct lookup.
        if "+" in file_id:
            smpid, spkid = file_id.split("+")[:2]
            audio_path = os.path.join(
                self.trainer_args.embody3d_path,
                smpid, spkid, "audio_separated", f"{smpid}.wav",
            )
        else:
            audio_path = os.path.join(
                self.trainer_args.beatx_data_path, "wave16k", f"{file_id}.wav",
            )
        in_audio_eval, sr = librosa.load(audio_path)
        in_audio_eval = librosa.resample(
            in_audio_eval, orig_sr=sr, target_sr=self.eval_audio_sr
        )
        a_offset = int(
            self.align_mask
            * (self.eval_audio_sr / self.eval_posefps)
        )
        onset_bt = self.alignmenter.load_audio(
            in_audio_eval[
                : int(self.eval_audio_sr / self.eval_posefps * n)
            ],
            a_offset,
            len(in_audio_eval) - a_offset,
            True,
        )
        beat_vel = self.alignmenter.load_pose(
            joints_rec, self.align_mask, n - self.align_mask, 30, True
        )
        self.align += self.alignmenter.calculate_align(
            onset_bt, beat_vel, 30
        ) * (n - 2 * self.align_mask)

        gt_beat_vel = self.gt_alignmenter.load_pose(
            joints_tar, self.align_mask, n - self.align_mask, 30, True
        )
        self.gt_align += self.gt_alignmenter.calculate_align(
            onset_bt, gt_beat_vel, 30
        ) * (n - 2 * self.align_mask)

        # SMPL-X face-only forward, chunked along the batch dim to keep
        # the shape-blendshape intermediate inside the 24 GB GPU budget.
        verts_rec_face = _smplx_face_vertices_chunked(
            self.smplx,
            betas=tar_beta.reshape(bs * n, 300),
            expression=rec_exps.reshape(bs * n, 100),
            jaw_pose=rec_pose[:, 66:69],
            leye_pose=rec_pose[:, 69:72],
            reye_pose=rec_pose[:, 72:75],
        )
        verts_tar_face = _smplx_face_vertices_chunked(
            self.smplx,
            betas=tar_beta.reshape(bs * n, 300),
            expression=tar_exps.reshape(bs * n, 100),
            jaw_pose=tar_pose[:, 66:69],
            leye_pose=tar_pose[:, 69:72],
            reye_pose=tar_pose[:, 72:75],
        )
        facial_rec = verts_rec_face.reshape(1, n, -1)[0, :n].cpu()
        facial_tar = verts_tar_face.reshape(1, n, -1)[0, :n].cpu()
        face_vel_loss = self.vel_loss(
            facial_rec[1:, :] - facial_tar[:-1, :],
            facial_tar[1:, :] - facial_tar[:-1, :],
        )
        l2 = self.reclatent_loss(facial_rec, facial_tar)
        self.facial_l2_all += l2.item() * n
        self.facial_lvel += face_vel_loss.item() * n

        self.dataset_framelen += n
        self.dataset_len += 1

    def compute_metrics(self):
        # If every update() call raised (e.g. file_id format doesn't match
        # what self.alignmenter expects -- BEATX vs. embody3d_dyadic),
        # `dataset_framelen` stays at 0. Bail out with a clear log message
        # instead of dividing by zero on the next line.
        if self.dataset_framelen == 0:
            logger.warning(
                "GestureMetrics: no samples were successfully processed "
                "(check earlier 'Error in metrics calculation for ...' logs). "
                "Skipping metric report."
            )
            return

        l2_avg = self.facial_l2_all / self.dataset_framelen
        lvel_avg = self.facial_lvel / self.dataset_framelen

        logger.info(f"Facial L2: {l2_avg}")
        logger.info(f"Facial L-Vel: {lvel_avg}")


        latent_out_all = np.concatenate(self.pred_latents, axis=0)
        latent_ori_all = np.concatenate(self.gt_latents, axis=0)
        fgd = FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        logger.info(f"fgd score: {fgd}")

        align_avg = self.align / (self.dataset_framelen - 2 * self.dataset_len * self.align_mask)
        logger.info(f"align score: {align_avg}")

        gt_align_avg = self.gt_align / (self.dataset_framelen - 2 * self.dataset_len * self.align_mask)
        logger.info(f"gt align score: {gt_align_avg}")

        l1div = self.l1_calculator.avg()
        logger.info(f"L1div score: {l1div}")

        gt_l1div = self.gt_l1_calculator.avg()
        logger.info(f"GT L1div score: {gt_l1div}")
        
class MPJPE:
    def __init__(self):
        self.total_error = 0.0
        self.total_joints = 0

    def compute_error(self, predicted, ground_truth, mask=None):
        """
        Compute MPJPE for a single pose or a batch of poses.

        Parameters:
        - predicted: numpy array of shape (N, J, 3) where N is the number of poses,
                     J is the number of joints, and 3 corresponds to (x, y, z) coordinates.
        - ground_truth: numpy array of shape (N, J, 3), the ground-truth joint positions.
        - mask: numpy array of shape (N, J, 1) with 0s for joints that are not visible and 1s for visible joints

        Returns:
        - mpjpe: the mean per joint position error for the given batch
        """
        # Ensure input arrays are numpy arrays
        predicted = np.asarray(predicted)
        ground_truth = np.asarray(ground_truth)

        # Compute the Euclidean distance for each joint
        error = np.linalg.norm(predicted - ground_truth, axis=-1)  # shape: (N, J)

        # Mask out joints that are not visible
        if mask is not None:
            error *= mask

        mpjpe = np.mean(error)  # Mean over all joints and all poses

        # Accumulate the total error and joint counts
        self.total_error += np.sum(error)
        self.total_joints += error.size

        return mpjpe

    def get_average_error(self):
        """
        Get the average MPJPE over all poses processed so far.

        Returns:
        - average_mpjpe: the average MPJPE across all accumulated data
        """
        if self.total_joints == 0:
            return 0.0
        return self.total_error / self.total_joints

    def reset(self):
        """
        Reset the accumulated error and joint counts.
        """
        self.total_error = 0.0
        self.total_joints = 0
 
        
class ReconMetrics:
    def __init__(self, trainer_args):

        # ---- BEATX vs Embody3D eval-config toggle ----
        # Same convention as GestureMetrics.__init__: "beatx" in
        # dataset_ratio picks the BEATX FID net, "embody" picks the
        # Embody3D one, unknown / empty defaults to BEATX. Codec trainers
        # all set `dataset_ratio` in their YAML (e.g. `full_beatx`,
        # `33embody_66beatx`), so the right branch fires automatically.
        dataset_ratio = (getattr(trainer_args, "dataset_ratio", "") or "").lower()
        if "beatx" in dataset_ratio:
            eval_source = "BEATX"
        elif "embody" in dataset_ratio:
            eval_source = "Embody3D"
        else:
            eval_source = "BEATX"
        logger.info(
            f"ReconMetrics: using {eval_source} eval config "
            f"(dataset_ratio={dataset_ratio!r})"
        )

        eval_args = type('Args', (), {})()
        eval_args.vae_length = 240
        eval_args.vae_test_dim = 330
        eval_args.variational = False
        eval_args.data_path_1 = "assets_dep/smplx_2020/"

        if eval_source == "BEATX":
            eval_args.vae_layer = 4
            eval_args.vae_grow = [1, 1, 2, 1]
            self.vae_test_len = 32
            eval_weight_path = "datasets/beat_english_v2.0.0/weights/AESKConv_240_100.bin"
        else:
            eval_args.vae_layer = 2
            eval_args.vae_grow = [1, 2]
            self.vae_test_len = 64
            eval_weight_path = "datasets/embody_3d_aiagent/beatxweights/last_13.bin"

        self.vae_length = eval_args.vae_length

        eval_copy = VAESKConv(eval_args).cuda()
        load_fid_checkpoints(
            eval_copy,
            eval_weight_path,
            'VAESKConv',
        )
        self.eval_copy = eval_copy

        self.trainer_args = trainer_args

        self.rec_mpjpe = MPJPE()

        self.pred_latents = []
        self.gt_latents = []

        # self.mpjpe = 0.0

        self.dataset_framelen = 0
        self.dataset_len = 0

        # Facial vertex metrics, mirroring GestureMetrics. Both are
        # frame-weighted by `n` in update() and divided by
        # `self.dataset_framelen` in compute_metrics() to get a per-frame
        # average. For non-face codec trainers `rec_exps == tar_exps` so
        # both accumulators stay at zero -- that's expected.
        self.reclatent_loss = torch.nn.MSELoss()
        self.vel_loss = torch.nn.L1Loss(reduction="mean")
        self.facial_l2_all = 0.0
        self.facial_lvel = 0.0


        self.eval_audio_sr = 16000
        self.eval_posefps = 30


        self.smplx = smplx.create(
            "assets_dep/smplx_2020/",
            model_type='smplx',
            gender='NEUTRAL_2020',
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100,
            ext='npz',
            use_pca=False,
        ).cuda().eval()


    def update(self, input_dict):
        """
        dict is of:
        {
            "rec_pose": rec_pose,
            "rec_exps": rec_exps,
            "rec_trans": rec_trans,
            "tar_pose": tar_pose,
            "tar_exps": tar_exps,
            "tar_beta": tar_beta,
            "tar_trans": tar_trans,
            "file_id": file_id,
        }

        """
        rec_pose_aa = input_dict["rec_pose"]
        tar_pose_aa = input_dict["tar_pose"]
        rec_exps = input_dict["rec_exps"]
        tar_exps = input_dict["tar_exps"]
        rec_trans = input_dict["rec_trans"]
        tar_trans = input_dict["tar_trans"]
        tar_beta = input_dict["tar_beta"]
        file_id = input_dict["file_id"]

        bs = 1
        n, j3 = rec_pose_aa.shape
        j = j3 // 3 
        assert j == 55, "Expected 55 joints in SMPLX pose"
        rec_pose = rec_pose_aa
        tar_pose = tar_pose_aa

        

        rec_pose = rec_pose.reshape(bs*n, j, 3).cpu().numpy()
        tar_pose = tar_pose.reshape(bs*n, j, 3).cpu().numpy()
        rec_pose = match_motion_fps(rec_pose, source_fps=25, target_fps=30)
        tar_pose = match_motion_fps(tar_pose, source_fps=25, target_fps=30)   
        new_n = rec_pose.shape[0] 
        rec_pose = torch.from_numpy(rec_pose).reshape(new_n, j*3)
        tar_pose = torch.from_numpy(tar_pose).reshape(new_n, j*3)

        tar_exps = tar_exps.reshape(bs*n, 100).cpu().numpy()
        tar_exps = match_fps_linear(tar_exps, source_fps=25, target_fps=30)
        rec_exps = rec_exps.reshape(bs*n, 100).cpu().numpy()
        rec_exps = match_fps_linear(rec_exps, source_fps=25, target_fps=30)
        assert rec_exps.shape[0] == new_n and tar_exps.shape[0] == new_n, "Expression FPS matching error"
        rec_exps = torch.from_numpy(rec_exps).reshape(new_n, 100)
        tar_exps = torch.from_numpy(tar_exps).reshape(new_n, 100)

        tar_trans = tar_trans.reshape(bs*n, 3).cpu().numpy()
        tar_trans = match_fps_linear(tar_trans, source_fps=25, target_fps=30)
        rec_trans = rec_trans.reshape(bs*n, 3).cpu().numpy()
        rec_trans = match_fps_linear(rec_trans, source_fps=25, target_fps=30)
        assert rec_trans.shape[0] == new_n and tar_trans.shape[0] == new_n, "Translation FPS matching error"
        rec_trans = torch.from_numpy(rec_trans).reshape(new_n, 3)
        tar_trans = torch.from_numpy(tar_trans).reshape(new_n, 3)
        
        # breakpoint()
        # for numpy array
        # tar_beta = tar_beta[None, :].repeat(new_n, 0)
        # # breakpoint()
        # tar_beta = torch.from_numpy(tar_beta).cuda().float()
        
        tar_beta = tar_beta[None, :].expand(new_n, -1)
        # breakpoint()
        tar_beta = tar_beta.cuda().float()
        

        n = new_n


        rec_pose[:, 66:69] = tar_pose.reshape(bs * n, 55 * 3)[:, 66:69]

        rec_pose = axis_angle_to_matrix(rec_pose.reshape(bs * n, j, 3))
        rec_pose = matrix_to_rotation_6d(rec_pose).reshape(bs, n, j * 6)
        tar_pose = axis_angle_to_matrix(tar_pose.reshape(bs * n, j, 3))
        tar_pose = matrix_to_rotation_6d(tar_pose).reshape(bs, n, j * 6)

        # breakpoint()

        rec_pose = rec_pose.cuda().float()
        tar_pose = tar_pose.cuda().float()
        tar_exps = tar_exps.cuda().float()
        rec_exps = rec_exps.cuda().float()
        tar_beta = tar_beta.cuda().float()
        rec_trans = rec_trans.cuda().float()
        tar_trans = tar_trans.cuda().float()

        remain = n % self.vae_test_len
        latent_out = self.eval_copy.map2latent(rec_pose[:, : n - remain]).reshape(-1, self.vae_length).detach().cpu().numpy()
        
        latent_ori = self.eval_copy.map2latent(tar_pose[:, : n - remain]).reshape(-1, self.vae_length).detach().cpu().numpy()
        
        self.pred_latents.append(latent_out)
        self.gt_latents.append(latent_ori)

        rec_pose = rotation_6d_to_matrix(rec_pose.reshape(bs * n, j, 6))
        rec_pose = matrix_to_axis_angle(rec_pose).reshape(bs * n, j * 3)
        tar_pose = rotation_6d_to_matrix(tar_pose.reshape(bs * n, j, 6))
        tar_pose = matrix_to_axis_angle(tar_pose).reshape(bs * n, j * 3)

        # Body-pose forward via the chunked helper to dodge the
        # shape-blendshape OOM on long clips. See
        # _smplx_forward_chunked for the size math.
        zero3_b = torch.zeros((bs * n, 3), device=tar_beta.device, dtype=tar_beta.dtype)
        zero100_b = torch.zeros((bs * n, 100), device=tar_beta.device, dtype=tar_beta.dtype)
        joints_rec = _smplx_forward_chunked(
            self.smplx,
            dict(
                betas=tar_beta.reshape(bs * n, 300),
                transl=zero3_b,
                expression=zero100_b,
                jaw_pose=rec_pose[:, 66:69],
                global_orient=rec_pose[:, :3],
                body_pose=rec_pose[:, 3 : 21 * 3 + 3],
                left_hand_pose=rec_pose[:, 25 * 3 : 40 * 3],
                right_hand_pose=rec_pose[:, 40 * 3 : 55 * 3],
                leye_pose=rec_pose[:, 69:72],
                reye_pose=rec_pose[:, 72:75],
            ),
            return_joints=True,
        )["joints"].cpu().numpy().reshape(bs, n, 127 * 3)[0, :n, : 55 * 3]

        joints_tar = _smplx_forward_chunked(
            self.smplx,
            dict(
                betas=tar_beta.reshape(bs * n, 300),
                transl=zero3_b,
                expression=zero100_b,
                jaw_pose=tar_pose[:, 66:69],
                global_orient=tar_pose[:, :3],
                body_pose=tar_pose[:, 3 : 21 * 3 + 3],
                left_hand_pose=tar_pose[:, 25 * 3 : 40 * 3],
                right_hand_pose=tar_pose[:, 40 * 3 : 55 * 3],
                leye_pose=tar_pose[:, 69:72],
                reye_pose=tar_pose[:, 72:75],
            ),
            return_joints=True,
        )["joints"].cpu().numpy().reshape(bs, n, 127 * 3)[0, :n, : 55 * 3]

        # breakpoint()

        mpjpe = self.rec_mpjpe.compute_error(joints_rec.reshape(n, 55, 3), joints_tar.reshape(n, 55, 3))

        # --- Facial vertex L2 + vertex velocity L2 ---
        # SMPL-X face-only forward, chunked to keep the shape-blendshape
        # intermediate inside the GPU memory budget. See
        # _smplx_face_vertices_chunked for the size math.
        verts_rec_face = _smplx_face_vertices_chunked(
            self.smplx,
            betas=tar_beta.reshape(bs * n, 300),
            expression=rec_exps.reshape(bs * n, 100),
            jaw_pose=rec_pose[:, 66:69],
            leye_pose=rec_pose[:, 69:72],
            reye_pose=rec_pose[:, 72:75],
        )
        verts_tar_face = _smplx_face_vertices_chunked(
            self.smplx,
            betas=tar_beta.reshape(bs * n, 300),
            expression=tar_exps.reshape(bs * n, 100),
            jaw_pose=tar_pose[:, 66:69],
            leye_pose=tar_pose[:, 69:72],
            reye_pose=tar_pose[:, 72:75],
        )
        facial_rec = verts_rec_face.reshape(1, n, -1)[0, :n].cpu()
        facial_tar = verts_tar_face.reshape(1, n, -1)[0, :n].cpu()
        face_vel_loss = self.vel_loss(
            facial_rec[1:, :] - facial_tar[:-1, :],
            facial_tar[1:, :] - facial_tar[:-1, :],
        )
        face_l2 = self.reclatent_loss(facial_rec, facial_tar)
        self.facial_l2_all += face_l2.item() * n
        self.facial_lvel += face_vel_loss.item() * n

        self.dataset_framelen += n
        self.dataset_len += 1

    def compute_metrics(self):
        


        latent_out_all = np.concatenate(self.pred_latents, axis=0)
        latent_ori_all = np.concatenate(self.gt_latents, axis=0)
        fgd = FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        logger.info(f"fgd score: {fgd}")
        avg_mpjpe = self.rec_mpjpe.get_average_error()
        logger.info(f"Reconstruction MPJPE: {avg_mpjpe}")
        # Facial vertex L2 + vertex velocity L2. Zero for upper/lower
        # codec runs (rec_exps == tar_exps); meaningful for the face codec.
        if self.dataset_framelen > 0:
            facial_l2_avg = self.facial_l2_all / self.dataset_framelen
            facial_lvel_avg = self.facial_lvel / self.dataset_framelen
            logger.info(f"Facial L2: {facial_l2_avg}")
            logger.info(f"Facial L-Vel: {facial_lvel_avg}")
        

        
