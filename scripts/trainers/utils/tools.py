import os
import numpy as np
import random
import torch
import sys
import pprint
import pandas as pd
from loguru import logger
from collections import OrderedDict
import hashlib
import yaml
import safetensors
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from .distributed import get_rank, get_world_size

# logger = logging.getLogger()

def map2color(s):
    m = hashlib.md5()
    m.update(s.encode('utf-8'))
    color_code = m.hexdigest()[:6]
    return '#' + color_code

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

def adjust_array(x, k):
    len_x = len(x)
    len_k = len(k)

    # If x is shorter than k, pad with zeros
    if len_x < len_k:
        return np.pad(x, (0, len_k - len_x), 'constant')

    # If x is longer than k, truncate x
    elif len_x > len_k:
        return x[:len_k]

    # If both are of same length
    else:
        return x

def onset_to_frame(onset_times, audio_length, fps):
    # Calculate total number of frames for the given audio length
    total_frames = int(audio_length * fps)
    
    # Create an array of zeros of shape (total_frames,)
    frame_array = np.zeros(total_frames, dtype=np.int32)
    
    # For each onset time, calculate the frame number and set it to 1
    for onset in onset_times:
        frame_num = int(onset * fps)
        # Check if the frame number is within the array bounds
        if 0 <= frame_num < total_frames:
            frame_array[frame_num] = 1
    
    return frame_array


def smooth_rotvec_animations(animation1, animation2, blend_frames):
    """
    Smoothly transition between two animation clips using SLERP.

    Parameters:
    - animation1: The first animation clip, a numpy array of shape [n, k].
    - animation2: The second animation clip, a numpy array of shape [n, k].
    - blend_frames: Number of frames over which to blend the two animations.

    Returns:
    - A smoothly blended animation clip of shape [2n, k].
    """
    
    # Ensure blend_frames doesn't exceed the length of either animation
    n1, k1 = animation1.shape
    n2, k2 = animation2.shape
    animation1 = animation1.reshape(n1, k1//3, 3)
    animation2 = animation2.reshape(n2, k2//3, 3)
    blend_frames = min(blend_frames, len(animation1), len(animation2))
    all_int = []
    for i in range(k1//3):
        # Convert rotation vectors to quaternion for the overlapping part
        q = R.from_rotvec(np.concatenate([animation1[0:1, i], animation2[-2:-1, i]], axis=0))#.as_quat()
        # q2 = R.from_rotvec()#.as_quat()
        times = [0, blend_frames * 2 - 1]
        slerp = Slerp(times, q)
        interpolated = slerp(np.arange(blend_frames * 2)) 
        interpolated_rotvecs = interpolated.as_rotvec()
        all_int.append(interpolated_rotvecs)
    interpolated_rotvecs = np.concatenate(all_int, axis=1)
    # result = np.vstack((animation1[:-blend_frames], interpolated_rotvecs, animation2[blend_frames:]))
    result = interpolated_rotvecs.reshape(2*n1, k1)
    return result

def smooth_animations(animation1, animation2, blend_frames):
    """
    Smoothly transition between two animation clips using linear interpolation.

    Parameters:
    - animation1: The first animation clip, a numpy array of shape [n, k].
    - animation2: The second animation clip, a numpy array of shape [n, k].
    - blend_frames: Number of frames over which to blend the two animations.

    Returns:
    - A smoothly blended animation clip of shape [2n, k].
    """
    
    # Ensure blend_frames doesn't exceed the length of either animation
    blend_frames = min(blend_frames, len(animation1), len(animation2))
    
    # Extract overlapping sections
    overlap_a1 = animation1[-blend_frames:-blend_frames+1, :]
    overlap_a2 = animation2[blend_frames-1:blend_frames, :]
    
    # Create blend weights for linear interpolation
    alpha = np.linspace(0, 1, 2 * blend_frames).reshape(-1, 1)
    
    # Linearly interpolate between overlapping sections
    blended_overlap = overlap_a1 * (1 - alpha) + overlap_a2 * alpha
    
    # Extend the animations to form the result with 2n frames
    if blend_frames == len(animation1) and blend_frames == len(animation2):
        result = blended_overlap
    else:
        before_blend = animation1[:-blend_frames]
        after_blend = animation2[blend_frames:]
        result = np.vstack((before_blend, blended_overlap, after_blend))
    return result

def interpolate_sequence(quaternions):
    bs, n, j, _ = quaternions.shape
    new_n = 2 * n
    new_quaternions = torch.zeros((bs, new_n, j, 4), device=quaternions.device, dtype=quaternions.dtype)

    for i in range(n):
        q1 = quaternions[:, i, :, :]
        new_quaternions[:, 2*i, :, :] = q1

        if i < n - 1:
            q2 = quaternions[:, i + 1, :, :]
            new_quaternions[:, 2*i + 1, :, :] = slerp(q1, q2, 0.5)
        else:
            # For the last point, duplicate the value
            new_quaternions[:, 2*i + 1, :, :] = q1

    return new_quaternions

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z

def quaternion_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def slerp(q1, q2, t):
    dot = torch.sum(q1 * q2, dim=-1, keepdim=True)

    flip = (dot < 0).float()
    q2 = (1 - flip * 2) * q2
    dot = dot * (1 - flip * 2)

    DOT_THRESHOLD = 0.9995
    mask = (dot > DOT_THRESHOLD).float()

    theta_0 = torch.acos(dot)
    theta = theta_0 * t

    q3 = q2 - q1 * dot
    q3 = q3 / torch.norm(q3, dim=-1, keepdim=True)

    interpolated = (torch.cos(theta) * q1 + torch.sin(theta) * q3)

    return mask * (q1 + t * (q2 - q1)) + (1 - mask) * interpolated

def estimate_linear_velocity(data_seq, dt):
    '''
    Given some batched data sequences of T timesteps in the shape (B, T, ...), estimates
    the velocity for the middle T-2 steps using a second order central difference scheme.
    The first and last frames are with forward and backward first-order
    differences, respectively
    - h : step size
    '''
    # first steps is forward diff (t+1 - t) / dt
    init_vel = (data_seq[:, 1:2] - data_seq[:, :1]) / dt
    # middle steps are second order (t+1 - t-1) / 2dt
    middle_vel = (data_seq[:, 2:] - data_seq[:, 0:-2]) / (2 * dt)
    # last step is backward diff (t - t-1) / dt
    final_vel = (data_seq[:, -1:] - data_seq[:, -2:-1]) / dt

    vel_seq = torch.cat([init_vel, middle_vel, final_vel], dim=1)
    return vel_seq

def velocity2position(data_seq, dt, init_pos):
    res_trans = []
    for i in range(data_seq.shape[1]):
        if i == 0:
            res_trans.append(init_pos.unsqueeze(1))
        else:
            res = data_seq[:, i-1:i] * dt + res_trans[-1]
            res_trans.append(res)
    return torch.cat(res_trans, dim=1)


def velocity2position_mixeddiff(vel_seq, dt, init_pos):
    '''
    Proper inverse of estimate_linear_velocity that accounts for the mixed finite difference scheme.
    
    Args:
        vel_seq: velocity sequence of shape (B, T, ...)
        dt: time step  
        init_pos: initial position of shape (B, ...)
    
    Returns:
        position sequence of shape (B, T, ...)
    '''
    B, T = vel_seq.shape[:2]
    positions = torch.zeros_like(vel_seq)
    
    # Set initial position
    positions[:, 0] = init_pos
    
    if T == 1:
        return positions
    
    # For the second position, use forward difference inverse
    # vel[0] = (pos[1] - pos[0]) / dt -> pos[1] = pos[0] + vel[0] * dt
    positions[:, 1] = positions[:, 0] + vel_seq[:, 0] * dt
    
    # For middle positions, use central difference inverse
    # vel[i] = (pos[i+1] - pos[i-1]) / (2*dt) -> pos[i+1] = pos[i-1] + vel[i] * 2*dt
    for i in range(1, T-1):
        positions[:, i+1] = positions[:, i-1] + vel_seq[:, i] * (2 * dt)
    
    # The last velocity is backward difference: vel[T-1] = (pos[T-1] - pos[T-2]) / dt
    # This is already satisfied by the central difference computation above
    # so we calculate the last position directly: pos[T-1] = pos[T-2] + vel_seq[:, T-1] * dt
    final_pos = positions[:, -1] + vel_seq[:, -1] * dt
    
    return positions, final_pos

def estimate_angular_velocity(rot_seq, dt):
    '''
    Given a batch of sequences of T rotation matrices, estimates angular velocity at T-2 steps.
    Input sequence should be of shape (B, T, ..., 3, 3)
    '''
    # see https://en.wikipedia.org/wiki/Angular_velocity#Calculation_from_the_orientation_matrix
    dRdt = estimate_linear_velocity(rot_seq, dt)
    R = rot_seq
    RT = R.transpose(-1, -2)
    # compute skew-symmetric angular velocity tensor
    w_mat = torch.matmul(dRdt, RT)
    # pull out angular velocity vector by averaging symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = torch.stack([w_x, w_y, w_z], axis=-1)
    return w


def print_exp_info(args):
    logger.info(pprint.pformat(vars(args)))
    logger.info(f"# ------------ {args.name} ----------- #")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"{torch.cuda.device_count()} GPUs")
    logger.info(f"Random Seed: {args.random_seed}")

class EpochTracker:
    def __init__(self, metric_names, metric_directions):
        assert len(metric_names) == len(metric_directions), "Metric names and directions should have the same length"


        self.metric_names = metric_names
        self.states = ['train', 'val', 'test']
        self.types = ['last', 'best']


        self.values = {name: {state: {type_: {'value': np.inf if not is_higher_better else -np.inf, 'epoch': 0}
                                       for type_ in self.types}
                              for state in self.states}
                      for name, is_higher_better in zip(metric_names, metric_directions)}
                     
        self.loss_meters = {name: {state: AverageMeter(f"{name}_{state}")
                                   for state in self.states}
                            for name in metric_names}


        self.is_higher_better = {name: direction for name, direction in zip(metric_names, metric_directions)}
        self.train_history = {name: [] for name in metric_names}
        self.val_history = {name: [] for name in metric_names}


    def update_meter(self, name, state, value):
        self.loss_meters[name][state].update(value)


    def update_values(self, name, state, epoch):
        value_avg = self.loss_meters[name][state].avg
        new_best = False


        if ((value_avg < self.values[name][state]['best']['value'] and not self.is_higher_better[name]) or
           (value_avg > self.values[name][state]['best']['value'] and self.is_higher_better[name])):
            self.values[name][state]['best']['value'] = value_avg
            self.values[name][state]['best']['epoch'] = epoch
            new_best = True
        self.values[name][state]['last']['value'] = value_avg
        self.values[name][state]['last']['epoch'] = epoch
        return new_best


    def get(self, name, state, type_):
        return self.values[name][state][type_]


    def reset(self):
        for name in self.metric_names:
            for state in self.states:
                self.loss_meters[name][state].reset()


    def flatten_values(self):
        flat_dict = {}
        for name in self.metric_names:
            for state in self.states:
                for type_ in self.types:
                    value_key = f"{name}_{state}_{type_}"
                    epoch_key = f"{name}_{state}_{type_}_epoch"
                    flat_dict[value_key] = self.values[name][state][type_]['value']
                    flat_dict[epoch_key] = self.values[name][state][type_]['epoch']
        return flat_dict
   
    def update_and_plot(self, name, epoch, save_path):
        new_best_train = self.update_values(name, 'train', epoch)
        new_best_val = self.update_values(name, 'val', epoch)


        self.train_history[name].append(self.loss_meters[name]['train'].avg)
        self.val_history[name].append(self.loss_meters[name]['val'].avg)


        train_values = self.train_history[name]
        val_values = self.val_history[name]
        epochs = list(range(1, len(train_values) + 1))


        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_values, label='Train')
        plt.plot(epochs, val_values, label='Val')
        plt.title(f'Train vs Val {name} over epochs')
        plt.xlabel('Epochs')
        plt.ylabel(name)
        plt.legend()
        plt.savefig(save_path)
        plt.close()


        return new_best_train, new_best_val

def record_trial(args, tracker):
    """
    1. record notes, score, env_name, experments_path,
    """
    csv_path = args.out_path + args.project_name+".csv"
    all_print_dict = vars(args)
    all_print_dict.update(tracker.flatten_values())
    if not os.path.exists(csv_path):
        pd.DataFrame([all_print_dict]).to_csv(csv_path, index=False)
    else:
        df_existing = pd.read_csv(csv_path)
        df_new = pd.DataFrame([all_print_dict])
        df_aligned = df_existing.append(df_new).fillna("")
        df_aligned.to_csv(csv_path, index=False)
        
def set_random_seed(args):
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = args.deterministic #args.CUDNN_DETERMINISTIC
    torch.backends.cudnn.benchmark = args.benchmark
    torch.backends.cudnn.enabled = args.cudnn_enabled
    
    
def save_checkpoints(
        save_path, 
        model, 
        opt=None, 
        epoch=None, 
        lrs=None, 
        save_dtype="float32"
    ):
    with torch.no_grad():
        offload_to_cpu = get_world_size() > 1
        save_dtype = getattr(torch, save_dtype)
        # if get_world_size() > 1:
        #     with model.summon_full_params(
        #         model, writeback=True, offload_to_cpu=offload_to_cpu
        #     ):
        #         states = model.state_dict()
        #         states = {k: v.to(dtype=save_dtype) for k, v in states.items()}
        # else:
        states = model.state_dict()
        states = {k: v.to(dtype=save_dtype) for k, v in states.items()}
                
        # if lrs is not None:
        #     states = { 'model_state': states,
        #             'epoch': epoch + 1,
        #             'opt_state': opt.state_dict(),
        #             'lrs':lrs.state_dict(),}
        # elif opt is not None:
        #     states = { 'model_state': states,
        #             'epoch': epoch + 1,
        #             'opt_state': opt.state_dict(),}
        # else:
        #     states = { 'model_state': states,}

    logger.info(f"Saving checkpoints to {save_path} with dtype {save_dtype}")
    # torch.distributed.barrier()
    torch.cuda.synchronize()

    # torch.save(states, save_path)
    save_path = save_path + ".safetensors"
    safetensors.torch.save_file(
        states, save_path, metadata={"format": "pt", "dtype": str(save_dtype)}
    )

def load_checkpoints(model, save_path, rank=0, after_distributed=False):
    # breakpoint()
    # states = torch.load(save_path)
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    states = safetensors.torch.load_file(save_path, device="cpu")
    # print([key for key in states.keys()])
    # print(after_distributed )
    if not after_distributed:
        new_weights = OrderedDict()
        flag=False
        for k, v in states.items():
            # if "audio_procemb" in k: breakpoint()
            if "module" not in k and not k.startswith("m."):
                break
            else:
                if "module." in k:
                    if k[7:].startswith("m."):
                        new_weights[k[9:]]=v
                    else:
                        new_weights[k[7:]]=v
                    flag=True
                elif k.startswith("m."):
                    new_weights[k[2:]]=v
                    flag=True
        if flag: 
            model.load_state_dict(new_weights)
            
        else:
            model.load_state_dict(states)
    else:
        model.load_state_dict(states)
    
    logger.info(f"load self-pretrained checkpoints")

def model_complexity(model, args):
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model,  (args.T_GLOBAL._DIM, args.TRAIN.CROP, args.TRAIN), 
        as_strings=False, print_per_layer_stat=False)
    logger.info('{:<30}  {:<8} BFlops'.format('Computational complexity: ', flops / 1e9))
    logger.info('{:<30}  {:<8} MParams'.format('Number of parameters: ', params / 1e6))
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



# Logger tools

def setup_logger(save_dir, distributed_rank=0, filename="log.txt", mode="a"):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        filename (string): log save name.
        mode(str): log file write mode, `append` or `override`. default is `a`.

    Return:
        logger instance.
    """
    loguru_format = (
        "<blue>{time: MM-DD HH:mm:ss}</blue> | "
        "<level>{level: <2}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        # "<level>{message}</level>"
    )

    # Custom filter to remove null characters
    def filter_null_chars(record):
        if record["message"]:
            record["message"] = record["message"].replace('\x00', '')#.replace('\0', '')
        return record

    logger.remove()
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    
    # only keep logger in rank0 process
    # if True: # distributed_rank == 0:
    logger.add(
        sys.stderr,
        format=loguru_format,
        level="INFO",
        # encoding="utf-8",
        # errors="replace",
        enqueue=True,
        filter=filter_null_chars
    )
    logger.add(save_file,
        format=loguru_format,
        # encoding="utf-8",
        # errors="replace",
        level="INFO",
        enqueue=True,
        filter=filter_null_chars  
    )


def set_args_and_logger(args, rank):
    """
    set logger file and print args
    """
    args_name_dir = args.out_path + args.name + args.notes + "/"
    # DDP-safe directory creation
    os.makedirs(args_name_dir, exist_ok=True)
    if rank == 0:
        
        args_name = args_name_dir + "/" + "config" +".yaml"
        if os.path.exists(args_name):
            s_add = 10
            logger.warning(f"Already exist args, add {s_add} to ran_seed to continue training")
            args.random_seed += s_add
        else:
            with open(args_name, "w+") as f:
                yaml.dump(args.__dict__, f, default_flow_style=False)
                #json.dump(args.__dict__, f)
    if args.is_train:
        setup_logger(args_name_dir, rank, filename=f"train_log.log")
    else:    
        setup_logger(args_name_dir, rank, filename=f"{args.name}_test.log")


def update_args_file(args, rank):
    """
    update args file with new random seed
    """
    args_name_dir = args.out_path + args.name + args.notes + "/"
    if rank == 0:
        args_name = args_name_dir + "/" + "config" +".yaml"
        with open(args_name, "w+") as f:
            yaml.dump(args.__dict__, f, default_flow_style=False)
            #json.dump(args.__dict__, f)



# Inverse selection for smplx model
def inverse_selection(filtered_t, selection_array, n):
        # Create an array of all zeros with shape n*165
        original_shape_t = np.zeros((n, selection_array.size))
        
        # Find the index position that is 1 in the selection array
        selected_indices = np.where(selection_array == 1)[0]
        
        # Fill the values filtered_t into the corresponding positions in original_shape_t
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
            
        return original_shape_t

def inverse_selection_tensor(filtered_t, selection_array, n):
    # Create an array of all zeros with shape n*165
    selection_array = torch.from_numpy(selection_array).cuda()
    original_shape_t = torch.zeros((n, 165)).cuda()
    
    # Find the index position that is 1 in the selection array
    selected_indices = torch.where(selection_array == 1)[0]
    # breakpoint()

    # Fill the values filtered_t into the corresponding positions in original_shape_t
    for i in range(n):
        original_shape_t[i, selected_indices] = filtered_t[i]
        
    return original_shape_t