import smplx
import os
import torch
# import logging
from loguru import logger
import time
import numpy as np
from tqdm import tqdm
import gc
# from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .basecausalcodec_trainer import BaseCausalCodecTrainer
from .utils import rotation_conversions as rc
from .utils.loss_factory import get_loss_func
from .utils import tools as other_tools
from .utils.optim_factory import create_optimizer
from .utils.scheduler_factory import create_scheduler
from .utils.mixed_precision import (
    prepare_mixed_precision, 
    upcast_mixed_precision,
    downcast_mixed_precision
)
from miburi.models import loaders, GestureMimiCodec

from .dataloaders.utils.visualize import (
    render_smplx_debug_video,
    stitch_videos_hstack,
)


class UpperBodyCausalCodecTrainer(BaseCausalCodecTrainer):
    def __init__(self, args):
        super(UpperBodyCausalCodecTrainer, self).__init__(args)

        self.tracker = other_tools.EpochTracker(
            ["rec_6d", "rec_rot", "rec_aa", "vel", "ver", "com", "kl", "acc", "loc_rec", "loc_vel", "loc_acc", "teacherloss", "laploss6d", "laplossloc", "velacc6d_penalty",  "velaccloc_penalty", "mmd"], 
            [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
        )
        
        if not self.args.rot6d: #"rot6d" not in args.pose_rep:
            logger.error(f"this script is for rot6d, your pose rep. is {args.pose_rep}")

        self.rec_loss = get_loss_func("GeodesicLoss", reduction='none')
        # TODO: add specific loss for contact and foot contact
        self.rot6d_loss = torch.nn.MSELoss(reduction='none')
        self.aa_loss = torch.nn.MSELoss(reduction='none')
        self.vel_loss = torch.nn.MSELoss(reduction='none')
        self.vectices_loss = torch.nn.MSELoss(reduction='none')
        self.loc_loss = torch.nn.MSELoss(reduction='none')
        self.loc_vel_loss = torch.nn.MSELoss(reduction='none')
        self.loc_acc_loss = torch.nn.MSELoss(reduction='none')

        # self.rot6d_loss = torch.nn.HuberLoss(reduction='none')
        # self.aa_loss = torch.nn.HuberLoss(reduction='none')
        # self.vel_loss = torch.nn.HuberLoss(reduction='none')
        # self.vectices_loss = torch.nn.HuberLoss(reduction='none')
        # self.loc_loss = torch.nn.HuberLoss(reduction='none')
        # self.loc_vel_loss = torch.nn.HuberLoss(reduction='none')
        # self.loc_acc_loss = torch.nn.HuberLoss(reduction='none')

        # self.trans_loss = torch.nn.MSELoss(reduction='none')
        self.teacher_loss = get_loss_func("CosineSimilarityLoss")
        self.laplacian_loss = get_loss_func("laplacian_loss", kernel_size=5, reduction='none')
        self.mmd_loss = get_loss_func("mmd_loss")
        self.smoothness_loss = get_loss_func("smoothness_loss", lambda_vel=self.args.accel_penalty_weight*0.01, lambda_acc=self.args.accel_penalty_weight, reduction='topk', topk_ratio=0.1)

        self.param_dtype = getattr(torch, args.param_dtype)
        self.optim_dtype = getattr(torch, args.optim_dtype)

        assert self.param_dtype == torch.float32, "param_dtype must be float32"

        # prepare_mixed_precision(
        #     self.model.parameters(), 
        #     param_dtype=self.param_dtype, 
        #     optim_dtype=self.optim_dtype
        # )
        


    def train(self, epoch):
        for p in self.model.parameters():
            if torch.isnan(p).any():
                raise ValueError("nan in model parameters")
        gc.collect()
        self.model.train()
        # self.teacher.eval()
        # self.vq0_codec.eval()
        t_start = time.time()
        self.tracker.reset()
        if self.args.is_continue and epoch != 0:
            self.opt_s.step(epoch)
        for its, dict_data in enumerate(self.train_loader):
            # if its % 10 == 0 and self.rank == 0: 
            #     self.train_recording(epoch, its, 0, 0, 0, 0)
            # continue
            tar_pose_upper = dict_data["motion_upper"]
            tar_pose_hands = dict_data["motion_hands"]
            tar_trans = dict_data["transl"].to(self.local_rank)
            tar_beta = dict_data["beta"].to(self.local_rank)
            
            # tar_inp_tokenvecs = dict_data['token_vecs'].to(self.rank)

            dataset_name_per_item = dict_data['dataset_name']
            dataset_name_per_item = np.array(dataset_name_per_item)
            # get indices for dataset
            # seamless_indices = np.where(dataset_name_per_item != 'BEATX')[0]
            embody3d_indices = np.where(dataset_name_per_item != 'BEATX')[0]
            beatx_indices = np.where(dataset_name_per_item == 'BEATX')[0]
            # breakpoint()
            
            # if self.rank == 0:
                # logger.info(f"@ rank {self.rank}, tar_pose_upper shape: {tar_pose_upper.shape}")
            # breakpoint() # check masks 
            upper_hands_joint_mask = self.train_data.upper_mask_for_flattened + \
                self.train_data.hands_mask_for_flattened
            
            tar_pose_upper = tar_pose_upper.to(self.local_rank)
            tar_pose_hands = tar_pose_hands.to(self.local_rank)
            
            # breakpoint()
            bs, n, udim = tar_pose_upper.shape
            uj = udim //3
            tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, uj, 3))
            tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, uj*6)

            hj = tar_pose_hands.shape[-1]
            hj = hj // 3
            tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, hj, 3))
            tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, hj*6)

            
            j = uj + hj 
            tar_pose = torch.cat([tar_pose_upper, tar_pose_hands], dim=-1)

            tar_exps = torch.zeros((bs, n, 100)).to(self.local_rank)
            

            t_data = time.time() - t_start
            
            self.opt.zero_grad()
            g_loss_final = 0
            # breakpoint()

            # input: pose, trans, contact -> output: pose only
            in_tar_pose_upper = torch.cat((tar_pose_upper, tar_pose_hands), dim=-1)

            # with torch.no_grad():
            #     teacher_features, additional_outs = self.teacher(in_tar_pose_upper, tar_inp_tokenvecs)
            

            # check tar_pose dim and netout dim # check tar/contact and joint seperation
            # in_tar_pose_upper = in_tar_pose_upper.to(self.param_dtype)
            # if torch.any(torch.isnan(in_tar_pose_upper)):
            #     raise ValueError("nan in motion")
            
            net_out, upper_q_res, z_encoder, loss_timemask = self.model(in_tar_pose_upper)
            # breakpoint() # check input netout dtype and z_encoder dtype
            # if torch.any(torch.isnan(net_out)):
            #     raise ValueError("nan in net_out")

            # for p in self.model.parameters():
            #     if torch.isnan(p).any():
            #         raise ValueError("nan in model parameters")

            
            # loss_timemask = loss_timemask.view(1, -1)
            
            
            rec_pose = net_out
            # rec_pose = tar_pose.clone()
            # loss_timemask = torch.ones(n).to(self.local_rank)
            # breakpoint()

            # 6d loss
            rec_pose = rec_pose.reshape(bs, n, j, 6)
            tar_pose = tar_pose.reshape(bs, n, j, 6)
            loss_6d = self.rot6d_loss(rec_pose, tar_pose) * loss_timemask.view(1, -1, 1, 1)
            if len(embody3d_indices) > 0: loss_6d[embody3d_indices, :, :3] *= 0.005; # weight down spine for embody3d
            loss_6d = loss_6d.mean() * self.args.rec_weight * self.args.rec_6d_weight
            self.tracker.update_meter("rec_6d", "train", loss_6d.item())
            g_loss_final += loss_6d

            if rec_pose.shape[1] <=6: 
                laploss_6d = torch.zeros_like(loss_6d)
            else:
                laploss_6d = self.laplacian_loss(rec_pose.reshape(bs, n, j*6), tar_pose.reshape(bs, n, j*6))
                if len(embody3d_indices) > 0: laploss_6d[embody3d_indices, :, :3*6] *= 0.005 # weight down spine for embody3d
                
            laploss_6d = laploss_6d.mean() * self.args.rec_weight * self.args.lap_weight
            self.tracker.update_meter("laploss6d", "train", laploss_6d.item())
            g_loss_final += laploss_6d


            # velocity penalty on 6d:
            if rec_pose.shape[1] <=6:
                smoothloss6d = torch.tensor(0.).to(rec_pose)
            else:
                smoothloss6d = self.smoothness_loss(
                    rec_pose.reshape(len(rec_pose), n, j*6)
                )
                if len(embody3d_indices) > 0:
                    # breakpoint()
                    smoothloss6d_spine = self.smoothness_loss(
                        rec_pose[embody3d_indices][:, :, :3].reshape(len(embody3d_indices), n, 3*6)
                    )
                    smoothloss6d = smoothloss6d + smoothloss6d_spine * 50

                # breakpoint()
            smoothloss6d_val = smoothloss6d * 0.01
            g_loss_final += smoothloss6d_val
            self.tracker.update_meter("velacc6d_penalty", "train", smoothloss6d_val.item())

            

            
            # rotation matrix loss
            rec_pose = rc.rotation_6d_to_matrix(rec_pose)  * loss_timemask.view(1, -1, 1, 1, 1)
            tar_pose = rc.rotation_6d_to_matrix(tar_pose)  * loss_timemask.view(1, -1, 1, 1, 1)
            loss_rec_rot = self.rec_loss(rec_pose, tar_pose) 
            # if len(embody3d_indices) > 0: breakpoint(); loss_rec_rot[embody3d_indices, :, :3] *= 0.01; # weight down spine for embody3d
            loss_rec_rot = loss_rec_rot.mean() * self.args.rec_weight * self.args.rec_pos_weight
            self.tracker.update_meter("rec_rot", "train", loss_rec_rot.item())
            g_loss_final += loss_rec_rot

            # axis angle loss
            rec_pose_aa = rc.matrix_to_axis_angle(rec_pose)
            tar_pose_aa = rc.matrix_to_axis_angle(tar_pose)
            loss_rec_aa = self.aa_loss(rec_pose_aa, tar_pose_aa) * loss_timemask.view(1, -1, 1, 1)
            if len(embody3d_indices) > 0: loss_rec_aa[embody3d_indices, :, :3] *= 0.005; # weight down spine for embody3d
            loss_rec_aa = loss_rec_aa.mean() * self.args.rec_weight * self.args.rec_aa_weight
            self.tracker.update_meter("rec_aa", "train", loss_rec_aa.item())
            g_loss_final += loss_rec_aa

            # # ------- vis embody3d  debug only -------
            # if self.global_rank == 0:
            #     print(f"@ epoch {epoch}, its {its}/{len(self.train_loader)}, vis sample for debug")
            #     sample_names = dict_data["filechunk_id"]
            #     for b_idx, sample_name in enumerate(sample_names):
            #         # breakpoint()
            #         if not (sample_name.endswith("C0") or sample_name.endswith("C2")):
            #             continue
            #         sample_save_path = os.path.join(
            #             "/CT/GestureSynth1/work/GestureMoshi/moshi/moshi/experiments_embody3d/embody3d_debug",
            #             sample_name + ".mp4"
            #         )
            #         if os.path.exists(sample_save_path):
            #             continue
            #         # breakpoint()
            #         num_frames = tar_pose_aa.shape[1]
            #         # breakpoint()
            #         # tar_pose_aa[:, :, :3] = 0.0
            #         tar_pose_aa_vis = tar_pose_aa[b_idx].reshape(num_frames, -1)
            #         tar_pose_aa_vis = self.train_data.inverse_selection_tensor(tar_pose_aa_vis, upper_hands_joint_mask, tar_pose_aa_vis.shape[0])
                
                    
            #         tar_pose_np = tar_pose_aa_vis.detach().cpu().numpy()

            #         # breakpoint()
            #         tar_trans_np = tar_trans[b_idx].detach().cpu().numpy()
            #         tar_beta_np = tar_beta[b_idx].detach().cpu().numpy()
            #         tar_exps_np = tar_exps[b_idx].detach().cpu().numpy()

            #         tar_trans_np = np.zeros_like(tar_trans_np)  # center the translation for visualization

                    

            #         # rec_pose = self.train_data.inverse_selection_tensor(rec_pose, upper_hands_joint_mask, rec_pose.shape[0])
                    

            #         tar_out = {
            #             "global_orient": tar_pose_np[:, :3],
            #             "body_pose": tar_pose_np[:, 1*3:22*3],
            #             "left_hand_pose": tar_pose_np[:, 25*3:40*3],
            #             "right_hand_pose": tar_pose_np[:, 40*3:55*3],
            #             "transl": tar_trans_np,
            #             "betas": tar_beta_np,
            #             "expression": tar_exps_np,
            #             "jaw_pose": tar_pose_np[:, 66:69],
            #             "leye_pose": tar_pose_np[:, 69:72],
            #             "reye_pose": tar_pose_np[:, 72:75],
            #         }
            #         # breakpoint()
            #         visualize_smpl(
            #             self.smplx_model, 
            #             tar_out, 
            #             sample_save_path,
            #             fps=self.args.motion_fps,
            #             mesh_color="light_pink",
            #             model='smplx',
            #         )
            #         logger.info(f"visualized sample saved at {sample_save_path}")
            # continue
            # # ---------------------------------------------- #
            
            # velocity loss and acceleration loss
            velocity_loss =  self.vel_loss(rec_pose[:, 1:] - rec_pose[:, :-1], tar_pose[:, 1:] - tar_pose[:, :-1]) * loss_timemask[1:].view(1, -1, 1, 1, 1)
            velocity_loss = velocity_loss.mean() * self.args.rec_weight * 30
            if rec_pose.shape[1] <=6:
                acceleration_loss = torch.zeros_like(velocity_loss)
            else:
                acceleration_loss =  self.vel_loss(rec_pose[:, 2:] + rec_pose[:, :-2] - 2 * rec_pose[:, 1:-1], tar_pose[:, 2:] + tar_pose[:, :-2] - 2 * tar_pose[:, 1:-1]) * loss_timemask[2:].view(1, -1, 1, 1, 1)
            
            acceleration_loss = acceleration_loss.mean() * self.args.rec_weight * 30

            
            self.tracker.update_meter("vel", "train", velocity_loss.item())
            self.tracker.update_meter("acc", "train", acceleration_loss.item())
            g_loss_final += velocity_loss
            g_loss_final += acceleration_loss

            # vertices loss and joint location loss
            if self.args.rec_ver_weight > 0 or self.args.rec_loc_weight > 0:
                tar_pose = tar_pose_aa.reshape(bs*n, j*3)
                rec_pose = rec_pose_aa.reshape(bs*n, j*3)
                rec_pose = self.train_data.inverse_selection_tensor(rec_pose, upper_hands_joint_mask, rec_pose.shape[0])
                tar_pose = self.train_data.inverse_selection_tensor(tar_pose, upper_hands_joint_mask, tar_pose.shape[0])
                need_verts = self.args.rec_ver_weight > 0
                zero_transl = torch.zeros((bs, n, 3), device=rec_pose.device, dtype=rec_pose.dtype)
                rec_joints, rec_verts = self._smplx_forward(
                    pose_aa_full=rec_pose.reshape(bs, n, 55, 3),
                    transl=zero_transl,
                    betas=tar_beta.reshape(bs, n, 300),
                    expressions=tar_exps.reshape(bs, n, 100),
                    return_verts=need_verts,
                )
                tar_joints, tar_verts = self._smplx_forward(
                    pose_aa_full=tar_pose.reshape(bs, n, 55, 3),
                    transl=zero_transl,
                    betas=tar_beta.reshape(bs, n, 300),
                    expressions=tar_exps.reshape(bs, n, 100),
                    return_verts=need_verts,
                )

                # vertices loss
                if self.args.rec_ver_weight > 0:
                    pred_vertices = rec_verts
                    tar_vertices = tar_verts

                    vectices_loss = self.vectices_loss(pred_vertices, tar_vertices) * loss_timemask.view(1, -1, 1, 1)
                    vectices_loss = vectices_loss.mean()
                    self.tracker.update_meter("ver", "train", vectices_loss.item()*self.args.rec_weight * self.args.rec_ver_weight)
                    g_loss_final += vectices_loss.mean()*self.args.rec_weight*self.args.rec_ver_weight

                    vertices_vel_loss = self.vel_loss(pred_vertices[:, 1:] - pred_vertices[:, :-1], tar_vertices[:, 1:] - tar_vertices[:, :-1]) * loss_timemask[1:].view(1, -1, 1, 1)
                    vertices_vel_loss = vertices_vel_loss.mean() * self.args.rec_weight * 10
                    if pred_vertices.shape[1] <=6: 
                        vertices_acc_loss = torch.zeros_like(vertices_vel_loss)
                    else:
                        vertices_acc_loss = self.vel_loss(pred_vertices[:, 2:] + pred_vertices[:, :-2] - 2 * pred_vertices[:, 1:-1], tar_vertices[:, 2:] + tar_vertices[:, :-2] - 2 * tar_vertices[:, 1:-1]) * loss_timemask[2:].view(1, -1, 1, 1)
                    vertices_acc_loss = vertices_acc_loss.mean() * self.args.rec_weight * 10
                    
                    if pred_vertices.shape[1] <=6:
                        vertices_lap_loss = torch.zeros_like(vertices_vel_loss)
                    else:
                        vertices_lap_loss = self.laplacian_loss(pred_vertices.reshape(bs, n, -1), tar_vertices.reshape(bs, n, -1))
                        
                    vertices_lap_loss = vertices_lap_loss.mean() * self.args.rec_weight * 10
                    g_loss_final += vertices_lap_loss #* self.args.rec_weight * self.args.rec_ver_weight
                    g_loss_final += vertices_vel_loss #* self.args.rec_weight * self.args.rec_ver_weight
                    g_loss_final += vertices_acc_loss #* self.args.rec_weight * self.args.rec_ver_weight

                # joint location/position loss
                if self.args.rec_loc_weight > 0:
                    pred_joints = rec_joints
                    tar_joints = tar_joints  # already (bs, n, 55, 3) from _smplx_forward
                    pred_vel = pred_joints[:, 1:] - pred_joints[:, :-1]
                    tar_vel = tar_joints[:, 1:] - tar_joints[:, :-1]
                    
                    pred_acc_joints = pred_joints[:, 2:] + pred_joints[:, :-2] - 2 * pred_joints[:, 1:-1]
                    tar_acc_joints = tar_joints[:, 2:] + tar_joints[:, :-2] - 2 * tar_joints[:, 1:-1]

                    joint_loc_loss = self.loc_loss(pred_joints, tar_joints) * loss_timemask.view(1, -1, 1, 1)
                    if len(embody3d_indices) > 0: joint_loc_loss[embody3d_indices][:, :, [3, 6, 9]] *= 0.005; # weight down spine for embody3d
                    self.tracker.update_meter("loc_rec", "train", joint_loc_loss.mean().item()*self.args.rec_weight * self.args.rec_loc_weight)
                    g_loss_final += joint_loc_loss.mean()*self.args.rec_weight*self.args.rec_loc_weight

                    joint_vel_loss = self.loc_vel_loss(pred_vel, tar_vel) * loss_timemask[1:].view(1, -1, 1, 1)
                    self.tracker.update_meter("loc_vel", "train", joint_vel_loss.mean().item()*self.args.rec_weight * self.args.rec_loc_weight * 30)
                    g_loss_final += joint_vel_loss.mean()*self.args.rec_weight*self.args.rec_loc_weight * 30
                    # breakpoint()
                    
                    if pred_joints.shape[1] <=6:
                        joint_acc_loss = torch.zeros_like(joint_vel_loss)
                    else:
                        joint_acc_loss =  self.loc_acc_loss(pred_acc_joints, tar_acc_joints) * loss_timemask[2:].view(1, -1, 1, 1)
                        
                    g_loss_final += joint_acc_loss.mean()*self.args.rec_weight*self.args.rec_loc_weight * 30
                    self.tracker.update_meter("loc_acc", "train", joint_acc_loss.mean().item()*self.args.rec_weight * self.args.rec_loc_weight  * 30)

                    if pred_joints.shape[1] <=6:
                        joint_lap_loss = torch.zeros_like(joint_vel_loss)
                    else:
                        joint_lap_loss = self.laplacian_loss(pred_joints.reshape(bs, n, -1), tar_joints.reshape(bs, n, -1))
                        if len(embody3d_indices) > 0: joint_lap_loss[embody3d_indices][:, :, [9, 10, 11] + [18, 19, 20] + [27, 28, 29]] *= 0.005; # weight down spine for embody3d
                    
                        
                    g_loss_final += joint_lap_loss.mean() * self.args.rec_weight * self.args.rec_loc_weight * 30
                    self.tracker.update_meter("laplossloc", "train", joint_lap_loss.mean().item()*self.args.rec_weight * self.args.rec_loc_weight * 30)


                    # velocity penalty on loc:
                    
                    if pred_joints.shape[1] <=6:
                        smoothloss_joints = torch.tensor(0.).to(pred_joints)
                    else:
                        smoothloss_joints = self.smoothness_loss(
                            pred_joints.reshape(len(pred_joints), n, -1)
                        )
                        if len(embody3d_indices) > 0:
                            # breakpoint()
                            smoothlossjoints_spine = self.smoothness_loss(
                                pred_joints[embody3d_indices][:, :, [3, 6, 9]].reshape(len(embody3d_indices), n, 3*3)
                            )
                            smoothloss_joints = smoothloss_joints + smoothlossjoints_spine * 100
                    
                    jointsmoothloss_val = smoothloss_joints * 0.01 # less penalty on beatx

                    g_loss_final += jointsmoothloss_val
                    self.tracker.update_meter("velaccloc_penalty", "train", jointsmoothloss_val.item())
                

            # ---------------------- vqvae loss -------------------------- #
            # breakpoint()
            upper_loss_embedding = upper_q_res.penalty
            if epoch < 1:
                embedding_weight = 0
            else:
                embedding_weight = min(1.0, epoch/self.args.kl_warmup_epochs) * self.args.comloss_weight
            g_loss_final += upper_loss_embedding * embedding_weight
            self.tracker.update_meter("com", "train", (upper_loss_embedding * embedding_weight).item()) # embedding_weight * 
            
            # ----------------------- MMD loss between two datasets -------------------------- #
            if self.args.mmd_weight > 0 and len(embody3d_indices) > 4 and len(beatx_indices) > 4:
                # breakpoint()
                emb3d_z = z_encoder[embody3d_indices].permute(0, 2, 1).contiguous().view(-1, z_encoder.shape[1])
                beatx_z = z_encoder[beatx_indices].permute(0, 2, 1).contiguous().view(-1, z_encoder.shape[1])
                mmd_loss = self.mmd_loss(emb3d_z, beatx_z)
                mmd_loss = mmd_loss * self.args.mmd_weight
                g_loss_final += mmd_loss.to(g_loss_final.dtype)
                self.tracker.update_meter("mmd", "train", mmd_loss.item())
            



            g_loss_final.backward()
            # breakpoint()
            # for name, param in self.model.named_parameters():
                # print(name, param.grad.data.norm())
            # breakpoint() # check dtype of self.model.parameters()

            # upcast_mixed_precision(self.model.parameters(), optim_dtype=self.optim_dtype)

            if self.args.grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.opt.step()

            # downcast_mixed_precision(self.model.parameters(), param_dtype=self.param_dtype)


            if self.args.lr_policy == "onecyclelr":
                self.opt_s.step()


            t_train = time.time() - t_start - t_data
            t_start = time.time()
            mem_cost = torch.cuda.memory_cached() / 1E9

            # print("rank", self.rank, "epoch", epoch, "its", its, "loss", g_loss_final.item(), "lr", self.opt.param_groups[0]['lr'], "mem_cost", mem_cost)
            lr_g = self.opt.param_groups[0]['lr']
            if its % self.args.log_period == 0: #and self.local_rank == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g)   
                if its % (self.args.log_period * 10) == 0:
                    metric_print = f"[GPU{self.global_rank}:{self.local_rank}]" + str({k: round(v.item(), 3) for k, v in upper_q_res.metrics.items()})
                    logger.info(metric_print)
            
            if self.global_rank == 0 and its % 1000 == 0 and its > 0:
                logger.info(f"[GPU {self.global_rank}] Saving checkpoints at epoch {epoch} iter {its}:")
                other_tools.save_checkpoints(os.path.join(self.checkpoint_path, f"iter_{epoch}_{its}"), self.model, opt=None, epoch=None, lrs=None, save_dtype=self.args.param_dtype)
                # trainer.test(epoch)
                self.args.test_ckpt = os.path.join(self.checkpoint_path, f"iter_{epoch}_{its}.safetensors")
                other_tools.update_args_file(self.args, rank=self.global_rank)
                # breakpoint() # add its>0 condition

            if self.args.debug:
                if its == 100: break
        torch.cuda.synchronize()
        # breakpoint()
        if self.args.lr_policy != "onecyclelr":
            self.opt_s.step(epoch)
                    
    def val(self, epoch):
        self.model.eval()
        # self.teacher.eval()
        # self.vq0_codec.eval()
        t_start = time.time()
        self._mpjpe_reset()
        with torch.no_grad():
            for its, dict_data in enumerate(self.val_loader):
                # if its % 10 == 0 and self.rank == 0: 
                #     logger.info(f"{its}/{len(self.val_loader)}")
                # continue
                tar_pose_upper = dict_data["motion_upper"]
                tar_pose_hands = dict_data["motion_hands"]
                
                tar_beta = dict_data["beta"].to(self.local_rank)
                tar_trans = dict_data["transl"].to(self.local_rank)
                # tar_inp_tokenvecs = dict_data['token_vecs'].to(self.rank)

                dataset_name_per_item = dict_data['dataset_name']
                dataset_name_per_item = np.array(dataset_name_per_item)
                # seamless_indices = np.where(dataset_name_per_item != 'BEATX')[0]
                embody3d_indices = np.where(dataset_name_per_item != 'BEATX')[0]
                beatx_indices = np.where(dataset_name_per_item == 'BEATX')[0]
                # breakpoint()
                

                tar_pose_upper = tar_pose_upper.to(self.local_rank)
                tar_pose_hands = tar_pose_hands.to(self.local_rank)
                
                upper_hands_joint_mask = self.val_data.upper_mask_for_flattened + \
                    self.val_data.hands_mask_for_flattened

                bs, n, udim = tar_pose_upper.shape
                uj = udim //3
                tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, uj, 3))
                tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, uj*6)

                hj = tar_pose_hands.shape[-1]
                hj = hj // 3
                tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, hj, 3))
                tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, hj*6)

                
                j = uj + hj 
                tar_pose = torch.cat([tar_pose_upper, tar_pose_hands], dim=-1)

                tar_exps = torch.zeros((bs, n, 100), device=self.local_rank)
                # breakpoint()

                # input: pose, trans, contact -> output: pose only
                in_tar_pose_upper = torch.cat((tar_pose_upper, tar_pose_hands), dim=-1)
                # teacher_features, additional_outs = self.teacher(in_tar_pose_upper, tar_inp_tokenvecs)
                
                # breakpoint() # check tar_pose dim and netout dim # check tar/contact and joint seperation
                in_tar_pose_upper = in_tar_pose_upper.to(self.param_dtype)
                net_out, upper_q_res, z_encoder, loss_timemask = self.model(in_tar_pose_upper)

                rec_pose = net_out
                # assert self.model.proj_vq, "Cant train with distillation without projection of vq0"
                # sem_emb = upper_q_res.sem_emb
                # sem_emb = self.model.vq_projector(sem_emb)
                # teacher_loss = self.teacher_loss(sem_emb, teacher_features)
                # teacher_loss = teacher_loss * self.args.teacherguidance_weight
                # self.tracker.update_meter("teacherloss", "val", teacher_loss.item())
                

                # 6d loss
                rec_pose = rec_pose.reshape(bs, n, j, 6)
                tar_pose = tar_pose.reshape(bs, n, j, 6)
                loss_6d = self.rot6d_loss(rec_pose, tar_pose) * loss_timemask.view(1, -1, 1, 1)
                loss_6d = loss_6d.mean() * self.args.rec_weight * self.args.rec_6d_weight
                self.tracker.update_meter("rec_6d", "val", loss_6d.item())
                # g_loss_final += loss_6d

                # rotation matrix loss
                rec_pose = rc.rotation_6d_to_matrix(rec_pose)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose)
                loss_rec_rot = self.rec_loss(rec_pose, tar_pose) * loss_timemask.view(1, -1, 1, 1, 1)
                loss_rec_rot = loss_rec_rot.mean() * self.args.rec_weight * self.args.rec_pos_weight
                self.tracker.update_meter("rec_rot", "val", loss_rec_rot.item())
                # g_loss_final += loss_rec_rot

                # axis angle loss
                rec_pose_aa = rc.matrix_to_axis_angle(rec_pose)
                tar_pose_aa = rc.matrix_to_axis_angle(tar_pose)
                loss_rec_aa = self.aa_loss(rec_pose_aa, tar_pose_aa) * loss_timemask.view(1, -1, 1, 1)
                loss_rec_aa = loss_rec_aa.mean() * self.args.rec_weight * self.args.rec_aa_weight
                self.tracker.update_meter("rec_aa", "val", loss_rec_aa.item())
                # g_loss_final += loss_rec_aa

                 # vertices loss / joint loc loss / MPJPE
                if (
                    self.args.rec_ver_weight > 0
                    or self.args.rec_loc_weight > 0
                    or getattr(self.args, "mpjpe_eval_enabled", True)
                ):
                    tar_pose = tar_pose_aa.reshape(bs*n, j*3)
                    rec_pose = rec_pose_aa.reshape(bs*n, j*3)
                    rec_pose = self.val_data.inverse_selection_tensor(rec_pose, upper_hands_joint_mask, rec_pose.shape[0])
                    tar_pose = self.val_data.inverse_selection_tensor(tar_pose, upper_hands_joint_mask, tar_pose.shape[0])
                    need_verts = self.args.rec_ver_weight > 0
                    zero_transl = torch.zeros((bs, n, 3), device=rec_pose.device, dtype=rec_pose.dtype)
                    rec_joints, rec_verts = self._smplx_forward(
                        pose_aa_full=rec_pose.reshape(bs, n, 55, 3),
                        transl=zero_transl,
                        betas=tar_beta.reshape(bs, n, 300),
                        expressions=tar_exps.reshape(bs, n, 100),
                        return_verts=need_verts,
                        no_grad=True,
                    )
                    tar_joints, tar_verts = self._smplx_forward(
                        pose_aa_full=tar_pose.reshape(bs, n, 55, 3),
                        transl=zero_transl,
                        betas=tar_beta.reshape(bs, n, 300),
                        expressions=tar_exps.reshape(bs, n, 100),
                        return_verts=need_verts,
                        no_grad=True,
                    )
                    if getattr(self.args, "mpjpe_eval_enabled", True):
                        self._mpjpe_update(rec_joints, tar_joints)
                    if self.args.rec_ver_weight > 0:
                        pred_vertices = rec_verts
                        tar_vertices = tar_verts

                        vectices_loss = self.vectices_loss(pred_vertices, tar_vertices) * loss_timemask.view(1, -1, 1, 1)
                        vectices_loss = vectices_loss.mean()
                        self.tracker.update_meter("ver", "val", vectices_loss.item()*self.args.rec_weight * self.args.rec_ver_weight)

                    if self.args.rec_loc_weight > 0:
                        pred_joints = rec_joints
                        # tar_joints already (bs, n, 55, 3) from _smplx_forward
                        joint_loc_loss = self.loc_loss(pred_joints, tar_joints) * loss_timemask.view(1, -1, 1, 1)
                        self.tracker.update_meter("loc_rec", "val", joint_loc_loss.mean().item()*self.args.rec_weight * self.args.rec_loc_weight)
                        # g_loss_final += joint_loc_loss.mean()*self.args.rec_weight*self.args.rec_loc_weight
                # ---------------------- vqvae loss -------------------------- #
                upper_loss_embedding = upper_q_res.penalty
                if epoch < 1:
                    embedding_weight = 0
                else:
                    embedding_weight = min(1.0, epoch/self.args.kl_warmup_epochs) * self.args.comloss_weight
                # g_loss_final += upper_loss_embedding + lower_loss_embedding # * embedding_weight
                self.tracker.update_meter("com", "val", (upper_loss_embedding * embedding_weight).item()) # embedding_weight * 


                # ----------------------- MMD loss between two datasets -------------------------- #
                if self.args.mmd_weight > 0 and len(embody3d_indices) > 4 and len(beatx_indices) > 4:
                    # breakpoint()
                    emb3d_z = z_encoder[embody3d_indices].permute(0, 2, 1).contiguous().view(-1, z_encoder.shape[1])
                    beatx_z = z_encoder[beatx_indices].permute(0, 2, 1).contiguous().view(-1, z_encoder.shape[1])
                    mmd_loss = self.mmd_loss(emb3d_z, beatx_z)
                    mmd_loss = mmd_loss * self.args.mmd_weight
                    # g_loss_final += mmd_loss.to(g_loss_final.dtype)
                    self.tracker.update_meter("mmd", "val", mmd_loss.item())
                
                if its == 100:
                    break

                if self.args.debug and its == 10: break

            if getattr(self.args, "mpjpe_eval_enabled", True):
                self._mpjpe_log(epoch, split="val")

            if self.global_rank == 0:
                self.val_recording(epoch)
            
    def test(self, epoch, visualize=False, max_batches=None, save=False):
        results_save_path = os.path.join(self.checkpoint_path,  f"{epoch}/") 
        if os.path.exists(results_save_path) and self.args.is_train:
            return 0
        
        if visualize:
            import shutil
            import tempfile
            # render_smplx_debug_video + stitch_videos_hstack come from
            # the new dataloaders/utils path (imported at module top).

        from .utils.metrics import ReconMetrics
        
        if not os.path.exists(results_save_path):
            os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0

        recon_metrics = ReconMetrics(self.args)

        logger.info(f"Length of test set: {len(self.test_data)} samples in {len(self.test_loader)} batches")
        self.model.eval()
        with torch.no_grad():
            for its, dict_data in tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc=f"Testing at epoch {epoch}"):
                if max_batches is not None and its >= max_batches:
                    break
                tar_pose_upper = dict_data["motion_upper"]
                tar_pose_hands = dict_data["motion_hands"]
                
                tar_beta = dict_data["beta"].to(self.local_rank)
                # tar_trans = dict_data["trans"].to(self.local_rank)
                tar_trans = torch.zeros_like(dict_data["transl"]).to(self.local_rank)

                # breakpoint()
                

                tar_pose_upper = tar_pose_upper.to(self.local_rank)
                tar_pose_hands = tar_pose_hands.to(self.local_rank)
                sample_names = dict_data["filechunk_id"]
                file_name =  dict_data["file_id"]
                # if not sample_names[0].startswith("c--"):
                #     continue
                
                upper_hands_joint_mask = self.test_data.upper_mask_for_flattened + \
                    self.test_data.hands_mask_for_flattened

                bs, n, udim = tar_pose_upper.shape
                uj = udim // 3
                tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, uj, 3))
                tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, uj*6)

                hj = tar_pose_hands.shape[-1]
                hj = hj // 3
                tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, hj, 3))
                tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, hj*6)

                
                j = uj + hj 
                tar_pose = torch.cat([tar_pose_upper, tar_pose_hands], dim=-1)

                tar_exps = torch.zeros((bs, n, 100)).to(self.local_rank)

                # if not "fulllength" in self.args.dataset_ratio:
                remain = n%self.args.frame_chunk_size
                tar_pose = tar_pose[:, :n-remain, :]
                tar_pose_upper = tar_pose_upper[:, :n-remain, :]
                tar_pose_hands = tar_pose_hands[:, :n-remain, :]
                tar_trans = tar_trans[:, :n-remain, :]
                tar_exps = tar_exps[:, :n-remain, :]
                tar_beta = tar_beta[:, :n-remain, :]

                

                # if self.model.causal:
                #     pass
                # else:
                # breakpoint()
                out_final = None
                # out_trans = None
                framechunk_size = self.args.frame_chunk_size
                num_frames = tar_pose.shape[1]
                if "fulllength" in self.args.dataset_ratio and num_frames < tar_pose_upper.shape[1]:
                    tar_pose_upper = tar_pose_upper[:, :num_frames, :].clone()
                    tar_pose_hands = tar_pose_hands[:, :num_frames, :].clone()
                    tar_trans = tar_trans[:, :num_frames, :].clone()
                    tar_pose = tar_pose[:, :num_frames, :].clone()

                in_tar_pose = torch.cat((tar_pose_upper, tar_pose_hands), dim=-1)
                # motion_chunks = []
                # print(file_name[0])
                with self.model.streaming(batch_size=tar_pose.shape[0]):
                    for offset in range(0, num_frames, framechunk_size):
                        
                        # tar_pose_uppernew = tar_pose_upper[:,i*(self.args.pose_length):i*(self.args.pose_length)+self.args.pose_length,:].clone()
                        # tar_pose_handsnew = tar_pose_hands[:,i*(self.args.pose_length):i*(self.args.pose_length)+self.args.pose_length,:].clone()
                        frame = in_tar_pose[:, offset:offset+framechunk_size, :]
                        # print(offset, num_frames, frame.shape)
                        codes = self.model.encode(frame)
                        assert codes.shape[-1] == 1
                        rec_pose = self.model.decode(codes)
                        # self.model.eval()

                        n = rec_pose.shape[1]
                        assert n == framechunk_size

                        rec_pose = rec_pose.reshape(bs, n, j, 6)
                        rec_pose = rc.rotation_6d_to_matrix(rec_pose)#
                        rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs, n, j*3)
                        
                        # rec_trans = rec_trans.cpu().numpy()

                        if offset != 0:
                            out_final = torch.cat([out_final, rec_pose], dim=1)
                        else:
                            out_final = rec_pose
                

                # breakpoint()

                rec_pose = out_final
                assert num_frames == rec_pose.shape[1]
                assert rec_pose.shape[0] == tar_pose.shape[0] == bs == 1
                # assert num_frames == rec_trans.shape[1]
                rec_pose = rec_pose.reshape(bs*num_frames, j*3)
                # rec_trans = rec_trans.reshape(bs*num_frames, 3)
                
                tar_pose = rc.rotation_6d_to_matrix(in_tar_pose.reshape(bs, num_frames, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*tar_pose.shape[1], j*3)
                tar_pose = self.test_data.inverse_selection_tensor(tar_pose, upper_hands_joint_mask, tar_pose.shape[0])
                rec_pose = self.test_data.inverse_selection_tensor(rec_pose, upper_hands_joint_mask, rec_pose.shape[0])
                
                

                assert num_frames == tar_trans.shape[1]
                tar_trans = tar_trans.reshape(bs*num_frames, 3)
                
                
                total_length += rec_pose.shape[0]
                # --- save --- #
                # breakpoint() 
                rec_pose = rec_pose.reshape(num_frames, len(self.test_data.smplx_joint_names), 3)
                tar_pose = tar_pose.reshape(num_frames, len(self.test_data.smplx_joint_names), 3)
                
                sample_save_path = os.path.join(results_save_path, sample_names[0])
                if (save or visualize) and not os.path.exists(sample_save_path):
                    os.makedirs(sample_save_path)

                tar_exps = tar_exps.reshape(bs*num_frames, -1)
                tar_beta = tar_beta.reshape(bs*num_frames, -1)
                rec_exps = tar_exps
                rec_trans = tar_trans

                metric_dict = {
                    "rec_pose": rec_pose.reshape(num_frames, -1),
                    "rec_exps": rec_exps,
                    "rec_trans": rec_trans,
                    "tar_pose": tar_pose.reshape(num_frames, -1),
                    "tar_exps": tar_exps,
                    "tar_beta": tar_beta[0],
                    "tar_trans": tar_trans,
                    "file_id": file_name[0],
                }
            
                recon_metrics.update(metric_dict)

                rec_pose = rec_pose.cpu().numpy()
                tar_pose = tar_pose.cpu().numpy()

                tar_trans = tar_trans.cpu().numpy()

                tar_exps = tar_exps.cpu().numpy()
                tar_beta = tar_beta.cpu().numpy()
                # breakpoint()
                rec_out = {
                    "global_orient": rec_pose[:, 0].reshape(num_frames, -1),
                    "body_pose": rec_pose[:, 1:22].reshape(num_frames, -1),
                    "left_hand_pose": rec_pose[:, 25:40].reshape(num_frames, -1),
                    "right_hand_pose": rec_pose[:, 40:55].reshape(num_frames, -1),
                    "transl": tar_trans,
                    "betas": tar_beta,
                    "expression": tar_exps,
                    "jaw_pose": tar_pose[:, 22:23].reshape(num_frames, -1),
                    "leye_pose": tar_pose[:, 23:24].reshape(num_frames, -1),
                    "reye_pose": tar_pose[:, 24:25].reshape(num_frames, -1),
                }

                tar_out = {
                    "global_orient": tar_pose[:, 0].reshape(num_frames, -1),
                    "body_pose": tar_pose[:, 1:22].reshape(num_frames, -1),
                    "left_hand_pose": tar_pose[:, 25:40].reshape(num_frames, -1),
                    "right_hand_pose": tar_pose[:, 40:55].reshape(num_frames, -1),
                    "transl": tar_trans,
                    "betas": tar_beta,
                    "expression": tar_exps,
                    "jaw_pose": tar_pose[:, 22:23].reshape(num_frames, -1),
                    "leye_pose": tar_pose[:, 23:24].reshape(num_frames, -1),
                    "reye_pose": tar_pose[:, 24:25].reshape(num_frames, -1),
                }
                # breakpoint()

                if save:
                    np.savez(os.path.join(sample_save_path, 'gt.npz'),
                        betas=tar_beta,
                        poses=tar_pose,
                        expressions=tar_exps,
                        trans=tar_trans,
                        model='smplx',
                        gender='NEUTRAL_2020',
                        mocap_frame_rate=30,
                    )
                    np.savez(os.path.join(sample_save_path, 'pred.npz'),
                        betas=tar_beta,
                        poses=rec_pose,
                        expressions=tar_exps,
                        trans=tar_trans,
                        model='smplx',
                        gender='NEUTRAL_2020',
                        mocap_frame_rate=30,
                    )

                # --- render --- #
                if visualize:
                    logger.info(f"visualizing {sample_names[0]}")
                    final_path = os.path.join(sample_save_path, "gt_pred_compared.mp4")
                    with tempfile.TemporaryDirectory(prefix="upper_sbs_") as tmpdir:
                        gt_path = os.path.join(tmpdir, "gt.mp4")
                        pred_path = os.path.join(tmpdir, "pred.mp4")
                        stitched = os.path.join(tmpdir, "stitched.mp4")
                        # tar_trans is already zeroed earlier (L681), so no
                        # per-frame re-centering is needed before passing
                        # to render_smplx_debug_video.
                        render_smplx_debug_video(
                            smplx_model=self.smplx_model,
                            poses=tar_pose.reshape(num_frames, -1),
                            transl=tar_trans,
                            expressions=tar_exps,
                            betas=tar_beta,
                            output_path=gt_path,
                            fps=self.args.motion_fps,
                            mesh_color=(180, 54, 54, 255),
                        )
                        render_smplx_debug_video(
                            smplx_model=self.smplx_model,
                            poses=rec_pose.reshape(num_frames, -1),
                            transl=tar_trans,
                            expressions=tar_exps,
                            betas=tar_beta,
                            output_path=pred_path,
                            fps=self.args.motion_fps,
                            mesh_color=(36, 73, 156, 255),
                        )
                        stitch_videos_hstack([gt_path, pred_path], stitched)
                        if not os.path.exists(stitched):
                            raise RuntimeError(f"hstack failed; no output at {stitched}")
                        shutil.move(stitched, final_path)
                    logger.info(f"output saved to {final_path}")
                                                               
                # if its == 1:break
        end_time = time.time() - start_time
        recon_metrics.compute_metrics()
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.motion_fps)} s motion")
