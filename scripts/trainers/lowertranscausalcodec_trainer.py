import smplx
import os
import torch
import numpy as np
# import logging
from loguru import logger
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from torch.utils.tensorboard import SummaryWriter
import gc

from .basecausalcodec_trainer import BaseCausalCodecTrainer
from .utils import rotation_conversions as rc
from .utils.loss_factory import get_loss_func
from .utils import tools as other_tools


# logger = logging.getLogger()


class LowerBodyCausalCodecTrainer(BaseCausalCodecTrainer):
    def __init__(self, args):
        assert "embody" not in args.dataset_ratio.lower(), (
            f"LowerBodyCausalCodecTrainer cannot be used with embody3d data: "
            f"embody3d lower-body motion is flagged lower_valid=False in the "
            f"unified dataset because its quality is insufficient for codec "
            f"training. Got dataset_ratio={args.dataset_ratio!r}. "
            f"Use a BEATX-only ratio (e.g. 'full_beatx_lowervalid')."
        )
        super(LowerBodyCausalCodecTrainer, self).__init__(args)

        self.tracker = other_tools.EpochTracker(
            ["rec_6d", "rec_rot", "rec_aa", "contact", "transv", "trans", "vel", "ver", "com", "kl", "acc", "loc_rec", "loc_vel", "loc_acc", "foot", "laploss6d", "laplossloc", "globalorient_smooth6d", "globalorient_smoothloc", "transl_smooth"], 
            [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
        )

        self.rec_loss = get_loss_func("GeodesicLoss", reduction='none')
        # TODO: add specific loss for contact and foot contact
        self.rot6d_loss = torch.nn.MSELoss(reduction='none')
        self.aa_loss = torch.nn.MSELoss(reduction='none')
        self.vel_loss = torch.nn.MSELoss(reduction='none')
        self.vectices_loss = torch.nn.MSELoss(reduction='none')
        self.loc_loss = torch.nn.MSELoss(reduction='none')
        self.loc_vel_loss = torch.nn.MSELoss(reduction='none')
        self.loc_acc_loss = torch.nn.MSELoss(reduction='none')
        self.trans_loss = torch.nn.MSELoss(reduction='none')
        self.laplacian_loss = get_loss_func("laplacian_loss", kernel_size=5, reduction='none')
        self.smoothness_loss = get_loss_func("smoothness_loss", lambda_vel=self.args.accel_penalty_weight, reduction='mean')

        self.param_dtype = getattr(torch, args.param_dtype)
        self.optim_dtype = getattr(torch, args.optim_dtype)

        assert self.param_dtype == torch.float32, "param_dtype must be float32"

    def train(self, epoch):
        for p in self.model.parameters():
            if torch.isnan(p).any():
                raise ValueError("nan in model parameters")
        
        gc.collect()
        self.model.train()
        # self.vq0_codec.eval()
        t_start = time.time()
        if self.args.is_continue and epoch != 0:
            self.opt_s.step(epoch)
        self.tracker.reset()
        for its, dict_data in enumerate(self.train_loader):
            
            tar_pose_lower = dict_data["motion_lower"]
            tar_beta = dict_data["beta"].to(self.local_rank)
            tar_trans = dict_data["transl"].to(self.local_rank)
            tar_contact = dict_data["contact"].to(self.local_rank)

            lower_joint_mask = self.train_data.lower_mask_for_flattened 

            # tar_trans_vel_x = other_tools.estimate_linear_velocity(
            #     tar_trans[:, :, 0:1], dt=1 / self.args.motion_fps
            # )
            # tar_trans_vel_z = other_tools.estimate_linear_velocity(
            #     tar_trans[:, :, 2:3], dt=1 / self.args.motion_fps
            # )
            # tar_trans_vel = torch.cat((tar_trans_vel_x, tar_trans[:, :, 1:2], tar_trans_vel_z), dim=-1)
            # tar_trans = tar_trans_vel

            lower_valid_mask = dict_data["lower_valid_mask"].to(self.local_rank) 
            # breakpoint()
            assert not (lower_valid_mask == 0).any(), "Lower valid mask should not be zero in training"
            

            # subtract the first frame translation
            tar_trans[:, :, 0] = tar_trans[:, :, 0] - tar_trans[:, 0:1, 0]
            tar_trans[:, :, 2] = tar_trans[:, :, 2] - tar_trans[:, 0:1, 2]

            tar_trans_vel = other_tools.estimate_linear_velocity(
                tar_trans, dt=1/self.args.motion_fps 
            )
            # breakpoint()

            tar_pose_lower = tar_pose_lower.to(self.local_rank)
            
            # breakpoint()

            bs, n, ldim = tar_pose_lower.shape
            lj = ldim //3
            tar_pose_lower = rc.axis_angle_to_matrix(tar_pose_lower.reshape(bs, n, lj, 3))
            tar_pose_lower = rc.matrix_to_rotation_6d(tar_pose_lower).reshape(bs, n, lj*6)

            j = lj
            tar_pose = tar_pose_lower
            tar_exps = torch.zeros((bs, n, 100)).to(self.local_rank)

            t_data = time.time() - t_start
            
            self.opt.zero_grad()
            g_loss_final = 0
            # breakpoint()

            # in_tar_pose_lower = torch.cat((tar_pose_lower, tar_trans, tar_contact), dim=-1)
            # tar_trans_vel = torch.zeros_like(tar_trans_vel)
            # tar_contact = torch.zeros_like(tar_contact)
            # tar_trans = torch.zeros_like(tar_trans)
            in_tar_pose_lower = torch.cat((tar_pose_lower, tar_trans_vel, tar_contact), dim=-1)

            # with torch.no_grad():
            #     # breakpoint()
            #     additional_outs = self.teacher.encode_motion(in_tar_pose)
            #     teacher_features = additional_outs[-1].clone()


            # breakpoint()
            # check tar_pose dim and netout dim # check tar/contact and joint seperation
            net_out, lower_q_res, z_sample, loss_timemask = self.model(in_tar_pose_lower)
            
            # # -------------- teacher loss -----------#
            # assert self.model.proj_vq, "Cant train with distillation without projection of vq0"
            # # breakpoint()
            # sem_emb = quantize_res.sem_emb
            # sem_emb = sem_emb.permute(0, 2, 1)
            # sem_emb = self.model.vq_projector(sem_emb)
            # teacher_loss = self.teacher_loss(sem_emb, teacher_features)
            # teacher_loss = teacher_loss.mean() * self.args.teacherguidance_weight
            # # teacher_loss = teacher_loss.mean()
            # self.tracker.update_meter("teacherloss", "train", teacher_loss.item())
            # g_loss_final += teacher_loss
            
            rec_pose = net_out[:, :, :j*6]
            

            # 6d loss
            # fileids = dict_data["filechunk_id"]
            rec_pose = rec_pose.reshape(bs, n, j, 6)
            tar_pose = tar_pose.reshape(bs, n, j, 6)
            loss_6d = self.rot6d_loss(rec_pose, tar_pose) # * loss_timemask.view(1, -1, 1, 1)
            loss_6d[:, :, :1] = loss_6d[:, :, :1] * self.args.globalorient_weight
            loss_6d = loss_6d.mean() * self.args.rec_weight * self.args.rec_6d_weight
            # if loss_6d.item() > 3: print ("->", f"loss {loss_6d.item()}", fileids); logger.info(f"-> loss {loss_6d.item()} @{fileids}"); breakpoint();
            self.tracker.update_meter("rec_6d", "train", loss_6d.item())
            g_loss_final += loss_6d

            if rec_pose.shape[1] <=6: 
                laploss_6d = torch.zeros_like(loss_6d)
            else:
                laploss_6d = self.laplacian_loss(rec_pose.reshape(bs, n, j*6), tar_pose.reshape(bs, n, j*6))
                laploss_6d[:, :, :3] = laploss_6d[:, :, :3] * self.args.globalorient_weight
            laploss_6d = laploss_6d.mean() * self.args.rec_weight * self.args.lap_weight
            self.tracker.update_meter("laploss6d", "train", laploss_6d.item())
            g_loss_final += laploss_6d


            # breakpoint()
            # smoothness loss on 6d on global orient
            if rec_pose.shape[1] <=6: 
                smoothness_6d = torch.tensor(0.).to(rec_pose)
            else:
                smoothness_6d = self.smoothness_loss(rec_pose[:, :, :1].reshape(bs, n, 6))
            
            g_loss_final += smoothness_6d
            self.tracker.update_meter("globalorient_smooth6d", "train", smoothness_6d.item())


            
            # rotation matrix loss
            rec_pose = rc.rotation_6d_to_matrix(rec_pose) # * loss_timemask.view(1, -1, 1, 1, 1)
            tar_pose = rc.rotation_6d_to_matrix(tar_pose) # * loss_timemask.view(1, -1, 1, 1, 1)
            loss_rec_rot = self.rec_loss(rec_pose, tar_pose) 
            # breakpoint()
            loss_rec_rot = loss_rec_rot.reshape(bs, n, j, 1)
            loss_rec_rot[:, :, :1] = loss_rec_rot[:, :, :1] * self.args.globalorient_weight
            loss_rec_rot = loss_rec_rot.mean() * self.args.rec_weight * self.args.rec_pos_weight
            self.tracker.update_meter("rec_rot", "train", loss_rec_rot.item())
            g_loss_final += loss_rec_rot

            # axis angle loss
            rec_pose_aa = rc.matrix_to_axis_angle(rec_pose)
            tar_pose_aa = rc.matrix_to_axis_angle(tar_pose)
            loss_rec_aa = self.aa_loss(rec_pose_aa, tar_pose_aa) #* loss_timemask.view(1, -1, 1, 1)
            loss_rec_aa[:, :, :1] = loss_rec_aa[:, :, :1] * self.args.globalorient_weight
            loss_rec_aa = loss_rec_aa.mean() * self.args.rec_weight * self.args.rec_aa_weight
            self.tracker.update_meter("rec_aa", "train", loss_rec_aa.item())
            g_loss_final += loss_rec_aa

            # contact loss
            # breakpoint() d
            rec_contact = net_out[:, :, j*6+3:j*6+7]
            loss_contact = self.vectices_loss(rec_contact, tar_contact) #* loss_timemask.view(1, -1, 1)
            loss_contact = loss_contact.mean() * self.args.rec_weight * self.args.rec_contact_weight
            self.tracker.update_meter("contact", "train", loss_contact.item())
            g_loss_final += loss_contact

            # ---------------------- translation losses -------------------------- #
            # rec_trans = net_out[:, :, j*6+0:j*6+3]
            # rec_x_trans = other_tools.velocity2position(
            #     rec_trans[:, :, 0:1], 1 / self.args.motion_fps, tar_trans[:, 0, 0:1]
            # )
            # rec_z_trans = other_tools.velocity2position(
            #     rec_trans[:, :, 2:3], 1 / self.args.motion_fps, tar_trans[:, 0, 2:3]
            # )
            # rec_y_trans = rec_trans[:, :, 1:2]
            # rec_xyz_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
            # ----
            # rec_trans_vel_x = other_tools.estimate_linear_velocity(
            #     rec_trans[:, :, 0:1], dt=1 / self.args.motion_fps
            # )
            # rec_trans_vel_z = other_tools.estimate_linear_velocity(
            #     rec_trans[:, :, 2:3], dt=1 / self.args.motion_fps
            # )
            # rec_trans_vel = torch.cat([rec_trans_vel_x, torch.zeros_like(rec_trans_vel_x), rec_trans_vel_z], dim=-1)
            # -----
            rec_trans_vel = net_out[:, :, j*6+0:j*6+3]
            rec_trans, final_pos = other_tools.velocity2position_mixeddiff(
                rec_trans_vel, dt=1/self.args.motion_fps, init_pos=tar_trans[:, 0]
            )
            rec_xyz_trans = rec_trans


            # --
            # loss_vel_x = self.vel_loss(rec_trans_vel[:, :, 0:1], tar_trans_vel_x) * loss_timemask.view(1, -1, 1)
            # loss_vel_z = self.vel_loss(rec_trans_vel[:, :, 2:3], tar_trans_vel_z) * loss_timemask.view(1, -1, 1)
            # loss_trans_vel = loss_vel_x.mean() + loss_vel_z.mean()
            # --
            loss_trans_vel = self.vel_loss(rec_trans_vel, tar_trans_vel) #* loss_timemask.view(1, -1, 1)
            loss_trans_vel = loss_trans_vel.mean()
            # ---
            loss_trans_vel = loss_trans_vel * self.args.rec_weight * self.args.rec_trans_weight

            self.tracker.update_meter("transv", "train", loss_trans_vel.item())
            g_loss_final += loss_trans_vel
            # breakpoint()
            loss_trans = self.trans_loss(rec_xyz_trans, tar_trans) #* loss_timemask.view(1, -1, 1)
            loss_trans = loss_trans.mean() * self.args.rec_weight * self.args.rec_trans_weight
            self.tracker.update_meter("trans", "train", loss_trans.item())
            g_loss_final += loss_trans


            # smoothness loss on translation
            if rec_pose.shape[1] <=6: 
                transl_smoothness = torch.tensor(0.).to(rec_xyz_trans)
            else:
                transl_smoothness = self.smoothness_loss(rec_xyz_trans)
            self.tracker.update_meter("transl_smooth", "train", transl_smoothness.item())
            g_loss_final += transl_smoothness

            # ----
            # v3 = (
            #     # x axis
            #     (self.vel_loss(
            #         rec_trans_vel[:, :, 0:1][:, 1:] - rec_trans_vel[:, :, 0:1][:, :-1],
            #         tar_trans_vel_x[:, 1:] - tar_trans_vel_x[:, :-1],
            #     ) * loss_timemask[1:].view(1, -1, 1)).mean()
            #     * self.args.rec_weight
            #     # z axis
            #     + (self.vel_loss(
            #         rec_trans_vel[:, :, 2:3][:, 1:] - rec_trans_vel[:, :, 2:3][:, :-1],
            #         tar_trans_vel_z[:, 1:] - tar_trans_vel_z[:, :-1],
            #     ) * loss_timemask[1:].view(1, -1, 1)).mean()
            #     * self.args.rec_weight
            # )
            # a3 = (
            #     # x axis
            #     (self.vel_loss(
            #         rec_trans_vel[:, :, 0:1][:, 2:]
            #         + rec_trans_vel[:, :, 0:1][:, :-2]
            #         - 2 * rec_trans_vel[:, :, 0:1][:, 1:-1],
            #         tar_trans_vel_x[:, 2:]
            #         + tar_trans_vel_x[:, :-2]
            #         - 2 * tar_trans_vel_x[:, 1:-1],
            #     )* loss_timemask[2:].view(1, -1, 1)).mean()
            #     * self.args.rec_weight
            #     # z axis
            #     + (self.vel_loss(
            #         rec_trans_vel[:, :, 2:3][:, 2:]
            #         + rec_trans_vel[:, :, 2:3][:, :-2]
            #         - 2 * rec_trans_vel[:, :, 2:3][:, 1:-1],
            #         tar_trans_vel_z[:, 2:]
            #         + tar_trans_vel_z[:, :-2]
            #         - 2 * tar_trans_vel_z[:, 1:-1],
            #     )* loss_timemask[2:].view(1, -1, 1)).mean()
            #     * self.args.rec_weight
            # )
            # ----
            v3 = (
                (self.vel_loss(
                    rec_trans_vel[:, 1:] - rec_trans_vel[:, :-1],
                    tar_trans_vel[:, 1:] - tar_trans_vel[:, :-1],
                ) #* loss_timemask[1:].view(1, -1, 1)
                ).mean()
                * self.args.rec_weight
            )
            a3 = (
                # x axis
                (self.vel_loss(
                    rec_trans_vel[:, 2:]
                    + rec_trans_vel[:, :-2]
                    - 2 * rec_trans_vel[:, 1:-1],
                    tar_trans_vel[:, 2:]
                    + tar_trans_vel[:, :-2]
                    - 2 * tar_trans_vel[:, 1:-1],
                )#* loss_timemask[2:].view(1, -1, 1)
                ).mean()
                * self.args.rec_weight
            )
            # ----
            if rec_pose.shape[1] <=6: v3, a3 = torch.zeros_like(v3), torch.zeros_like(v3)
            if torch.isnan(v3) or torch.isnan(a3):
                breakpoint()
            g_loss_final += v3 * self.args.rec_trans_weight
            g_loss_final += a3 * self.args.rec_trans_weight

            # overall translation velocity and acceleration loss
            v2 = (
                (self.vel_loss(
                    rec_xyz_trans[:, 1:] - rec_xyz_trans[:, :-1],
                    tar_trans[:, 1:] - tar_trans[:, :-1],
                ) #* loss_timemask[1:].view(1, -1, 1)
                ).mean()
                * self.args.rec_weight
            )
            a2 = (
                (self.vel_loss(
                    rec_xyz_trans[:, 2:]
                    + rec_xyz_trans[:, :-2]
                    - 2 * rec_xyz_trans[:, 1:-1],
                    tar_trans[:, 2:] + tar_trans[:, :-2] - 2 * tar_trans[:, 1:-1],
                ) #* loss_timemask[2:].view(1, -1, 1)
                ).mean()
                * self.args.rec_weight
            )
            if rec_pose.shape[1] <=6: v2, a2 = torch.zeros_like(v2), torch.zeros_like(v2)
            if torch.isnan(v2) or torch.isnan(a2):
                breakpoint()
            g_loss_final += v2 * self.args.rec_trans_weight
            g_loss_final += a2 * self.args.rec_trans_weight

            # ---------------------------------- #
            # breakpoint()

            # velocity loss and acceleration loss
            velocity_loss =  self.vel_loss(rec_pose[:, 1:] - rec_pose[:, :-1], tar_pose[:, 1:] - tar_pose[:, :-1]) #* loss_timemask[1:].view(1, -1, 1, 1, 1)
            velocity_loss[:, :, :1] = velocity_loss[:, :, :1] * self.args.globalorient_weight
            velocity_loss = velocity_loss.mean() * self.args.rec_weight * 30
            
            if rec_pose.shape[1] <=6:
                acceleration_loss = torch.zeros_like(velocity_loss)
            else:
                acceleration_loss =  self.vel_loss(rec_pose[:, 2:] + rec_pose[:, :-2] - 2 * rec_pose[:, 1:-1], tar_pose[:, 2:] + tar_pose[:, :-2] - 2 * tar_pose[:, 1:-1]) #* loss_timemask[2:].view(1, -1, 1, 1, 1)
                acceleration_loss[:, :, :1] = acceleration_loss[:, :, :1] * self.args.globalorient_weight
                acceleration_loss = acceleration_loss.mean() * self.args.rec_weight * 30

            self.tracker.update_meter("vel", "train", velocity_loss.item())
            self.tracker.update_meter("acc", "train", acceleration_loss.item())
            if torch.isnan(acceleration_loss):
                breakpoint()

            g_loss_final += velocity_loss
            g_loss_final += acceleration_loss

            # vertices loss and joint location loss
            if self.args.rec_ver_weight > 0 or self.args.rec_loc_weight > 0:
                tar_pose = tar_pose_aa.reshape(bs*n, j*3)
                rec_pose = rec_pose_aa.reshape(bs*n, j*3)
                rec_pose = self.train_data.inverse_selection_tensor(rec_pose, lower_joint_mask, rec_pose.shape[0])
                tar_pose = self.train_data.inverse_selection_tensor(tar_pose, lower_joint_mask, tar_pose.shape[0])
                need_verts = self.args.rec_ver_weight > 0
                rec_joints, rec_verts = self._smplx_forward(
                    pose_aa_full=rec_pose.reshape(bs, n, 55, 3),
                    transl=rec_xyz_trans.reshape(bs, n, 3),
                    betas=tar_beta.reshape(bs, n, 300),
                    expressions=tar_exps.reshape(bs, n, 100),
                    return_verts=need_verts,
                )
                tar_joints, tar_verts = self._smplx_forward(
                    pose_aa_full=tar_pose.reshape(bs, n, 55, 3),
                    transl=tar_trans.reshape(bs, n, 3),
                    betas=tar_beta.reshape(bs, n, 300),
                    expressions=tar_exps.reshape(bs, n, 100),
                    return_verts=need_verts,
                )

                # vertices loss
                if self.args.rec_ver_weight > 0:
                    pred_vertices = rec_verts
                    tar_vertices = tar_verts

                    vectices_loss = self.vectices_loss(pred_vertices, tar_vertices) #* loss_timemask.view(1, -1, 1, 1)
                    vectices_loss = vectices_loss.mean()
                    self.tracker.update_meter("ver", "train", vectices_loss.item()*self.args.rec_weight * self.args.rec_ver_weight)
                    g_loss_final += vectices_loss.mean()*self.args.rec_weight*self.args.rec_ver_weight
                    # breakpoint()
                    vertices_vel_loss = self.vel_loss(pred_vertices[:, 1:] - pred_vertices[:, :-1], tar_vertices[:, 1:] - tar_vertices[:, :-1]) #* loss_timemask[1:].view(1, -1, 1, 1)
                    # vertices_vel_loss[:, :, :1] = vertices_vel_loss[:, :, :1] * self.args.globalorient_weight
                    vertices_vel_loss = vertices_vel_loss.mean() * self.args.rec_weight * 10
                    
                    if pred_vertices.shape[1] <=6: 
                        vertices_acc_loss = torch.zeros_like(vertices_vel_loss)
                    else:
                        vertices_acc_loss = self.vel_loss(pred_vertices[:, 2:] + pred_vertices[:, :-2] - 2 * pred_vertices[:, 1:-1], tar_vertices[:, 2:] + tar_vertices[:, :-2] - 2 * tar_vertices[:, 1:-1]) #* loss_timemask[2:].view(1, -1, 1, 1)
                        # vertices_acc_loss[:, :, :1] = vertices_acc_loss[:, :, :1] * self.args.globalorient_weight
                    vertices_acc_loss = vertices_acc_loss.mean() * self.args.rec_weight * 10
                    
                    if pred_vertices.shape[1] <=6:
                        vertices_lap_loss = torch.zeros_like(vertices_vel_loss)
                    else:
                        vertices_lap_loss = self.laplacian_loss(pred_vertices.reshape(bs, n, -1), tar_vertices.reshape(bs, n, -1))
                     
                    # vertices_lap_loss[:, :, :1] = vertices_lap_loss[:, :, :1] * self.args.globalorient_weight
                    vertices_lap_loss = vertices_lap_loss.mean() * self.args.rec_weight * 10
                    g_loss_final += vertices_lap_loss #* self.args.rec_weight * self.args.rec_ver_weight
                    g_loss_final += vertices_vel_loss #* self.args.rec_weight * self.args.rec_ver_weight
                    g_loss_final += vertices_acc_loss #* self.args.rec_weight * self.args.rec_ver_weight

                # joint location/position loss
                if self.args.rec_loc_weight > 0:
                    pred_joints = rec_joints
                    # tar_joints already (bs, n, 55, 3) from _smplx_forward
                    pred_vel = pred_joints[:, 1:] - pred_joints[:, :-1]
                    tar_vel = tar_joints[:, 1:] - tar_joints[:, :-1]

                    pred_acc_joints = pred_joints[:, 2:] + pred_joints[:, :-2] - 2 * pred_joints[:, 1:-1]
                    tar_acc_joints = tar_joints[:, 2:] + tar_joints[:, :-2] - 2 * tar_joints[:, 1:-1]
                    # breakpoint()
                    joint_loc_loss = self.loc_loss(pred_joints, tar_joints) #* loss_timemask.view(1, -1, 1, 1)
                    # joint_loc_loss[:, :, :1] = joint_loc_loss[:, :, :1] * self.args.globalorient_weight
                    self.tracker.update_meter("loc_rec", "train", joint_loc_loss.mean().item()*self.args.rec_weight * self.args.rec_loc_weight)
                    g_loss_final += joint_loc_loss.mean()*self.args.rec_weight*self.args.rec_loc_weight

                    joint_vel_loss = self.loc_vel_loss(pred_vel, tar_vel) #* loss_timemask[1:].view(1, -1, 1, 1)
                    joint_vel_loss[:, :, :1] = joint_vel_loss[:, :, :1] * self.args.globalorient_weight
                    self.tracker.update_meter("loc_vel", "train", joint_vel_loss.mean().item()*self.args.rec_weight * self.args.rec_loc_weight * 30)
                    g_loss_final += joint_vel_loss.mean()*self.args.rec_weight*self.args.rec_loc_weight * 30

                    if pred_joints.shape[1] <=6: 
                        joint_acc_loss = torch.zeros_like(joint_vel_loss)
                    else:
                        joint_acc_loss =  self.loc_acc_loss(pred_acc_joints, tar_acc_joints) #* loss_timemask[2:].view(1, -1, 1, 1)
                        joint_acc_loss[:, :, :1] = joint_acc_loss[:, :, :1] * self.args.globalorient_weight
                        # if torch.isnan(joint_acc_loss).any():
                        #     breakpoint()
                            
                    # if torch.isnan(joint_acc_loss.mean()):
                    #     breakpoint()
                    # print(joint_acc_loss.mean())
                    g_loss_final += joint_acc_loss.mean()*self.args.rec_weight*self.args.rec_loc_weight * 30
                    self.tracker.update_meter("loc_acc", "train", joint_acc_loss.mean().item()*self.args.rec_weight * self.args.rec_loc_weight * 30)

                    if pred_joints.shape[1] <=6: 
                        joint_lap_loss = torch.zeros_like(joint_vel_loss)
                    else:
                        joint_lap_loss = self.laplacian_loss(pred_joints.reshape(bs, n, -1), tar_joints.reshape(bs, n, -1))
                    
                        joint_lap_loss[:, :, :1] = joint_lap_loss[:, :, :1] * self.args.globalorient_weight
                    g_loss_final += joint_lap_loss.mean() * self.args.rec_weight * self.args.rec_loc_weight * 30
                    self.tracker.update_meter("laplossloc", "train", joint_lap_loss.mean().item()*self.args.rec_weight * self.args.rec_loc_weight * 30)

                    if pred_joints.shape[1] <=6: 
                        smoothness_jointpos = torch.tensor(0.).to(pred_joints)
                    else:
                        smoothness_jointpos = self.smoothness_loss(pred_joints[:, :, :1].reshape(bs, n, 3))
                    
                    self.tracker.update_meter("globalorient_smoothloc", "train", smoothness_jointpos.item())
                    g_loss_final += smoothness_jointpos

                    # foot contact loss
                    foot_idx = [7, 8, 10, 11]
                    model_contact = net_out[:, :, j*6+3:j*6+7]
                    # find static indices consistent with model's own predictions
                    static_idx = model_contact > 0.98  # N x S x 4
                    # print(model_contact,static_idx)
                    model_feet = pred_joints[:, :, foot_idx]  # foot positions (N, S, 4, 3)
                    model_foot_v = torch.zeros_like(model_feet)
                    model_foot_v[:, :-1] = (
                        model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
                    )  # (N, S-1, 4, 3)
                    model_foot_v[~static_idx] = 0
                    # breakpoint()
                    foot_loss = self.vel_loss(
                        model_foot_v, torch.zeros_like(model_foot_v)
                    ) #* loss_timemask.view(1, -1, 1, 1)
                    self.tracker.update_meter("foot", "train", foot_loss.mean().item()*self.args.rec_weight * self.args.rec_loc_weight*20)
                    g_loss_final += foot_loss.mean()*self.args.rec_weight*self.args.rec_loc_weight*20
            

            # ---------------------- vqvae loss -------------------------- #
            # breakpoint()
            lower_loss_embedding = lower_q_res.penalty
            if epoch < 1:
                embedding_weight = 0
            else:
                embedding_weight = min(1.0, epoch/self.args.kl_warmup_epochs) * self.args.comloss_weight
            g_loss_final += lower_loss_embedding * embedding_weight
            self.tracker.update_meter("com", "train", (lower_loss_embedding * embedding_weight).item()) # embedding_weight * 
            
            
            g_loss_final.backward()
            # breakpoint()
            # for name, param in self.model.named_parameters():
                # print(name, param.grad.data.norm())
            # breakpoint()
            if self.args.grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.opt.step()

            if self.args.lr_policy == "onecyclelr":
                self.opt_s.step()
            
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            if its % self.args.log_period == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g)   
                if its % (self.args.log_period * 10) == 0:
                    metric_print = f"[GPU{self.global_rank}:{self.local_rank}]" + str({k: round(v.item(), 3) for k, v in lower_q_res.metrics.items()})
                    logger.info(metric_print)
            if self.args.debug:
                if its == 2: break
        torch.cuda.synchronize()
        # breakpoint()
        if self.args.lr_policy != "onecyclelr":
            self.opt_s.step(epoch)
                    
    def val(self, epoch):
        self.model.eval()
        # self.vq0_codec.eval()
        # self.visualize_codebook(self.checkpoint_path, epoch)
        t_start = time.time()
        self._mpjpe_reset()
        with torch.no_grad():
            for its, dict_data in enumerate(self.val_loader):

                tar_pose_lower = dict_data["motion_lower"]
                tar_beta = dict_data["beta"].to(self.local_rank)
                tar_trans = dict_data["transl"].to(self.local_rank)
                tar_contact = dict_data["contact"].to(self.local_rank)
                lower_joint_mask = self.train_data.lower_mask_for_flattened 

                # tar_trans_vel_x = other_tools.estimate_linear_velocity(
                #     tar_trans[:, :, 0:1], dt=1 / self.args.motion_fps
                # )
                # tar_trans_vel_z = other_tools.estimate_linear_velocity(
                #     tar_trans[:, :, 2:3], dt=1 / self.args.motion_fps
                # )
                # # tar_trans_vel = torch.cat((tar_trans_vel_x, tar_trans[:, :, 1:2], tar_trans_vel_z), dim=-1)
                # # tar_trans = tar_trans_vel

                # subtract the first frame translation
                tar_trans[:, :, 0] = tar_trans[:, :, 0] - tar_trans[:, 0:1, 0]
                tar_trans[:, :, 2] = tar_trans[:, :, 2] - tar_trans[:, 0:1, 2]

                tar_trans_vel = other_tools.estimate_linear_velocity(
                    tar_trans, dt=1/self.args.motion_fps, 
                )

                tar_pose_lower = tar_pose_lower.to(self.local_rank)
                
                # breakpoint()

                bs, n, ldim = tar_pose_lower.shape
                lj = ldim //3
                tar_pose_lower = rc.axis_angle_to_matrix(tar_pose_lower.reshape(bs, n, lj, 3))
                tar_pose_lower = rc.matrix_to_rotation_6d(tar_pose_lower).reshape(bs, n, lj*6)

                j = lj
                tar_pose = tar_pose_lower
                tar_exps = torch.zeros((bs, n, 100)).to(self.local_rank)
                t_data = time.time() - t_start
                # breakpoint()

                # input: pose, trans, contact -> output: pose only
                # in_tar_pose_lower = torch.cat((tar_pose_lower, tar_trans, tar_contact), dim=-1)
                # tar_trans_vel = torch.zeros_like(tar_trans_vel)
                # tar_contact = torch.zeros_like(tar_contact)
                # tar_trans = torch.zeros_like(tar_trans)
                in_tar_pose_lower = torch.cat((tar_pose_lower, tar_trans_vel, tar_contact), dim=-1)

                # with torch.no_grad():
                #     # breakpoint()
                #     additional_outs = self.teacher.encode_motion(in_tar_pose)
                #     teacher_features = additional_outs[-1].clone()

                # breakpoint() # check tar_pose dim and netout dim # check tar/contact and joint seperation
                net_out, lower_q_res, z_sample, loss_timemask = self.model(in_tar_pose_lower)
                
                # # -------------------- teacher loss -----------#
                # # assert self.model.proj_vq, "Cant train with distillation without projection of vq0"
                # # # breakpoint()
                # sem_emb = quantize_res.sem_emb
                # sem_emb = sem_emb.permute(0, 2, 1)
                # sem_emb = self.model.vq_projector(sem_emb)
                # teacher_loss = self.teacher_loss(sem_emb, teacher_features)
                # teacher_loss = teacher_loss.mean() * self.args.teacherguidance_weight
                # # teacher_loss = teacher_loss.mean()
                # self.tracker.update_meter("teacherloss", "val", teacher_loss.item())
                # # g_loss_final += teacher_loss

                rec_pose = net_out[:, :, :j*6]

                # 6d loss
                rec_pose = rec_pose.reshape(bs, n, j, 6)
                tar_pose = tar_pose.reshape(bs, n, j, 6)
                loss_6d = self.rot6d_loss(rec_pose, tar_pose) #* loss_timemask.view(1, -1, 1, 1)
                loss_6d = loss_6d.mean() * self.args.rec_weight * self.args.rec_6d_weight
                self.tracker.update_meter("rec_6d", "val", loss_6d.item())
                # g_loss_final += loss_6d

                # rotation matrix loss
                rec_pose = rc.rotation_6d_to_matrix(rec_pose)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose)
                loss_rec_rot = self.rec_loss(rec_pose, tar_pose) #* loss_timemask.view(1, -1, 1, 1, 1)
                loss_rec_rot = loss_rec_rot.mean() * self.args.rec_weight * self.args.rec_pos_weight
                self.tracker.update_meter("rec_rot", "val", loss_rec_rot.item())
                # g_loss_final += loss_rec_rot

                # axis angle loss
                rec_pose_aa = rc.matrix_to_axis_angle(rec_pose)
                tar_pose_aa = rc.matrix_to_axis_angle(tar_pose)
                loss_rec_aa = self.aa_loss(rec_pose_aa, tar_pose_aa) # * loss_timemask.view(1, -1, 1, 1)
                loss_rec_aa = loss_rec_aa.mean() * self.args.rec_weight * self.args.rec_aa_weight
                self.tracker.update_meter("rec_aa", "val", loss_rec_aa.item())
                # g_loss_final += loss_rec_aa

                # contact loss
                # breakpoint() d
                rec_contact = net_out[:, :, j*6+3:j*6+7]
                loss_contact = self.vectices_loss(rec_contact, tar_contact) # * loss_timemask.view(1, -1, 1)
                loss_contact = loss_contact.mean() * self.args.rec_weight * self.args.rec_contact_weight
                self.tracker.update_meter("contact", "val", loss_contact.item())
                # g_loss_final += loss_contact

                # ---------------------- translation losses -------------------------- #
                # rec_trans = net_out[:, :, j*6+0:j*6+3]
                # # rec_x_trans = other_tools.velocity2position(
                # #     rec_trans[:, :, 0:1], 1 / self.args.motion_fps, tar_trans[:, 0, 0:1]
                # # )
                # # rec_z_trans = other_tools.velocity2position(
                # #     rec_trans[:, :, 2:3], 1 / self.args.motion_fps, tar_trans[:, 0, 2:3]
                # # )
                # # rec_y_trans = rec_trans[:, :, 1:2]
                # # rec_xyz_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
                # rec_trans_vel_x = other_tools.estimate_linear_velocity(
                #     rec_trans[:, :, 0:1], dt=1 / self.args.motion_fps
                # )
                # rec_trans_vel_z = other_tools.estimate_linear_velocity(
                #     rec_trans[:, :, 2:3], dt=1 / self.args.motion_fps
                # )
                # rec_trans_vel = torch.cat([rec_trans_vel_x, torch.zeros_like(rec_trans_vel_x), rec_trans_vel_z], dim=-1)
                # ----
                rec_trans_vel = net_out[:, :, j*6+0:j*6+3]
                rec_trans, final_pos = other_tools.velocity2position_mixeddiff(
                    rec_trans_vel, 1/self.args.motion_fps, init_pos=tar_trans[:, 0]
                )
                rec_xyz_trans = rec_trans

                # ---
                # loss_vel_x = self.vel_loss(rec_trans_vel[:, :, 0:1], tar_trans_vel_x) * loss_timemask.view(1, -1, 1)
                # loss_vel_z = self.vel_loss(rec_trans_vel[:, :, 2:3], tar_trans_vel_z) * loss_timemask.view(1, -1, 1)
                # loss_trans_vel = loss_vel_x.mean() + loss_vel_z.mean()
                # ---
                loss_trans_vel = self.vel_loss(rec_trans_vel, tar_trans_vel) #* loss_timemask.view(1, -1, 1)
                loss_trans_vel = loss_trans_vel.mean()
                # ---

                loss_trans_vel = loss_trans_vel * self.args.rec_weight * self.args.rec_trans_weight

                self.tracker.update_meter("transv", "val", loss_trans_vel.item())
                # g_loss_final += loss_trans_vel
                loss_trans = self.trans_loss(rec_xyz_trans, tar_trans) #* loss_timemask.view(1, -1, 1)
                loss_trans = loss_trans.mean() * self.args.rec_weight * self.args.rec_trans_weight
                self.tracker.update_meter("trans", "val", loss_trans.item())
                # g_loss_final += loss_trans

                # smoothness loss on translation
                if rec_pose.shape[1] <=6:
                    transl_smoothness = torch.tensor(0.).to(rec_xyz_trans)
                else:
                    transl_smoothness = self.smoothness_loss(rec_xyz_trans)
                self.tracker.update_meter("transl_smooth", "val", transl_smoothness.item())
                # g_loss_final += transl_smoothness

                 # vertices loss / joint loc loss / MPJPE
                if (
                    self.args.rec_ver_weight > 0
                    or self.args.rec_loc_weight > 0
                    or getattr(self.args, "mpjpe_eval_enabled", True)
                ):
                    tar_pose = tar_pose_aa.reshape(bs*n, j*3)
                    rec_pose = rec_pose_aa.reshape(bs*n, j*3)
                    rec_pose = other_tools.inverse_selection_tensor(rec_pose, lower_joint_mask, rec_pose.shape[0])
                    tar_pose = other_tools.inverse_selection_tensor(tar_pose, lower_joint_mask, tar_pose.shape[0])
                    need_verts = self.args.rec_ver_weight > 0
                    rec_joints, rec_verts = self._smplx_forward(
                        pose_aa_full=rec_pose.reshape(bs, n, 55, 3),
                        transl=rec_xyz_trans.reshape(bs, n, 3),
                        betas=tar_beta.reshape(bs, n, 300),
                        expressions=tar_exps.reshape(bs, n, 100),
                        return_verts=need_verts,
                        no_grad=True,
                    )
                    tar_joints, tar_verts = self._smplx_forward(
                        pose_aa_full=tar_pose.reshape(bs, n, 55, 3),
                        transl=tar_trans.reshape(bs, n, 3),
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

                        vectices_loss = self.vectices_loss(pred_vertices, tar_vertices)
                        vectices_loss = vectices_loss.mean()
                        self.tracker.update_meter("ver", "val", vectices_loss.item()*self.args.rec_weight * self.args.rec_ver_weight)

                    if self.args.rec_loc_weight > 0:
                        pred_joints = rec_joints
                        # tar_joints already (bs, n, 55, 3) from _smplx_forward
                        joint_loc_loss = self.loc_loss(pred_joints, tar_joints) #* loss_timemask.view(1, -1, 1, 1)
                        self.tracker.update_meter("loc_rec", "train", joint_loc_loss.mean().item()*self.args.rec_weight * self.args.rec_loc_weight)
                        # g_loss_final += joint_loc_loss.mean()*self.args.rec_weight*self.args.rec_loc_weight

                        # foot contact loss
                        foot_idx = [7, 8, 10, 11]
                        model_contact = net_out[:, :, j*6+3:j*6+7]
                        # find static indices consistent with model's own predictions
                        static_idx = model_contact > 0.98  # N x S x 4
                        # print(model_contact,static_idx)
                        model_feet = pred_joints[:, :, foot_idx]  # foot positions (N, S, 4, 3)
                        model_foot_v = torch.zeros_like(model_feet)
                        model_foot_v[:, :-1] = (
                            model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
                        )  # (N, S-1, 4, 3)
                        model_foot_v[~static_idx] = 0
                        foot_loss = self.vel_loss(
                            model_foot_v, torch.zeros_like(model_foot_v)
                        ) #* loss_timemask.view(1, -1, 1, 1)
                        self.tracker.update_meter("foot", "val", foot_loss.mean().item()*self.args.rec_weight * self.args.rec_loc_weight*20)
                
                lower_loss_embedding = lower_q_res.penalty
                if epoch < 1:
                    embedding_weight = 0
                else:
                    embedding_weight = min(1.0, epoch/self.args.kl_warmup_epochs) * self.args.comloss_weight
                # g_loss_final += lower_loss_embedding * embedding_weight
                self.tracker.update_meter("com", "val", (lower_loss_embedding * embedding_weight).item()) # embedding_weight * 
                # break
                if self.args.debug and its == 20: break

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
            from .dataloaders.utils.visualize import (
                render_smplx_debug_video,
                stitch_videos_hstack,
            )

        from .utils.metrics import ReconMetrics

        if not os.path.exists(results_save_path):
            os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        # test_seq_list = self.test_data.selected_file

        recon_metrics = ReconMetrics(self.args)

        logger.info(f"Length of test set: {len(self.test_data)} samples in {len(self.test_loader)} batches")
        self.model.eval()
        with torch.no_grad():
            for its, dict_data in tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc=f"Testing at epoch {epoch}"):
                if max_batches is not None and its >= max_batches:
                    break
                tar_pose_lower = dict_data["motion_lower"]
                tar_trans = dict_data["transl"].to(self.local_rank)
                tar_contact = dict_data["contact"].to(self.local_rank)
                sample_names = dict_data["filechunk_id"]
                file_name = dict_data["file_id"]
                tar_beta = dict_data["beta"].to(self.local_rank)
                
                # subtract the first frame translation
                tar_trans[:, :, 0] = tar_trans[:, :, 0] - tar_trans[:, 0:1, 0]
                tar_trans[:, :, 2] = tar_trans[:, :, 2] - tar_trans[:, 0:1, 2]

                tar_trans_vel = other_tools.estimate_linear_velocity(
                    tar_trans, dt=1/self.args.motion_fps, 
                )

                lower_joint_mask = self.test_data.lower_mask_for_flattened

                tar_pose_lower = tar_pose_lower.to(self.local_rank)
                
                # breakpoint()

                bs, n, ldim = tar_pose_lower.shape
                lj = ldim //3
                tar_pose_lower = rc.axis_angle_to_matrix(tar_pose_lower.reshape(bs, n, lj, 3))
                tar_pose_lower = rc.matrix_to_rotation_6d(tar_pose_lower).reshape(bs, n, lj*6)

                j = lj
                tar_pose = tar_pose_lower
                tar_exps = torch.zeros((bs, n, 100)).to(self.local_rank)
                
                
                remain = n%self.args.pose_length
                tar_pose = tar_pose[:, :n-remain, :]
                tar_contact = tar_contact[:, :n-remain, :]
                tar_trans = tar_trans[:, :n-remain, :]
                tar_trans_vel = tar_trans_vel[:, :n-remain, :]
                # breakpoint()

                final_pos = torch.zeros_like(tar_trans[:, 0])
                # if self.model.causal:
                #     pass
                # else:
                # breakpoint()
                out_final = None
                out_trans = None
                framechunk_size = self.args.frame_chunk_size
                num_frames = tar_pose.shape[1]
                in_tar_pose = torch.cat((tar_pose, tar_trans_vel, tar_contact), dim=-1)
                # motion_chunks = []
                with self.model.streaming(batch_size=tar_pose.shape[0]):
                    for offset in range(0, num_frames, framechunk_size):
                        # tar_pose_new = tar_pose[:,i*(self.args.pose_length):i*(self.args.pose_length)+self.args.pose_length,:].clone()
                        # tar_cntct_new = tar_contact[:,i*(self.args.pose_length):i*(self.args.pose_length)+self.args.pose_length,:].clone()
                        # tar_trans_new = tar_trans[:,i*(self.args.pose_length):i*(self.args.pose_length)+self.args.pose_length,:].clone()
                        # breakpoint()
                        frame = in_tar_pose[:, offset:offset+framechunk_size, :]
                        # breakpoint()
                        # subtract the first frame translation
                        # frame[:, :, j*6+0] = frame[:, :, j*6+0] - frame[:, 0:1, j*6+0]
                        # frame[:, :, j*6+2] = frame[:, :, j*6+2] - frame[:, 0:1, j*6+2]

                        # input: pose, trans, contact -> output: pose only
                        # in_tar_pose = torch.cat((tar_pose_new, tar_trans_new, tar_cntct_new), dim=-1)
                        # net_out, quantize_res, loss_timemask = self.model(in_tar_pose)

                        codes = self.model.encode(frame)
                        assert codes.shape[-1] == 1
                        net_out = self.model.decode(codes)

                        n = net_out.shape[1]
                        assert n == framechunk_size
                        
                        rec_pose = net_out[:, :, :j*6]
                        rec_trans_vel = net_out[:, :, j*6:j*6+3]

                        rec_pose = rec_pose.reshape(bs, n, j, 6)
                        rec_pose = rc.rotation_6d_to_matrix(rec_pose)#
                        rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs, n, j*3)
                        rec_trans, nextfinalpos = other_tools.velocity2position_mixeddiff(
                            rec_trans_vel, 1/self.args.motion_fps, init_pos=final_pos
                        )
                        final_pos = nextfinalpos
                        
                        # rec_pose_out = rec_pose.cpu().numpy()
                        # rec_trans_out = rec_trans.cpu().numpy()

                        if offset != 0:
                            # if out_trans is not None:
                            #     # rec_trans[:, :, 0] = rec_trans[:, :, 0] + out_trans[:, -1, 0]
                            #     # rec_trans[:, :, 2] = rec_trans[:, :, 2] + out_trans[:, -1, 2]
                            #     pass
                            # else:
                            #     raise ValueError
                            out_final = torch.cat((out_final, rec_pose), dim=1)
                            out_trans = torch.cat((out_trans, rec_trans), dim=1)
                        else:
                            out_final = rec_pose
                            out_trans = rec_trans #rec_trans

                rec_pose = out_final
                rec_trans = out_trans
                assert num_frames == rec_pose.shape[1]
                assert num_frames == rec_trans.shape[1]
                rec_pose = rec_pose.reshape(bs*num_frames, j*3)
                rec_trans = rec_trans.reshape(bs*num_frames, 3)
                
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs, num_frames, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*tar_pose.shape[1], j*3)
                # tar_pose = tar_pose.cpu().numpy()

                tar_pose = self.test_data.inverse_selection_tensor(tar_pose, lower_joint_mask, tar_pose.shape[0])
                rec_pose = self.test_data.inverse_selection_tensor(rec_pose, lower_joint_mask, rec_pose.shape[0])

                # Reconstruction metrics (MPJPE / FGD / ...): mirrors the
                # upper-body trainer. Runs BEFORE the .cpu().numpy() chain
                # below because ReconMetrics.update expects torch tensors
                # on the inference device (it does .cpu()/.cuda() and
                # .expand() internally).
                assert num_frames == tar_trans.shape[1]
                tar_trans_metric = tar_trans.reshape(bs*num_frames, 3)
                rec_trans_metric = rec_trans  # already (bs*num_frames, 3)
                tar_exps_metric = tar_exps.reshape(bs*num_frames, -1)
                tar_beta_metric = tar_beta.reshape(bs*num_frames, -1)
                rec_exps_metric = tar_exps_metric  # lower codec doesn't predict expressions
                metric_dict = {
                    "rec_pose": rec_pose,
                    "rec_exps": rec_exps_metric,
                    "rec_trans": rec_trans_metric,
                    "tar_pose": tar_pose,
                    "tar_exps": tar_exps_metric,
                    "tar_beta": tar_beta_metric[0],
                    "tar_trans": tar_trans_metric,
                    "file_id": file_name[0],
                }
                recon_metrics.update(metric_dict)

                rec_pose = rec_pose.cpu().numpy()
                tar_pose = tar_pose.cpu().numpy()

                tar_trans = tar_trans_metric
                tar_trans = tar_trans.cpu().numpy()
                rec_trans = rec_trans.cpu().numpy()
                
                total_length += rec_pose.shape[0]

                # --- save --- #
                rec_pose = rec_pose.reshape(num_frames, len(self.test_data.smplx_joint_names), 3)
                tar_pose = tar_pose.reshape(num_frames, len(self.test_data.smplx_joint_names), 3)
                rec_trans = rec_trans.reshape(num_frames, 3)
                tar_trans = tar_trans.reshape(num_frames, 3)
                # breakpoint()

                sample_save_path = os.path.join(results_save_path, sample_names[0])
                if (save or visualize) and not os.path.exists(sample_save_path):
                    os.makedirs(sample_save_path)

                tar_exps = tar_exps.reshape(bs*num_frames, -1)
                tar_beta = tar_beta.reshape(bs*num_frames, -1)
                tar_exps = tar_exps.cpu().numpy()
                tar_beta = tar_beta.cpu().numpy()
                
                rec_out = {
                    "global_orient": rec_pose[:, 0].reshape(num_frames, -1),
                    "body_pose": rec_pose[:, 1:22].reshape(num_frames, -1),
                    "left_hand_pose": rec_pose[:, 25:40].reshape(num_frames, -1),
                    "right_hand_pose": rec_pose[:, 40:55].reshape(num_frames, -1),
                    "transl": rec_trans,
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
                        trans=rec_trans,
                        model='smplx',
                        gender='NEUTRAL_2020',
                        mocap_frame_rate=30,
                    )


                # --- render --- #
                if visualize:
                    logger.info(f"visualizing {sample_names[0]}")
                    # Re-center each trajectory at origin so the renderer's
                    # camera + floor framing land correctly relative to the
                    # feet (matches run_gestinference.py and the BEATX
                    # chunk-saving convention).
                    tar_trans_viz = tar_trans - tar_trans[0:1, :]
                    rec_trans_viz = rec_trans - rec_trans[0:1, :]

                    final_path = os.path.join(sample_save_path, "gt_pred_compared.mp4")
                    with tempfile.TemporaryDirectory(prefix="lower_sbs_") as tmpdir:
                        gt_path = os.path.join(tmpdir, "gt.mp4")
                        pred_path = os.path.join(tmpdir, "pred.mp4")
                        stitched = os.path.join(tmpdir, "stitched.mp4")
                        render_smplx_debug_video(
                            smplx_model=self.smplx_model,
                            poses=tar_pose.reshape(num_frames, -1),
                            transl=tar_trans_viz,
                            expressions=tar_exps,
                            betas=tar_beta,
                            output_path=gt_path,
                            fps=self.args.motion_fps,
                            mesh_color=(180, 54, 54, 255),
                        )
                        render_smplx_debug_video(
                            smplx_model=self.smplx_model,
                            poses=rec_pose.reshape(num_frames, -1),
                            transl=rec_trans_viz,
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
                    logger.info(f"saved to {final_path}")

                                                                                                
                #if its == 1:break
        end_time = time.time() - start_time
        recon_metrics.compute_metrics()
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.motion_fps)} s motion")

    def visualize_codebook(self, checkpoint_path, epoch):
        # breakpoint()
        
        codebook_emb = self.model.quantizer.vq.layers[0]._codebook.embedding.clone().detach()
        codebook_emb = codebook_emb.cpu().numpy()

        pca = PCA(n_components=3)
        codebook_3d = pca.fit_transform(codebook_emb)

        # Create a 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot
        ax.scatter(codebook_3d[:, 0], codebook_3d[:, 1], codebook_3d[:, 2], alpha=0.7)

        # Labels and title
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
        ax.set_title(f"3D PCA Projection of VQVAE Codebook: Epoch {epoch}")
        plt.grid(True)
        plt.savefig(checkpoint_path + f"/vq0_codebook_{epoch}.png")