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


class FaceExpCausalCodecTrainer(BaseCausalCodecTrainer):
    def __init__(self, args):
        assert "embody" not in args.dataset_ratio.lower(), (
            f"FaceExpCausalCodecTrainer cannot be used with embody3d data: "
            f"embody3d face/expression tracking is unreliable, so training the "
            f"face codec on it would corrupt the learned latent. "
            f"Got dataset_ratio={args.dataset_ratio!r}. "
            f"Use a BEATX-only ratio (e.g. 'full_beatx')."
        )
        super(FaceExpCausalCodecTrainer, self).__init__(args)

        self.tracker = other_tools.EpochTracker(
            ["rec_6d", "rec_rot", "rec_aa", "vel", "ver", "com", "kl", "acc",  "face", "face_vel", "face_acc", "face_laploss", "laploss6d"], 
            [False, False, False, False, False, False, False, False, False, False, False, False, False]
        )
        
        if not self.args.rot6d: #"rot6d" not in args.pose_rep:
            logger.error(f"this script is for rot6d, your pose rep. is {args.pose_rep}")

        self.rec_loss = get_loss_func("GeodesicLoss", reduction='none')
        # TODO: add specific loss for contact and foot contact
        self.rot6d_loss = torch.nn.MSELoss(reduction='none')
        self.aa_loss = torch.nn.MSELoss(reduction='none')
        self.vel_loss = torch.nn.MSELoss(reduction='none')
        self.vectices_loss = torch.nn.MSELoss(reduction='none')
        self.face_loss = torch.nn.L1Loss(reduction='none')

        # self.rot6d_loss = torch.nn.HuberLoss(reduction='none')
        # self.aa_loss = torch.nn.HuberLoss(reduction='none')
        # self.vel_loss = torch.nn.HuberLoss(reduction='none')
        # self.vectices_loss = torch.nn.HuberLoss(reduction='none')
        # self.loc_loss = torch.nn.HuberLoss(reduction='none')
        # self.loc_vel_loss = torch.nn.HuberLoss(reduction='none')
        # self.loc_acc_loss = torch.nn.HuberLoss(reduction='none')

        # self.trans_loss = torch.nn.MSELoss(reduction='none')
        self.laplacian_loss = get_loss_func("laplacian_loss", kernel_size=5, reduction='none')
        
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
            tar_pose_face = dict_data["motion_face"]
            tar_exps = dict_data["expressions"].to(self.local_rank)
            tar_beta = dict_data["beta"].to(self.local_rank)
            
            # tar_inp_tokenvecs = dict_data['token_vecs'].to(self.rank)

            dataset_name_per_item = dict_data['dataset_name']
            dataset_name_per_item = np.array(dataset_name_per_item)
            # get indices for dataset
            # seamless_indices = np.where(dataset_name_per_item != 'BEATX')[0]
            embody3d_indices = np.where(dataset_name_per_item != 'BEATX')[0]
            beatx_indices = np.where(dataset_name_per_item == 'BEATX')[0]

            assert len(embody3d_indices) == 0
            # breakpoint()
            
            # if self.rank == 0:
                # logger.info(f"@ rank {self.rank}, tar_pose_upper shape: {tar_pose_upper.shape}")
            # breakpoint() # check masks 
            face_joint_mask = self.train_data.face_mask_for_flattened 
            
            tar_pose_face = tar_pose_face.to(self.local_rank)
            
            # breakpoint()
            bs, n, fdim = tar_pose_face.shape
            j = fdim //3
            tar_pose_face = rc.axis_angle_to_matrix(tar_pose_face.reshape(bs, n, j, 3))
            tar_pose_face = rc.matrix_to_rotation_6d(tar_pose_face).reshape(bs, n, j*6)
            tar_pose = tar_pose_face.clone()

            t_data = time.time() - t_start
            
            self.opt.zero_grad()
            g_loss_final = 0
            # breakpoint()

            # input: facial 6d + expression # 106
            in_tar_pose_face = torch.cat((tar_pose_face, tar_exps), dim=-1)

            
            
            net_out, face_q_res, z_encoder, loss_timemask = self.model(in_tar_pose_face)
            
            
            rec_pose = net_out[:, :, :j*6]
            rec_exps = net_out[:, :, j*6:]
            # rec_pose = tar_pose.clone()
            # loss_timemask = torch.ones(n).to(self.local_rank)
            # breakpoint()

            # 6d loss
            rec_pose = rec_pose.reshape(bs, n, j, 6)
            tar_pose = tar_pose.reshape(bs, n, j, 6)
            loss_6d = self.rot6d_loss(rec_pose, tar_pose) * loss_timemask.view(1, -1, 1, 1)
            
            loss_6d = loss_6d.mean() * self.args.rec_weight * self.args.rec_6d_weight
            self.tracker.update_meter("rec_6d", "train", loss_6d.item())
            g_loss_final += loss_6d

            if rec_pose.shape[1] <=6: 
                laploss_6d = torch.zeros_like(loss_6d)
            else:
                laploss_6d = self.laplacian_loss(rec_pose.reshape(bs, n, j*6), tar_pose.reshape(bs, n, j*6))
                
            laploss_6d = laploss_6d.mean() * self.args.rec_weight * self.args.lap_weight
            self.tracker.update_meter("laploss6d", "train", laploss_6d.item())
            g_loss_final += laploss_6d
            

            
            # rotation matrix loss
            rec_pose = rc.rotation_6d_to_matrix(rec_pose)  * loss_timemask.view(1, -1, 1, 1, 1)
            tar_pose = rc.rotation_6d_to_matrix(tar_pose)  * loss_timemask.view(1, -1, 1, 1, 1)
            loss_rec_rot = self.rec_loss(rec_pose, tar_pose) 
            loss_rec_rot = loss_rec_rot.mean() * self.args.rec_weight * self.args.rec_pos_weight
            self.tracker.update_meter("rec_rot", "train", loss_rec_rot.item())
            g_loss_final += loss_rec_rot

            # axis angle loss
            rec_pose_aa = rc.matrix_to_axis_angle(rec_pose)
            tar_pose_aa = rc.matrix_to_axis_angle(tar_pose)
            loss_rec_aa = self.aa_loss(rec_pose_aa, tar_pose_aa) * loss_timemask.view(1, -1, 1, 1)
            loss_rec_aa = loss_rec_aa.mean() * self.args.rec_weight * self.args.rec_aa_weight
            self.tracker.update_meter("rec_aa", "train", loss_rec_aa.item())
            g_loss_final += loss_rec_aa

            
            
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

            # facial expression loss 
            # breakpoint()
            loss_face = self.face_loss(rec_exps, tar_exps) * self.args.rec_weight * self.args.rec_face_weight
            loss_face = loss_face * loss_timemask.view(1, -1, 1)
            loss_face = loss_face.mean()
            self.tracker.update_meter("face", "train", loss_face.item())
            g_loss_final += loss_face
            # facial expression velocity and acceleration loss
            face_velocity_loss = self.vel_loss(rec_exps[:, 1:] - rec_exps[:, :-1], tar_exps[:, 1:] - tar_exps[:, :-1]) * loss_timemask[1:].view(1, -1, 1)
            face_velocity_loss = face_velocity_loss.mean() * self.args.rec_weight * 10
            if rec_exps.shape[1] <=6:
                face_acceleration_loss = torch.zeros_like(face_velocity_loss)
            else:
                face_acceleration_loss = self.vel_loss(rec_exps[:, 2:] + rec_exps[:, :-2] - 2 * rec_exps[:, 1:-1], tar_exps[:, 2:] + tar_exps[:, :-2] - 2 * tar_exps[:, 1:-1]) * loss_timemask[2:].view(1, -1, 1)
            face_acceleration_loss = face_acceleration_loss.mean() * self.args.rec_weight * 10
            self.tracker.update_meter("face_vel", "train", face_velocity_loss.item())
            self.tracker.update_meter("face_acc", "train", face_acceleration_loss.item())
            g_loss_final += face_velocity_loss
            g_loss_final += face_acceleration_loss
            if rec_exps.shape[1] <=6:
                face_lap_loss = torch.zeros_like(face_velocity_loss)
            else:
                face_lap_loss = self.laplacian_loss(rec_exps, tar_exps)
            face_lap_loss = face_lap_loss.mean() * self.args.rec_weight * 10
            g_loss_final += face_lap_loss
            self.tracker.update_meter("face_laploss", "train", face_lap_loss.item())


            # vertices loss and joint location loss
            if self.args.rec_ver_weight > 0:
                tar_pose = tar_pose_aa.reshape(bs*n, j*3)
                rec_pose = rec_pose_aa.reshape(bs*n, j*3)
                rec_pose = self.train_data.inverse_selection_tensor(rec_pose, face_joint_mask, rec_pose.shape[0])
                tar_pose = self.train_data.inverse_selection_tensor(tar_pose, face_joint_mask, tar_pose.shape[0])
                zero_transl = torch.zeros((bs, n, 3), device=rec_pose.device, dtype=rec_pose.dtype)
                rec_joints, rec_verts = self._smplx_forward(
                    pose_aa_full=rec_pose.reshape(bs, n, 55, 3),
                    transl=zero_transl,
                    betas=tar_beta.reshape(bs, n, 300),
                    expressions=rec_exps.reshape(bs, n, 100),
                    return_verts=True,
                )
                tar_joints, tar_verts = self._smplx_forward(
                    pose_aa_full=tar_pose.reshape(bs, n, 55, 3),
                    transl=zero_transl,
                    betas=tar_beta.reshape(bs, n, 300),
                    expressions=tar_exps.reshape(bs, n, 100),
                    return_verts=True,
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

                

            # ---------------------- vqvae loss -------------------------- #
            # breakpoint()
            face_loss_embedding = face_q_res.penalty
            if epoch < 1:
                embedding_weight = 0
            else:
                embedding_weight = min(1.0, epoch/self.args.kl_warmup_epochs) * self.args.comloss_weight
            g_loss_final += face_loss_embedding * embedding_weight
            self.tracker.update_meter("com", "train", (face_loss_embedding * embedding_weight).item()) # embedding_weight * 
            
            # ------------------------------------------------------------ #

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
                    metric_print = f"[GPU{self.global_rank}:{self.local_rank}]" + str({k: round(v.item(), 3) for k, v in face_q_res.metrics.items()})
                    logger.info(metric_print)
            
            if self.global_rank == 0 and its % 1000 == 0 and its > 0:
                logger.info(f"[GPU {self.global_rank}] Saving checkpoints at epoch {epoch} iter {its}:")
                other_tools.save_checkpoints(os.path.join(self.checkpoint_path, f"iter_{epoch}_{its}"), self.model, opt=None, epoch=None, lrs=None, save_dtype=self.args.param_dtype)
                # trainer.test(epoch)
                self.args.test_ckpt = os.path.join(self.checkpoint_path, f"iter_{epoch}_{its}.safetensors")
                other_tools.update_args_file(self.args, rank=self.global_rank)
                # breakpoint() # add its>0 condition

            if self.args.debug:
                if its == 10: break
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
                tar_pose_face = dict_data["motion_face"]
                tar_exps = dict_data["expressions"].to(self.local_rank)
                tar_beta = dict_data["beta"].to(self.local_rank)
                
                # tar_inp_tokenvecs = dict_data['token_vecs'].to(self.rank)

                dataset_name_per_item = dict_data['dataset_name']
                dataset_name_per_item = np.array(dataset_name_per_item)
                # get indices for dataset
                # seamless_indices = np.where(dataset_name_per_item != 'BEATX')[0]
                embody3d_indices = np.where(dataset_name_per_item != 'BEATX')[0]
                beatx_indices = np.where(dataset_name_per_item == 'BEATX')[0]

                assert len(embody3d_indices) == 0
                # breakpoint()
                
                # if self.rank == 0:
                    # logger.info(f"@ rank {self.rank}, tar_pose_upper shape: {tar_pose_upper.shape}")
                # breakpoint() # check masks 
                face_joint_mask = self.train_data.face_mask_for_flattened 
                
                tar_pose_face = tar_pose_face.to(self.local_rank)
                
                # breakpoint()
                bs, n, fdim = tar_pose_face.shape
                j = fdim //3
                tar_pose_face = rc.axis_angle_to_matrix(tar_pose_face.reshape(bs, n, j, 3))
                tar_pose_face = rc.matrix_to_rotation_6d(tar_pose_face).reshape(bs, n, j*6)
                tar_pose = tar_pose_face.clone()

                # input: facial 6d + expression # 106
                in_tar_pose_face = torch.cat((tar_pose_face, tar_exps), dim=-1)

                
                
                net_out, face_q_res, z_encoder, loss_timemask = self.model(in_tar_pose_face)
                
                
                rec_pose = net_out[:, :, :j*6]
                rec_exps = net_out[:, :, j*6:]
                # rec_pose = tar_pose.clone()
                # loss_timemask = torch.ones(n).to(self.local_rank)
                # breakpoint()

                # 6d loss
                rec_pose = rec_pose.reshape(bs, n, j, 6)
                tar_pose = tar_pose.reshape(bs, n, j, 6)
                loss_6d = self.rot6d_loss(rec_pose, tar_pose) * loss_timemask.view(1, -1, 1, 1)
                
                loss_6d = loss_6d.mean() * self.args.rec_weight * self.args.rec_6d_weight
                self.tracker.update_meter("rec_6d", "val", loss_6d.item())
                # g_loss_final += loss_6d

                if rec_pose.shape[1] <=6: 
                    laploss_6d = torch.zeros_like(loss_6d)
                else:
                    laploss_6d = self.laplacian_loss(rec_pose.reshape(bs, n, j*6), tar_pose.reshape(bs, n, j*6))
                    
                laploss_6d = laploss_6d.mean() * self.args.rec_weight * self.args.lap_weight
                self.tracker.update_meter("laploss6d", "val", laploss_6d.item())
                # g_loss_final += laploss_6d
                

                
                # rotation matrix loss
                rec_pose = rc.rotation_6d_to_matrix(rec_pose)  * loss_timemask.view(1, -1, 1, 1, 1)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose)  * loss_timemask.view(1, -1, 1, 1, 1)
                loss_rec_rot = self.rec_loss(rec_pose, tar_pose) 
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

                
                
                # velocity loss and acceleration loss
                velocity_loss =  self.vel_loss(rec_pose[:, 1:] - rec_pose[:, :-1], tar_pose[:, 1:] - tar_pose[:, :-1]) * loss_timemask[1:].view(1, -1, 1, 1, 1)
                velocity_loss = velocity_loss.mean() * self.args.rec_weight * 30
                if rec_pose.shape[1] <=6:
                    acceleration_loss = torch.zeros_like(velocity_loss)
                else:
                    acceleration_loss =  self.vel_loss(rec_pose[:, 2:] + rec_pose[:, :-2] - 2 * rec_pose[:, 1:-1], tar_pose[:, 2:] + tar_pose[:, :-2] - 2 * tar_pose[:, 1:-1]) * loss_timemask[2:].view(1, -1, 1, 1, 1)
                
                acceleration_loss = acceleration_loss.mean() * self.args.rec_weight * 30

                
                self.tracker.update_meter("vel", "val", velocity_loss.item())
                self.tracker.update_meter("acc", "val", acceleration_loss.item())
                # g_loss_final += velocity_loss
                # g_loss_final += acceleration_loss

                # facial expression loss 
                # breakpoint()
                loss_face = self.face_loss(rec_exps, tar_exps) * self.args.rec_weight * self.args.rec_face_weight
                loss_face = loss_face * loss_timemask.view(1, -1, 1)
                loss_face = loss_face.mean()
                self.tracker.update_meter("face", "val", loss_face.item())
                # g_loss_final += loss_face
                # facial expression velocity and acceleration loss
                face_velocity_loss = self.vel_loss(rec_exps[:, 1:] - rec_exps[:, :-1], tar_exps[:, 1:] - tar_exps[:, :-1]) * loss_timemask[1:].view(1, -1, 1)
                face_velocity_loss = face_velocity_loss.mean() * self.args.rec_weight * 10
                if rec_exps.shape[1] <=6:
                    face_acceleration_loss = torch.zeros_like(face_velocity_loss)
                else:
                    face_acceleration_loss = self.vel_loss(rec_exps[:, 2:] + rec_exps[:, :-2] - 2 * rec_exps[:, 1:-1], tar_exps[:, 2:] + tar_exps[:, :-2] - 2 * tar_exps[:, 1:-1]) * loss_timemask[2:].view(1, -1, 1)
                face_acceleration_loss = face_acceleration_loss.mean() * self.args.rec_weight * 10
                self.tracker.update_meter("face_vel", "val", face_velocity_loss.item())
                self.tracker.update_meter("face_acc", "val", face_acceleration_loss.item())
                # g_loss_final += face_velocity_loss
                # g_loss_final += face_acceleration_loss
                if rec_exps.shape[1] <=6:
                    face_lap_loss = torch.zeros_like(face_velocity_loss)
                else:
                    face_lap_loss = self.laplacian_loss(rec_exps, tar_exps)
                face_lap_loss = face_lap_loss.mean() * self.args.rec_weight * 10
                self.tracker.update_meter("face_laploss", "val", face_lap_loss.item())


                # vertices loss and MPJPE
                if self.args.rec_ver_weight > 0 or getattr(self.args, "mpjpe_eval_enabled", True):
                    tar_pose = tar_pose_aa.reshape(bs*n, j*3)
                    rec_pose = rec_pose_aa.reshape(bs*n, j*3)
                    rec_pose = self.val_data.inverse_selection_tensor(rec_pose, face_joint_mask, rec_pose.shape[0])
                    tar_pose = self.val_data.inverse_selection_tensor(tar_pose, face_joint_mask, tar_pose.shape[0])
                    need_verts = self.args.rec_ver_weight > 0
                    zero_transl = torch.zeros((bs, n, 3), device=rec_pose.device, dtype=rec_pose.dtype)
                    rec_joints, rec_verts = self._smplx_forward(
                        pose_aa_full=rec_pose.reshape(bs, n, 55, 3),
                        transl=zero_transl,
                        betas=tar_beta.reshape(bs, n, 300),
                        expressions=rec_exps.reshape(bs, n, 100),
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

                    # vertices loss
                    if self.args.rec_ver_weight > 0:
                        pred_vertices = rec_verts
                        tar_vertices = tar_verts

                        vectices_loss = self.vectices_loss(pred_vertices, tar_vertices) * loss_timemask.view(1, -1, 1, 1)
                        vectices_loss = vectices_loss.mean()
                        self.tracker.update_meter("ver", "val", vectices_loss.item()*self.args.rec_weight * self.args.rec_ver_weight)
                        # g_loss_final += vectices_loss.mean()*self.args.rec_weight*self.args.rec_ver_weight

                        
                    

                # ---------------------- vqvae loss -------------------------- #
                # breakpoint()
                face_loss_embedding = face_q_res.penalty
                if epoch < 1:
                    embedding_weight = 0
                else:
                    embedding_weight = min(1.0, epoch/self.args.kl_warmup_epochs) * self.args.comloss_weight
                # g_loss_final += face_loss_embedding * embedding_weight
                self.tracker.update_meter("com", "val", (face_loss_embedding * embedding_weight).item()) # embedding_weight * 
                
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
            # `only_face=True` selects the same tight head framing the old
            # visualize_smpl(only_face=True) used, so the visual is unchanged
            # from before -- just routed through the new renderer.

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
                tar_pose_face = dict_data["motion_face"]
                tar_exps = dict_data["expressions"].to(self.local_rank)
                tar_beta = dict_data["beta"].to(self.local_rank)
                # tar_trans = dict_data["trans"].to(self.local_rank)
                tar_trans = torch.zeros_like(dict_data["transl"]).to(self.local_rank)
                

                tar_pose_face = tar_pose_face.to(self.local_rank)
                sample_names = dict_data["filechunk_id"]
                file_name = dict_data["file_id"]
                # if not sample_names[0].startswith("c--"):
                #     continue
                
                face_joint_mask = self.test_data.face_mask_for_flattened 

                bs, n, fdim = tar_pose_face.shape
                j = fdim //3
                tar_pose_face = rc.axis_angle_to_matrix(tar_pose_face.reshape(bs, n, j, 3))
                tar_pose_face = rc.matrix_to_rotation_6d(tar_pose_face).reshape(bs, n, j*6)
                tar_pose = tar_pose_face.clone()

                remain = n%self.args.frame_chunk_size
                tar_pose = tar_pose[:, :n-remain, :]
                tar_pose_face = tar_pose_face[:, :n-remain, :]
                tar_trans = tar_trans[:, :n-remain, :]
                

                # if self.model.causal:
                #     pass
                # else:
                # breakpoint()
                out_final = None
                out_exps = None
                framechunk_size = self.args.frame_chunk_size
                num_frames = tar_pose.shape[1]
                in_tar_pose = torch.cat((tar_pose_face, tar_exps), dim=-1)
                # motion_chunks = []
                with self.model.streaming(batch_size=tar_pose.shape[0]):
                    for offset in range(0, num_frames, framechunk_size):
                        # tar_pose_uppernew = tar_pose_upper[:,i*(self.args.pose_length):i*(self.args.pose_length)+self.args.pose_length,:].clone()
                        # tar_pose_handsnew = tar_pose_hands[:,i*(self.args.pose_length):i*(self.args.pose_length)+self.args.pose_length,:].clone()
                        frame = in_tar_pose[:, offset:offset+framechunk_size, :]

                        codes = self.model.encode(frame)
                        assert codes.shape[-1] == 1
                        net_out = self.model.decode(codes)

                        rec_pose = net_out[:, :, :j*6]
                        rec_exps = net_out[:, :, j*6:]
                        # self.model.eval()

                        n = rec_pose.shape[1]
                        assert n == framechunk_size

                        rec_pose = rec_pose.reshape(bs, n, j, 6)
                        rec_pose = rc.rotation_6d_to_matrix(rec_pose)#
                        rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs, n, j*3)
                        
                        # rec_trans = rec_trans.cpu().numpy()

                        if offset != 0:
                            out_final = torch.cat([out_final, rec_pose], dim=1)
                            out_exps = torch.cat([out_exps, rec_exps], dim=1)
                        else:
                            out_final = rec_pose
                            out_exps = rec_exps
                

                # breakpoint()

                rec_pose = out_final
                rec_exps = out_exps
                assert num_frames == rec_pose.shape[1]
                assert num_frames == rec_exps.shape[1]
                assert rec_pose.shape[0] == tar_pose.shape[0] == bs == 1
                # assert num_frames == rec_trans.shape[1]
                # breakpoint() # check shapes going forward
                rec_pose = rec_pose.reshape(bs*num_frames, j*3)
                rec_exps = rec_exps.reshape(bs*num_frames, -1)
                tar_exps = tar_exps.reshape(bs*num_frames, -1)
                
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs, num_frames, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*tar_pose.shape[1], j*3)
                tar_pose = self.test_data.inverse_selection_tensor(tar_pose, face_joint_mask, tar_pose.shape[0])
                rec_pose = self.test_data.inverse_selection_tensor(rec_pose, face_joint_mask, rec_pose.shape[0])

                # Reconstruction metrics (FGD / MPJPE / Facial L2 / Facial
                # L-Vel). Runs BEFORE the .cpu().numpy() chain below because
                # ReconMetrics.update expects torch tensors on the inference
                # device (it does .cpu()/.cuda()/.expand() internally).
                assert num_frames == tar_trans.shape[1]
                tar_trans_metric = tar_trans.reshape(bs*num_frames, 3)
                rec_trans_metric = tar_trans_metric  # face codec doesn't predict translation
                tar_beta_metric = tar_beta.reshape(bs*num_frames, -1)
                metric_dict = {
                    "rec_pose": rec_pose,
                    "rec_exps": rec_exps,
                    "rec_trans": rec_trans_metric,
                    "tar_pose": tar_pose,
                    "tar_exps": tar_exps,
                    "tar_beta": tar_beta_metric[0],
                    "tar_trans": tar_trans_metric,
                    "file_id": file_name[0],
                }
                recon_metrics.update(metric_dict)

                rec_pose = rec_pose.cpu().numpy()
                tar_pose = tar_pose.cpu().numpy()

                tar_trans = tar_trans_metric
                tar_trans = tar_trans.cpu().numpy()

                rec_exps = rec_exps.cpu().numpy()
                tar_exps = tar_exps.cpu().numpy()
                
                
                total_length += rec_pose.shape[0]
                # --- save --- #
                # breakpoint() 
                rec_pose = rec_pose.reshape(num_frames, len(self.test_data.smplx_joint_names), 3)
                tar_pose = tar_pose.reshape(num_frames, len(self.test_data.smplx_joint_names), 3)
                
                sample_save_path = os.path.join(results_save_path, sample_names[0])
                if (save or visualize) and not os.path.exists(sample_save_path):
                    os.makedirs(sample_save_path)

                tar_beta = tar_beta.reshape(bs*num_frames, -1)
                tar_beta = tar_beta.cpu().numpy()
                # breakpoint()
                rec_out = {
                    "global_orient": rec_pose[:, 0].reshape(num_frames, -1),
                    "body_pose": rec_pose[:, 1:22].reshape(num_frames, -1),
                    "left_hand_pose": rec_pose[:, 25:40].reshape(num_frames, -1),
                    "right_hand_pose": rec_pose[:, 40:55].reshape(num_frames, -1),
                    "transl": tar_trans,
                    "betas": tar_beta,
                    "expression": rec_exps,
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
                    with tempfile.TemporaryDirectory(prefix="face_sbs_") as tmpdir:
                        gt_path = os.path.join(tmpdir, "gt.mp4")
                        pred_path = os.path.join(tmpdir, "pred.mp4")
                        stitched = os.path.join(tmpdir, "stitched.mp4")
                        # GT pose + GT expressions on left, GT pose +
                        # predicted expressions on right -- so any visual
                        # difference between the two panels comes from the
                        # codec's reconstruction of the expression coeffs.
                        render_smplx_debug_video(
                            smplx_model=self.smplx_model,
                            poses=tar_pose.reshape(num_frames, -1),
                            transl=tar_trans,
                            expressions=tar_exps,
                            betas=tar_beta,
                            output_path=gt_path,
                            fps=self.args.motion_fps,
                            mesh_color=(180, 54, 54, 255),
                            only_face=True,
                        )
                        render_smplx_debug_video(
                            smplx_model=self.smplx_model,
                            poses=tar_pose.reshape(num_frames, -1),
                            transl=tar_trans,
                            expressions=rec_exps,
                            betas=tar_beta,
                            output_path=pred_path,
                            fps=self.args.motion_fps,
                            mesh_color=(36, 73, 156, 255),
                            only_face=True,
                        )
                        stitch_videos_hstack([gt_path, pred_path], stitched)
                        if not os.path.exists(stitched):
                            raise RuntimeError(f"hstack failed; no output at {stitched}")
                        shutil.move(stitched, final_path)
                    logger.info(f"output saved to {final_path}")
                    # breakpoint()
                                                               
                # if its == 1:break
        end_time = time.time() - start_time
        recon_metrics.compute_metrics()
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.motion_fps)} s motion")
