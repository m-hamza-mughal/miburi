import smplx
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# import logging
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from .utils.optim_factory import create_optimizer
from .utils.scheduler_factory import create_scheduler
from .utils.tools import save_checkpoints, load_checkpoints
from .utils.distributed import get_rank, get_world_size
from .utils.wrapped_model import get_fsdp_model
from . import dataloaders as dataset
from miburi.models import loaders, GestureMimiCodec


class BaseCausalCodecTrainer(object):
    def __init__(self, args):
        self.args = args
        if args.ddp: dist.barrier()
        
        self.local_rank = int(os.environ.get("LOCAL_RANK", "N/A")) if args.ddp else 0
        self.global_rank = get_rank() if args.ddp else 0

        logger.info(f"[GPU{self.global_rank}:{self.local_rank}] Init trainer...")

        self.checkpoint_path = args.out_path + args.name + args.notes + "/"
        if not args.is_train:
            # breakpoint()
            self.checkpoint_path = os.path.dirname(args.test_ckpt) + "/"
        
        if self.global_rank==0 and self.args.is_train: #  and args.code_save_split is None:
            self.writer = SummaryWriter(log_dir=self.checkpoint_path)
        
        # print(f"rank {self.rank} init BaseCausalVAETrainer")

         # breakpoint()
        # with torch.device("meta"):
        # self.model, self.teacher, self.vq0_codec = self.get_model(args)
        self.model = self.get_model(args)

        if args.is_train:
            if args.is_continue: # and self.rank == 0:
                torch.cuda.empty_cache() 
                continue_ckpt = args.continue_ckpt
                logger.info(f"[GPU{self.global_rank}:{self.local_rank}] Continue training from {continue_ckpt}")
                if not os.path.exists(continue_ckpt):
                    logger.error(f"[GPU{self.global_rank}:{self.local_rank}] Continue checkpoint {continue_ckpt} does not exist!")
                    raise FileNotFoundError(f"[GPU{self.global_rank}:{self.local_rank}] Continue checkpoint {continue_ckpt} does not exist!")
                # if self.rank == 0:
                load_checkpoints(self.model, continue_ckpt, rank=self.local_rank, after_distributed=False)

                self.model = self.model.to(self.local_rank)

                # for param in self.model.parameters():
                #     torch.distributed.broadcast(param.data, src=0)
            else:
                self.model.init_weights()
                continue_ckpt = None
                self.model = self.model.to(self.local_rank)
        else:
            # breakpoint()
            continue_ckpt = args.test_ckpt
            if not os.path.exists(continue_ckpt):
                logger.error(f"Test checkpoint {continue_ckpt} does not exist!")
                raise FileNotFoundError(f"Test checkpoint {continue_ckpt} does not exist!")
            load_checkpoints(self.model, continue_ckpt, rank=self.local_rank, after_distributed=False)
            self.model = self.model.to(self.local_rank)

        

        # change dtype of model if needed

        # self.model = get_fsdp_model(args, self.model, continue_ckpt)
        if args.ddp:
            assert get_world_size() > 1, "DDP requires more than one process"
            process_group = torch.distributed.new_group()
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model, process_group)
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank,
                                broadcast_buffers=False, find_unused_parameters=False)
        
        

        for p in self.model.parameters():
            if torch.isnan(p).any():
                raise ValueError("nan in model parameters")
        # for p in self.model.parameters():
        #     print(f"param {p.shape} {p.dtype} {p.device}")
        #     break

        # param_dtype = getattr(torch, args.param_dtype)
        # self.model = self.model.to(param_dtype)
        # self.teacher = self.teacher.to(param_dtype)
        # self.vq0_codec = self.vq0_codec.to(param_dtype)
        # if args.ddp:
        #     self.model = self.model.to(self.local_rank)
        #     process_group = torch.distributed.new_group()
        #     self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model, process_group)
        #     self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank,
        #                      broadcast_buffers=False, find_unused_parameters=False)
        # else:
        #     self.model = self.model.cuda()
        # # self.model = torch.nn.DataParallel(getattr(model_module, args.g_name)(args), args.gpus).cuda()
        

        self.tracker = None
        self.dataset_file = dataset
        dataset_class = getattr(self.dataset_file, args.dataset.upper() + "Dataset")

        if args.is_train:
            self.train_data = dataset_class(
                args, 
                "train", 
                only_motion=True, 
                dataset_ratio=args.dataset_ratio,
                debug=args.debug,
                varying_frame_length=args.varying_frame_length 
            )

            self.val_data = dataset_class(
                args, 
                "val", 
                only_motion=True,
                dataset_ratio=args.dataset_ratio, 
                debug=args.debug,
                varying_frame_length=args.varying_frame_length
            )

            if args.ddp:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.train_data,
                    shuffle=True,
                    drop_last=True
                )
                val_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.val_data,
                    shuffle=True,
                    drop_last=False
                )
            else:
                train_sampler = None
                val_sampler = None
            
            
            
            self.train_loader = torch.utils.data.DataLoader(
                self.train_data,
                batch_size=args.batch_size,
                shuffle= False if args.ddp else True,
                num_workers=args.loader_workers,
                drop_last=True,
                sampler=train_sampler,
                collate_fn=self.train_data.collate_fn,
                pin_memory=True,  # Recommended for GPU training
                persistent_workers=True if args.loader_workers > 0 else False, 
            )
            
            self.train_length = len(self.train_loader)
            logger.info(f"[GPU{self.global_rank}:{self.local_rank}] train length: {self.train_length}")
            logger.info(f"[GPU{self.global_rank}:{self.local_rank}] Init train dataloader success")
        
            
            self.val_loader = torch.utils.data.DataLoader(
                self.val_data,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.loader_workers,
                drop_last=False,
                sampler=val_sampler,
                collate_fn=self.val_data.collate_fn,
                pin_memory=True,  # Recommended for GPU training
                persistent_workers=True if args.loader_workers > 0 else False, 
            )
            logger.info(f"[GPU{self.global_rank}:{self.local_rank}] Init val dataloader success")

        if self.local_rank == 0:
            self.test_data = dataset_class(
                args, 
                "test", 
                only_motion=True, 
                dataset_ratio=args.dataset_ratio,
                debug=args.debug,
                varying_frame_length=False
            )
            self.test_loader = torch.utils.data.DataLoader(
                self.test_data,
                batch_size=1, # TODO: chnage
                shuffle=True,
                num_workers=args.loader_workers,
                drop_last=False, # TODO: change
                sampler=None,
                collate_fn=self.test_data.collate_fn,
            )
            logger.info(f"[GPU{self.global_rank}:{self.local_rank}] Init test dataloader success")

        if self.global_rank == 0:
            logger.info(self.model)
        model_device = next(self.model.parameters()).device
        logger.info(f"[GPU{self.global_rank}:{self.local_rank}] init model success {model_device}")          
        
        if args.is_train:
            self.opt = create_optimizer(args, self.model)
            # self.opt_latproj = create_optimizer(args, self.latent_proj)
            self.opt_s = create_scheduler(args, self.opt, total_steps=self.train_length*args.epochs)
            # self.opts_latproj = create_scheduler(args, self.opt_latproj)

        # if args.is_train:
        #     self.joints = self.train_data.joints
        # else:
        #     self.joints = self.test_data.joints
        self.smplx_model = smplx.create(
                self.args.deps_path + 'smplx_2020/', 
                model_type="smplx",
                gender="NEUTRAL_2020",
                flat_hand_mean=True,
                num_betas=300,
                num_expression_coeffs=100,
                use_pca=False,
            ).to(self.local_rank).eval()
        for p in self.smplx_model.parameters():
            p.requires_grad = False

        torch.cuda.empty_cache()
        # inverse selection functions here or in utils? #TODO

    # ---------------------------------------------------------------- SMPLX
    # Batched / chunked SMPL-X forward shared by the rvq trainers' vertex &
    # joint-location losses. Ported from gesture-style-transfer's vqvae trainer
    # so all rvq trainers go through one code path. Two tunable axes:
    #   smplx_chunk_per_batch_multiplier - chunk size scales with training
    #                                     batch (chunk_frames = M * batch_size).
    #   smplx_skip_expressions_for_loss - drop face expressions when only body
    #                                     joints/verts are needed (huge speed-up
    #                                     for upper/lower codecs).
    # When ``return_verts=False`` a fast path computes joints via batched
    # rigid-transform skinning (no vertex blend-shape pass). A parity check
    # vs the full forward fires for the first ``smplx_fast_parity_check_steps``
    # grad-enabled calls and asserts max-abs joint difference <= tol.

    def _expand_framewise(self, x, batch_size: int, nframes: int, dim: int) -> torch.Tensor:
        device = next(self.smplx_model.parameters()).device
        if x is None:
            return torch.zeros((batch_size * nframes, dim), device=device, dtype=torch.float32)
        x = x.to(device).float()
        if x.ndim == 1:
            return x.view(1, -1).repeat(batch_size * nframes, 1)
        if x.ndim == 2:
            if x.shape[0] == batch_size:
                return x[:, None, :].repeat(1, nframes, 1).reshape(batch_size * nframes, -1)
            if x.shape[0] == 1:
                return x.repeat(batch_size * nframes, 1)
        if x.ndim == 3 and x.shape[0] == batch_size and x.shape[1] == nframes:
            return x.reshape(batch_size * nframes, -1)
        return x.reshape(-1, x.shape[-1])[:1].repeat(batch_size * nframes, 1)

    def _smplx_forward(
        self,
        pose_aa_full: torch.Tensor,
        transl: torch.Tensor,
        betas=None,
        expressions=None,
        return_verts: bool = True,
        no_grad: bool = False,
    ):
        """Chunked SMPL-X forward.

        Args:
            pose_aa_full: (B, T, 55, 3) full-body axis-angle pose.
            transl:      (B, T, 3) root translation.
            betas:       broadcastable shape coeffs (None, (D,), (B,D), or (B,T,D)).
            expressions: broadcastable expression coeffs (same broadcast rules).
            return_verts: if True, also return vertices; if False, take the fast
                joint-only path (skips vertex skinning).
            no_grad: run under ``torch.no_grad()`` (eval / metrics path).

        Returns:
            joints: (B, T, 55, 3)
            verts:  (B, T, V, 3) if ``return_verts`` else ``None``.
        """
        from smplx.lbs import (
            batch_rigid_transform,
            batch_rodrigues,
            blend_shapes,
            vertices2joints,
        )

        smplx_model = self.smplx_model
        b, t, _, _ = pose_aa_full.shape
        n = b * t
        pose = pose_aa_full.reshape(n, 55, 3)
        tr = transl.reshape(n, 3)
        num_betas = int(getattr(self.args, "smplx_num_betas", 300))
        num_expr = int(getattr(self.args, "smplx_num_expression_coeffs", 100))
        betas_f = self._expand_framewise(betas, batch_size=b, nframes=t, dim=num_betas)

        skip_expr = bool(getattr(self.args, "smplx_skip_expressions_for_loss", False))
        if skip_expr:
            expressions = None
            if not hasattr(self, "_smplx_skip_expr_logged"):
                logger.info("SMPL-X loss path: expressions disabled (smplx_skip_expressions_for_loss=True).")
                self._smplx_skip_expr_logged = True
        expr_f = self._expand_framewise(expressions, batch_size=b, nframes=t, dim=num_expr)

        # SMPL-X forward chunk size derived from the training batch.
        # The chunk iterates over `b * t` frames; we scale the per-call chunk to
        # ``smplx_chunk_per_batch_multiplier * args.batch_size`` frames so it
        # tracks with whatever batch the user picked.
        multiplier = int(getattr(self.args, "smplx_chunk_per_batch_multiplier", 64))
        batch_size = max(1, multiplier * int(self.args.batch_size))

        def _full_forward_chunk(pose_chunk, transl_chunk, betas_chunk, expr_chunk, need_verts):
            out = smplx_model(
                betas=betas_chunk,
                transl=transl_chunk,
                expression=expr_chunk,
                jaw_pose=pose_chunk[:, 22, :],
                global_orient=pose_chunk[:, 0, :],
                body_pose=pose_chunk[:, 1:22, :].reshape(pose_chunk.shape[0], -1),
                left_hand_pose=pose_chunk[:, 25:40, :].reshape(pose_chunk.shape[0], -1),
                right_hand_pose=pose_chunk[:, 40:55, :].reshape(pose_chunk.shape[0], -1),
                leye_pose=pose_chunk[:, 23, :],
                reye_pose=pose_chunk[:, 24, :],
                return_verts=need_verts,
                return_joints=True,
            )
            return out.joints[:, :55, :], (out.vertices if need_verts else None)

        parity_steps = int(getattr(self.args, "smplx_fast_parity_check_steps", 0))
        parity_tol = float(getattr(self.args, "smplx_fast_parity_tol", 1e-5))
        parity_active = (parity_steps > 0) and (not no_grad)

        # We need the fast-path machinery whenever either:
        #   (a) the caller asked for joint-only (return_verts=False), or
        #   (b) parity check is enabled (we compare fast vs full on the first
        #       N grad-enabled calls regardless of which path the caller picked).
        need_fast_machinery = (not return_verts) or parity_active
        joints_out = []
        verts_out = []
        shapedirs = None
        j_rest_seq = None
        parity_full_calls = 0
        parity_full_ms = 0.0
        if need_fast_machinery:
            shapedirs = torch.cat([smplx_model.shapedirs, smplx_model.expr_dirs], dim=-1)
            # When betas/expressions are constant across frames within each
            # sequence (the common case), pre-compute the shape-blended rest
            # joints once per sequence instead of every chunk.
            cache_seq_shapes = (
                (betas is None or (torch.is_tensor(betas) and betas.ndim <= 2))
                and (expressions is None or (torch.is_tensor(expressions) and expressions.ndim <= 2))
            )
            if cache_seq_shapes:
                if not hasattr(self, "_smplx_seq_shape_cache_logged"):
                    logger.info("SMPL-X fast path: per-sequence j_rest cache enabled.")
                    self._smplx_seq_shape_cache_logged = True
                betas_seq = self._expand_framewise(betas, batch_size=b, nframes=1, dim=num_betas)[:b]
                expr_seq = self._expand_framewise(expressions, batch_size=b, nframes=1, dim=num_expr)[:b]
                shape_seq = torch.cat([betas_seq, expr_seq], dim=-1)
                v_shaped_seq = smplx_model.v_template.unsqueeze(0) + blend_shapes(shape_seq, shapedirs)
                j_rest_seq = vertices2joints(smplx_model.J_regressor, v_shaped_seq)

        def _fast_joints_chunk(pose_chunk, transl_chunk, betas_chunk, expr_chunk, chunk_start):
            if j_rest_seq is not None:
                seq_idx = torch.arange(
                    chunk_start, chunk_start + pose_chunk.shape[0],
                    device=pose_chunk.device, dtype=torch.long,
                ) // t
                j_rest = j_rest_seq[seq_idx]
                dtype_for_rigid = j_rest.dtype
            else:
                shape_components = torch.cat([betas_chunk, expr_chunk], dim=-1)
                assert shapedirs is not None
                v_shaped = smplx_model.v_template.unsqueeze(0) + blend_shapes(shape_components, shapedirs)
                j_rest = vertices2joints(smplx_model.J_regressor, v_shaped)
                dtype_for_rigid = shape_components.dtype

            full_pose = pose_chunk.reshape(pose_chunk.shape[0], -1)
            pose_mean = getattr(smplx_model, "pose_mean", None)
            if pose_mean is not None:
                full_pose = full_pose + pose_mean.view(1, -1).to(device=full_pose.device, dtype=full_pose.dtype)
            rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(pose_chunk.shape[0], -1, 3, 3)
            j_transformed, _ = batch_rigid_transform(rot_mats, j_rest, smplx_model.parents, dtype=dtype_for_rigid)
            return j_transformed[:, :55, :] + transl_chunk.unsqueeze(1)

        tic = time.perf_counter()
        grad_ctx = torch.no_grad() if no_grad else nullcontext()
        with grad_ctx:
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                pose_chunk = pose[start:end]
                betas_chunk = betas_f[start:end]
                expr_chunk = expr_f[start:end]
                transl_chunk = tr[start:end]

                primary_joints = None
                if return_verts:
                    primary_joints, full_verts = _full_forward_chunk(
                        pose_chunk, transl_chunk, betas_chunk, expr_chunk, need_verts=True
                    )
                    joints_out.append(primary_joints)
                    assert full_verts is not None
                    verts_out.append(full_verts)
                else:
                    primary_joints = _fast_joints_chunk(
                        pose_chunk, transl_chunk, betas_chunk, expr_chunk, chunk_start=start
                    )
                    joints_out.append(primary_joints)

                # Parity check: run the *other* path and compare joints. Fires
                # regardless of which path was primary, so the log surfaces for
                # standard codec configs (rec_ver_weight > 0 → full primary).
                if parity_active:
                    done = int(getattr(self, "_smplx_fast_parity_done", 0))
                    if done < parity_steps:
                        if return_verts:
                            # primary = full; run fast for comparison.
                            alt_joints = _fast_joints_chunk(
                                pose_chunk, transl_chunk, betas_chunk, expr_chunk, chunk_start=start,
                            )
                            full_joints_chk, fast_joints_chk = primary_joints, alt_joints
                        else:
                            # primary = fast; run full for comparison.
                            full_chk_tic = time.perf_counter()
                            alt_joints, _ = _full_forward_chunk(
                                pose_chunk, transl_chunk, betas_chunk, expr_chunk, need_verts=False,
                            )
                            parity_full_calls += 1
                            parity_full_ms += (time.perf_counter() - full_chk_tic) * 1000.0
                            full_joints_chk, fast_joints_chk = alt_joints, primary_joints

                        diff = (fast_joints_chk - full_joints_chk).abs()
                        max_abs = float(diff.max().item())
                        mean_abs = float(diff.mean().item())
                        status = "PASS" if max_abs <= parity_tol else "FAIL"
                        log_fn = logger.info if status == "PASS" else logger.warning
                        log_fn(
                            "SMPL-X fast parity step={}: max_abs={:.6g} mean_abs={:.6g} tol={:.6g} status={}",
                            done, max_abs, mean_abs, parity_tol, status,
                        )
                        self._smplx_fast_parity_done = done + 1

        joints = torch.cat(joints_out, dim=0).reshape(b, t, 55, 3)
        verts = torch.cat(verts_out, dim=0).reshape(b, t, -1, 3) if return_verts else None

        elapsed_ms = (time.perf_counter() - tic) * 1000.0
        timing = getattr(self, "_smplx_path_timing", None)
        if timing is None:
            timing = {"full_calls": 0, "full_ms": 0.0, "fast_calls": 0, "fast_ms": 0.0, "ratio_logged": False}
            self._smplx_path_timing = timing
        if return_verts:
            timing["full_calls"] += 1
            timing["full_ms"] += float(elapsed_ms)
        else:
            fast_elapsed_ms = max(0.0, float(elapsed_ms) - float(parity_full_ms))
            timing["fast_calls"] += 1
            timing["fast_ms"] += fast_elapsed_ms
            if parity_full_calls > 0:
                timing["full_calls"] += int(parity_full_calls)
                timing["full_ms"] += float(parity_full_ms)
        if (not return_verts) and (not getattr(self, "_smplx_fast_path_logged", False)):
            logger.info(
                "SMPL-X joint-only fast path active: return_verts=False skips vertex skinning "
                "(call={:.2f} ms, frames={}).", elapsed_ms, n,
            )
            self._smplx_fast_path_logged = True
        if (timing["full_calls"] > 0) and (timing["fast_calls"] > 0) and (not timing["ratio_logged"]):
            full_avg = timing["full_ms"] / max(1, timing["full_calls"])
            fast_avg = timing["fast_ms"] / max(1, timing["fast_calls"])
            speedup = (full_avg / fast_avg) if fast_avg > 0 else float("inf")
            logger.info(
                "SMPL-X timing: joint-only avg={:.2f} ms vs full avg={:.2f} ms ({:.2f}x)",
                fast_avg, full_avg, speedup,
            )
            timing["ratio_logged"] = True
        return joints, verts

    # ------------------------------------------------------------- MPJPE
    # Lightweight per-validation accumulator. The rvq trainers' val loops call
    # `_mpjpe_reset` at the start, `_mpjpe_update(rec_joints, tar_joints)` once
    # per batch (joints come for free from `_smplx_forward`), and `_mpjpe_log`
    # once after the loop ends. Stored as instance state instead of going
    # through `self.tracker` because we want a joint-count-weighted mean rather
    # than a per-batch arithmetic mean.

    def _mpjpe_reset(self) -> None:
        self._mpjpe_total_error = 0.0
        self._mpjpe_total_joints = 0

    def _mpjpe_update(self, pred_joints, tar_joints, mask=None) -> None:
        err = torch.linalg.norm(pred_joints.detach() - tar_joints.detach(), dim=-1)
        if mask is not None:
            m = mask.to(err.device).bool()
            if m.ndim == err.ndim - 1:
                m = m.unsqueeze(-1)
            m = m.expand_as(err)
            err = err * m.float()
            self._mpjpe_total_joints += int(m.sum().item())
        else:
            self._mpjpe_total_joints += int(err.numel())
        self._mpjpe_total_error += float(err.sum().item())

    def _mpjpe_mean(self) -> float:
        total = int(getattr(self, "_mpjpe_total_joints", 0))
        if total == 0:
            return 0.0
        return float(getattr(self, "_mpjpe_total_error", 0.0)) / float(total)

    def _mpjpe_log(self, epoch: int, split: str = "val") -> None:
        mean = self._mpjpe_mean()
        if self.global_rank == 0:
            writer = getattr(self, "writer", None)
            if writer is not None:
                writer.add_scalar(f"{split}/mpjpe", mean, epoch)
            logger.info(
                f"[GPU {self.global_rank}] {split} MPJPE: {mean:.4f} "
                f"(over {int(getattr(self, '_mpjpe_total_joints', 0))} joint observations)"
            )

    # logger = logging.getLogger()

    @staticmethod
    def get_model(args):

        _codec_kwargs_by_body_part = {
            "upper": loaders.get_uppergesturecodec_kwargs,
            "lower": loaders.get_lowergesturecodec_kwargs,
            "face": loaders.get_facegesturecodec_kwargs,
        }
        if args.body_part not in _codec_kwargs_by_body_part:
            raise ValueError(
                f"BaseCausalCodecTrainer.get_model: unsupported body_part={args.body_part!r}. "
                f"Expected one of {sorted(_codec_kwargs_by_body_part)} "
                f"(causal codec trainers are body-part specific)."
            )
        gesture_codec_kwargs = _codec_kwargs_by_body_part[args.body_part]()
        # gesture_codec = GestureBodyPartCodec(
        # gesture_codec = GestureCodec(
        # breakpoint()
        gesture_codec = GestureMimiCodec(
            num_frames=args.num_frames,
            frame_chunk_size=args.frame_chunk_size,
            nfeats=args.nfeats,
            motion_fps=args.motion_fps,
            num_heads=args.transformer_heads,
            # causal=True,
            # num_layers=args.transformer_layers,
            transformer_layers=args.transformer_layers,
            convblock_layers=args.convblock_layers,
            # decoder_sizefactor = args.decoder_sizefactor,
            # dtype=torch.float32 if args.model_dtype == "float32" else torch.bfloat16,
            latent_dtype=args.latent_dtype,
            use_wavelet = args.use_wavelet,
            **gesture_codec_kwargs
        )
        return gesture_codec 



    def train_recording(self, epoch, its, t_data, t_train, mem_cost, lr_g, lr_d=None):
        pstr = f"[GPU{self.global_rank}:{self.local_rank}] [{epoch:03}][{its:03}/{self.train_length:03d}]"
        for name, states in self.tracker.loss_meters.items():
            metric = states['train']
            if metric.count > 0:
                pstr += f"{name}: {metric.avg:.4f}\t"
                if self.global_rank == 0:self.writer.add_scalar(f"train/{name}", metric.avg, epoch*self.train_length+its)
        pstr += f"glr: {lr_g:.1e}\t"
        if self.global_rank == 0:self.writer.add_scalar("lr/glr", lr_g, epoch*self.train_length+its)
        if lr_d is not None:
            pstr += f"dlr: {lr_d:.1e}\t"
            if self.global_rank == 0:self.writer.add_scalar("lr/dlr", lr_d, epoch*self.train_length+its)
        pstr += f"dtime: {t_data*1000:04}\t"        
        pstr += f"ntime: {t_train*1000:04}\t"
        pstr += f"mem: {mem_cost*len(self.args.gpus):.2f} "
        logger.info(pstr)
        
     
    def val_recording(self, epoch):
        pstr_curr = f"[GPU{self.global_rank}:{self.local_rank}] Curr info >>>>  "
        pstr_best = f"[GPU{self.global_rank}:{self.local_rank}] Best info >>>>  "
        for name, states in self.tracker.loss_meters.items():
            metric = states['val']
            if metric.count > 0:
                pstr_curr += f"{name}: {metric.avg:.4f}     \t"
                if epoch != 0:
                    if self.global_rank == 0:self.writer.add_scalars(f"val/{name}", {name+"_val":metric.avg, name+"_train":states['train'].avg}, epoch*self.train_length)
                    new_best_train, new_best_val = self.tracker.update_and_plot(name, epoch, self.checkpoint_path+f"{name}_{self.args.name+self.args.notes}.png")
                    # if new_best_val:
                        # save_checkpoints(os.path.join(self.checkpoint_path, f"{name}.bin"), self.model, opt=None, epoch=None, lrs=None)        
        for k, v in self.tracker.values.items():
            metric = v['val']['best']
            if self.tracker.loss_meters[k]['val'].count > 0:
                pstr_best += f"{k}: {metric['value']:.3f}({metric['epoch']:03d})\t"
        logger.info(pstr_curr)
        logger.info(pstr_best)
   
    def test_recording(self, dict_name, value, epoch):
        self.tracker.update_meter(dict_name, "test", value)
        _ = self.tracker.update_values(dict_name, 'test', epoch)