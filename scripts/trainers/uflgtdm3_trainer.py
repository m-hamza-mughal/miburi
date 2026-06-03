import smplx
import os
import torch
import torch.nn.functional as F
import numpy as np
# import logging
from loguru import logger
import time
import numpy as np
import copy
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from torch.utils.tensorboard import SummaryWriter

from .baseglm_trainer import BaseGLMTrainer
from miburi.models import loaders
from miburi.models import GTemporalDepthModel3, GestureLMGen

from .utils import rotation_conversions as rc
from .utils.loss_factory import get_loss_func
from .utils import tools as other_tools
from .dataloaders.utils.visualize import (
    render_smplx_debug_video,
    stitch_videos_hstack,
    mux_audio_into_video,
)
from .utils.mixed_precision import (
    prepare_mixed_precision, 
    upcast_mixed_precision,
    downcast_mixed_precision
)


# logger = logging.getLogger()


class UpperFaceLowerGTDM3Trainer(BaseGLMTrainer):
    def __init__(self, args):
        super(UpperFaceLowerGTDM3Trainer, self).__init__(args)
        

        self.tracker = other_tools.EpochTracker(
            ["ce_loss", "ce_upper", "ce_lower", "ce_face", "perplexity", "cont_upper_loss", "cont_face_loss", "cont_lower_loss", "embody_upper_cont_loss", "embody_face_mmd_loss", "embody_lower_mmd_loss", "mse_upper_loss", "mse_face_loss", "mse_lower_loss", "mse_transl_loss", "embody_mse_upper_loss", "vad_loss"], 
            [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
            )

        # self.ce_temporal = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=self.model.initial_token_id)
        # self.ce_depth = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=self.model.initial_token_id)
        # self.temporal_depth_consistency = torch.nn.MSELoss(reduction="none")
        self.modelout_ignore_index = self.model.module.pad_token_id if self.args.ddp else self.model.pad_token_id
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=self.modelout_ignore_index)
        if self.args.contrastive_loss_weight > 0:
            self.contrastive_loss_func = get_loss_func("contrastive_latent_loss", temperature=0.1, segment_len=12)
        

        self.param_dtype = getattr(torch, args.param_dtype)
        self.optim_dtype = getattr(torch, args.optim_dtype)

        self.body_parts = 3  # upper, face, lower

        # assert self.param_dtype == torch.float32, "param_dtype must be float32"

        prepare_mixed_precision(
            self.model.parameters(), 
            param_dtype=self.param_dtype, 
            optim_dtype=self.optim_dtype
        )

        if not self.args.is_train:
            from .utils.metrics import GestureMetrics # lazy import to avoid circular import

            self.gesture_metrics = GestureMetrics(self.args) # if visualize else None
    
    def get_model(self, args):

        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(loaders.DEFAULT_REPO)
        
        lm = checkpoint_info.get_moshi()
        text_procemb = copy.deepcopy(lm.text_emb.weight.data)
        audio_procemb = [copy.deepcopy(a_emb.weight.data) for a_emb in lm.emb[:8]]
        # breakpoint()  # get mimi cardinality and text cardinality 
        
        # print(text_procemb.weight[0])
        # print(audio_procemb[0].weight[0])
        # text_procemb = text_procemb.eval()
        # audio_procemb = audio_procemb.eval()
        del lm
        logger.info(f"[GPU{self.global_rank}:{self.local_rank}] text/audio emb processors loaded")

        # for p in text_procemb.parameters():
        #     p.requires_grad = False
        # for p in audio_procemb.parameters():
        #     p.requires_grad = False

        # breakpoint()
        mimi_frame_rate = loaders.FRAME_RATE
        gesture_lm_kwargs = loaders.get_gesturelm_kwargs()
        # mimi_cardinality = loaders._quantizer_kwargs["bins"]
        # breakpoint()
        # text_tokenizer = checkpoint_info.get_text_tokenizer()
        # logger.info(f"Text tokenizer loaded {text_tokenizer.vocab_size()}")

        # breakpoint() # check lens of each layer
        # each layer of codec layers takes in [*] e.g. B, T and returns [*, D] e.g. B, T, D
        upper_codec_layers = copy.deepcopy(self.upper_gesture_codec.quantizer.vq.layers) # nn.ModuleList)
        lower_codec_layers = copy.deepcopy(self.lower_gesture_codec.quantizer.vq.layers) # nn.ModuleList
        face_codec_layers = copy.deepcopy(self.face_gesture_codec.quantizer.vq.layers) # nn.ModuleList
        
        assert len(upper_codec_layers) + len(face_codec_layers) + len(lower_codec_layers) == gesture_lm_kwargs["n_q"], \
            f"Number of codec layers {len(upper_codec_layers)} + {len(face_codec_layers)} + {len(lower_codec_layers)} does not match n_q {gesture_lm_kwargs['n_q']}"
        gesture_codec_layers = upper_codec_layers + lower_codec_layers + face_codec_layers
        # gesture_codec_layers = upper_codec_layers + lower_codec_layers + face_codec_layers

        body_parts = 3

        # freeze the gesture codec layers
        for gcodec_layer in gesture_codec_layers:
            for p in gcodec_layer.parameters():
                p.requires_grad = False
        gesture_codec_layers.eval()

        # gesturelm = GLMModel(
        # gesturelm = GLMModelAudio(
        # gesturelm = GTemporalDepthModel(
        gesturelm = GTemporalDepthModel3(
            num_heads=args.gestureformer_heads,
            num_layers=args.gestureformer_layers,
            depformer_heads=args.gestureformer_depformer_heads,
            depformer_layers=args.gestureformer_depformer_layers,
            query2mem_scale=self.codec_difference,
            num_temp_classifiers=args.num_temp_classifiers,
            dtype=torch.float32 if args.param_dtype == "float32" else torch.bfloat16,
            text_procemb=text_procemb,
            audio_procemb=audio_procemb,
            gesture_codec_layers=gesture_codec_layers,
            vad_guidance=args.vad_guidance,
            vad_use_face_logits=args.vad_use_face_logits,
            body_parts=body_parts,
            bp_dist=None, #  [0]*8 + [1]*8 + [2]*4, # [0]*8 + [1]*4 + [2]*8, #  # Example for 20 codebooks with 3 body parts 8 upper, 4 face, 8 lower # len is K
            textaudio_emb_freeze=args.textaudio_emb_freeze,
            **gesture_lm_kwargs
        )

        return gesturelm
    
    def process_conditions(self, audio_codes, text_codes):
        """
        Process the condition tensors using text_procemb and audio_procemb
        
        Args:
            audio_codes: B x K=8 x T=125
            text_codes: B x K=1 x T=125
        Returns:
            audio_codes: B x K=1 x T=125 x dim (added across K)
            text_codes: B x K=1 x T=125 x dim
        """
        # breakpoint()
        audio_embs = []
        for k in range(audio_codes.shape[1]):
            audio_emb = self.audio_procemb[k](audio_codes[:, k, :])
            audio_embs.append(audio_emb)
        audio_emb = torch.stack(audio_embs, dim=1) # B x K=8 x T=125 x dim
        audio_emb = audio_emb.sum(dim=1, keepdim=True) # B x K=1 x T=125 x dim

        if text_codes is not None:
            text_emb = self.text_procemb(text_codes)
        else:
            text_emb = None

        return audio_emb, text_emb
    
    def calculate_perplexity(self, logits, target_tokens, pad_token_id=None):
        """
        Calculate perplexity from model logits.
        
        Args:
            logits: Model output logits of shape [B, K, T, vocab_size]
            target_tokens: Ground truth tokens of shape [B, K, T]
            
        Returns:
            perplexity: The perplexity score
        """
        B, K, T, vocab_size = logits.shape
        
        # Flatten the batch, codebook, and time dimensions
        logits_flat = logits.reshape(B*K*T, vocab_size)
        targets_flat = target_tokens.reshape(B*K*T)
        
        # Convert logits to log probabilities (applying log softmax)
        log_probs = F.log_softmax(logits_flat, dim=-1)
        
        # Get the log probability of the target tokens
        target_log_probs = log_probs[torch.arange(B*K*T), targets_flat]
        
        mask = (targets_flat != pad_token_id)

        # Calculate negative log likelihood
        nll = -target_log_probs[mask].mean()
        
        # Calculate perplexity
        perplexity = torch.exp(nll)
        
        return perplexity.item()
    
    @staticmethod
    def inject_token_noise(tokens, noise_prob=0.05, vocab_size=None):
        # tokens: batch_size x K x T
        # breakpoint()
        noise_mask = (torch.rand(tokens.shape) < noise_prob).to(tokens.device)
        random_tokens = torch.randint(0, vocab_size, tokens.shape, device=tokens.device)
        noisy_tokens = torch.where(noise_mask, random_tokens, tokens)
        return noisy_tokens
    
    @staticmethod
    def add_embedding_noise(memory, noise_std=0.01):
        noise = torch.randn_like(memory) * noise_std
        return memory + noise
    
    @staticmethod
    def dropout_memory_tokens(memory, replace_token, dropout_prob=0.1, ):
        # TODO: add replace token to model emb
        # breakpoint()
        mask = torch.rand(memory.size(0), 1, 1, device=memory.device) > dropout_prob
        mask = mask.expand(-1, memory.size(1), memory.size(2))
        memory = torch.where(mask, memory, replace_token)

        return memory
    
    def _gumbel_logits_to_q_latent(self, logits, codec, pad_id, tau=1.0, hard=True):
        """
        logits: [B, K, T, V]
        returns:
          q_latent: [B, T, D]  (summed over K RVQ stages)
          probs:   [B, K, T, V] (one-hot if hard=True, soft otherwise)
        """
        # avoid in-place on shared tensor
        sample_logits = logits.clone()
        # sample_logits[..., pad_id] = -1e9
        #slice out pad token logits
        assert sample_logits.shape[-1] == pad_id + 1, "Pad token id does not match vocab size"
        sample_logits = sample_logits[..., :pad_id]

        probs = F.gumbel_softmax(sample_logits, tau=tau, hard=hard, dim=-1)  # [B,K,T,V]

        # project per-stage one-hot to embeddings, then sum RVQ stages
        latents_per_k = []
        for k in range(codec.num_codebooks):
            # breakpoint()
            # get codebook matrix [V, D]
            layer = codec.quantizer.vq.layers[k]
            cb = layer.embedding
            # probs_k: [B, T, V]
            probs_k = probs[:, k]  # select stage k
            # [B,T,V] x [V,D] -> [B,T,D]
            cb = cb.to(probs_k.dtype)
            # breakpoint()
            lat_k = torch.einsum("btv,vd->btd", probs_k, cb)
            latents_per_k.append(lat_k)

        q_latent = torch.stack(latents_per_k, dim=1).sum(dim=1)  # sum RVQ stages -> [B,T,D]
        return q_latent, probs

    def _debug_visualize_batch(self, dict_data, epoch: int, its: int, max_samples: int = 3):
        """Render a few raw samples from the batch as SMPLX videos.

        Only fires when ``args.debug`` is on. Runs on the first batch of each
        epoch from rank 0 only. Renders are best-effort: a failed render logs a
        warning rather than killing training.
        """
        if not getattr(self.args, "debug", False):
            return
        if its != 0:
            return
        if self.global_rank != 0:
            return

        out_dir = os.path.join(self.checkpoint_path, "debug_viz", f"epoch_{epoch:03d}")
        os.makedirs(out_dir, exist_ok=True)

        motion = dict_data.get("motion")
        transl = dict_data.get("transl")
        if motion is None or transl is None:
            logger.warning("[debug_viz] batch missing motion or transl; skipping.")
            return

        motion_np = motion.detach().cpu().float().numpy()         # (B, T, 165)
        transl_np = transl.detach().cpu().float().numpy()          # (B, T, 3)
        expr = dict_data.get("expressions")
        expr_np = expr.detach().cpu().float().numpy() if expr is not None else None
        betas = dict_data.get("beta")
        betas_np = betas.detach().cpu().float().numpy() if betas is not None else None
        sample_names = dict_data.get("filechunk_id") or []
        dataset_name = dict_data.get("dataset_name") or []

        B = motion_np.shape[0]
        n_to_render = min(max_samples, B)
        logger.info(f"[debug_viz] epoch {epoch}: rendering {n_to_render} sample(s) to {out_dir}")

        for i in range(n_to_render):
            try:
                sample_id = (
                    sample_names[i].replace("/", "_") if i < len(sample_names) and isinstance(sample_names[i], str)
                    else f"sample_{i}"
                )
                dset_tag = dataset_name[i] if i < len(dataset_name) else "unknown"
                output_path = os.path.join(out_dir, f"{sample_id}__{dset_tag}.mp4")

                # First-frame betas only (per-frame broadcast is identity by construction).
                betas_i = betas_np[i, 0] if betas_np is not None else None
                render_smplx_debug_video(
                    smplx_model=self.smplx_model,
                    poses=motion_np[i].astype(np.float32),
                    transl=transl_np[i].astype(np.float32),
                    expressions=expr_np[i].astype(np.float32) if expr_np is not None else None,
                    betas=betas_i,
                    output_path=output_path,
                    fps=int(self.args.motion_fps),
                    width=640,
                    height=960,
                    audio_path=None,
                )
            except Exception as exc:
                logger.warning(f"[debug_viz] sample {i} failed: {exc}")

    def train(self, epoch):
        gc.collect()
        self.upper_gesture_codec.eval()
        self.lower_gesture_codec.eval()
        self.face_gesture_codec.eval()

        self.model.train()
        t_start = time.time()
        self.tracker.reset()
        if self.args.is_continue and epoch != 0:
            self.opt_s.step(epoch)
        for its, dict_data in enumerate(self.train_loader):
            # if its == 1:
            #     break
            # breakpoint()
            tar_pose_upper = dict_data["motion_upper"]
            tar_pose_hands = dict_data["motion_hands"]
            tar_pose_face = dict_data["motion_face"]

            tar_pose_lower = dict_data["motion_lower"]
            tar_trans = dict_data["transl"].to(self.local_rank)
            tar_contact = dict_data["contact"].to(self.local_rank)
            tar_exps = dict_data["expressions"].to(self.local_rank)
            # print(dict_data["sample_name"])

            sample_names = dict_data["filechunk_id"]

            # Debug-mode batch visualization: render a few samples at the start
            # of each epoch so we can eyeball what the model is actually being
            # fed. No-op when --debug is off; rank-0 only.
            self._debug_visualize_batch(dict_data, epoch=epoch, its=its)

            # breakpoint()
            tar_audio_tokens = dict_data["audio_tokens"]
            tar_text_tokens = dict_data["text_tokens"]
            tar_spk = dict_data["speaker_id"] 

            lower_valid_mask = dict_data["lower_valid_mask"].to(self.local_rank)
            assert lower_valid_mask is not None, "Lower valid mask should not be None in training"

            if self.args.dataset_ratio == "full_beatx_lowervalid":
                assert not (lower_valid_mask == 0).any(), "Lower valid mask should not be zero in training" 
        

            dataset_name_per_item = dict_data['dataset_name']
            dataset_name_per_item = np.array(dataset_name_per_item)
            
            # get indices for seamless interaction and beatx
            # breakpoint()
            # seamless_indices = np.where(dataset_name_per_item != 'BEATX')[0]
            # beatx_indices = np.where(dataset_name_per_item == 'BEATX')[0]
            seamless_indices = torch.where(lower_valid_mask == 0)[0].to(self.local_rank)
            beatx_indices = torch.where(lower_valid_mask == 1)[0].to(self.local_rank)
            # breakpoint()

            if self.args.vad_guidance:
                # breakpoint()
                tar_vad = dict_data["vad_bits"].to(self.local_rank) # B x 250

                vad_loss_mask = tar_vad != -1 # B x 250
                vad_loss_mask = vad_loss_mask.all(dim=-1) # B

                # pool in the time dimension from 250 to 125 using framechunksize
                tar_vad = tar_vad.unfold(1, self.args.frame_chunk_size, self.args.frame_chunk_size)
                # breakpoint() # change the threshold for vad
                tar_vad = (tar_vad.sum(dim=-1) >= self.args.frame_chunk_size/2).to(self.param_dtype) # B x 125
                tar_vad = torch.where(vad_loss_mask[:, None], tar_vad, -1 * torch.ones_like(tar_vad)) # set to -1 if any frame is -1 in the chunk
                


            # subtract the first frame translation
            tar_trans[:, :, 0] = tar_trans[:, :, 0] - tar_trans[:, 0:1, 0]
            tar_trans[:, :, 2] = tar_trans[:, :, 2] - tar_trans[:, 0:1, 2]

            # upper_hands_joint_mask = self.train_data.upper_mask_for_flattened + \
            #     self.train_data.hands_mask_for_flattened
            # lower_joint_mask = self.train_data.lower_mask_for_flattened 
            # upper_lower_joint_mask = upper_hands_joint_mask + lower_joint_mask

            # -- lower motion -- #
            tar_trans_vel = other_tools.estimate_linear_velocity(
                tar_trans, dt=1/self.args.motion_fps
            )
            tar_pose_lower = tar_pose_lower.to(self.local_rank)
            
            # breakpoint()
            bs, n, ldim = tar_pose_lower.shape
            lj = ldim //3
            tar_pose_lower = rc.axis_angle_to_matrix(tar_pose_lower.reshape(bs, n, lj, 3))
            tar_pose_lower = rc.matrix_to_rotation_6d(tar_pose_lower).reshape(bs, n, lj*6)

            # -- upper motion -- #
            tar_pose_upper = tar_pose_upper.to(self.local_rank)
            tar_pose_hands = tar_pose_hands.to(self.local_rank)

            bs, n, udim = tar_pose_upper.shape
            uj = udim //3
            tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, uj, 3))
            tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, uj*6)

            hj = tar_pose_hands.shape[-1]
            hj = hj // 3
            tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, hj, 3))
            tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, hj*6)

            # -- face motion -- #
            tar_pose_face = tar_pose_face.to(self.local_rank)
            bs, n, fdim = tar_pose_face.shape
            fj = fdim //3
            tar_pose_face = rc.axis_angle_to_matrix(tar_pose_face.reshape(bs, n, fj, 3))
            tar_pose_face = rc.matrix_to_rotation_6d(tar_pose_face).reshape(bs, n, fj*6)

            
            j = uj + hj + fj + lj
            
            
            tar_audio_tokens = tar_audio_tokens.to(self.local_rank)
            tar_text_tokens = tar_text_tokens.to(self.local_rank)
            tar_spk = tar_spk.to(self.local_rank)

            t_data = time.time() - t_start
            
            
            if epoch >= self.args.pretrain_warmup_epochs:
                audio_codes = tar_audio_tokens # B x K=8 x T=125
                text_codes = tar_text_tokens # B x K=1 x T=125

                if self.args.memory_dropout_prob > 0:
                    # assert self.args.memory_embnoise_prob == 0
                    # decreasing prob from 1 to self.args.memory_dropout_prob
                    # at self.args.pretrain_warmup_epochs* 2 epoch, it reaches its desired value
                    drop_prob = max(
                        self.args.memory_dropout_prob,
                        1 - (its + (epoch - self.args.pretrain_warmup_epochs) * len(self.train_loader)) / (self.args.pretrain_warmup_epochs*len(self.train_loader))
                    )
                    # drop_prob = self.args.memory_dropout_prob
                    # print("drop_prob", drop_prob)
                    # breakpoint() # check the cardinality and vocab size
                    audio_codes = self.dropout_memory_tokens(audio_codes, self.audio_codec_nulltoken, drop_prob)
                    text_codes = self.dropout_memory_tokens(text_codes, self.text_codec_nulltoken, drop_prob)
                    
                    if its % self.args.log_period == 0 and drop_prob > self.args.memory_dropout_prob and self.global_rank == 0:
                        # breakpoint()
                        logger.info(f"Applied memory dropout with prob {drop_prob:.4f} at epoch {epoch}, iter {its}")

            else:
                # breakpoint() # check number of codebooks of mimi
                audio_codes = torch.full(
                    tar_audio_tokens.shape, # (tar_audio.shape[0], self.mimi.num_codebooks, upper_codes.shape[2] * self.codec_difference),
                    self.audio_codec_nulltoken,
                    device=tar_audio_tokens.device,
                    dtype=torch.long,
                )
                text_codes = torch.full(
                    tar_text_tokens.shape, # (tar_audio.shape[0], 1, upper_codes.shape[2] * self.codec_difference),
                    self.text_codec_nulltoken,
                    device=tar_audio_tokens.device,
                    dtype=torch.long,
                )


            


            # convert audio/text codes to emb using moshi-lm 
            # (moved to model forward pass)
            # audio_emb, text_emb = self.process_conditions(audio_codes, text_codes) # B x K=1 x T=125 x dim
            
            # # # reshape audio and text codes 
            # audio_emb = audio_emb.squeeze(1) # B x T=125 x dim
            # text_emb = text_emb.squeeze(1) # B x T=125 dim

            # # if self.args.memory_embnoise_prob > 0 and epoch > self.args.pretrain_warmup_epochs:
            # #     # breakpoint()
            # #     assert self.args.memory_dropout_prob == 0
            # #     # some equation that maps epoch to increasing prob from 0 to self.args.memory_embnoise_prob
            # #     mem_emb_noise = min(
            # #         self.args.memory_embnoise_prob, 
            # #         self.args.memory_embnoise_prob * (epoch - self.args.pretrain_warmup_epochs + 1) / (self.args.pretrain_warmup_epochs)
            # #     )
            # #     # print("mem_emb_noise", mem_emb_noise)
            # #     audio_emb = self.add_embedding_noise(audio_emb, mem_emb_noise)
            # #     text_emb = self.add_embedding_noise(text_emb, mem_emb_noise)
            
            # conditions = (audio_emb, text_emb)
            
            self.opt.zero_grad()
            g_loss_final = 0
            # breakpoint()

            in_tar_pose_upper = torch.cat((tar_pose_upper, tar_pose_hands), dim=-1) # 258
            # in_tar_pose_lower = torch.cat((tar_pose_lower, tar_trans, tar_contact), dim=-1)
            in_tar_pose_lower = torch.cat((tar_pose_lower, tar_trans_vel, tar_contact), dim=-1) # 61
            in_tar_pose_face = torch.cat((tar_pose_face, tar_exps), dim=-1) # 106
            


            # lower_valid_mask = torch.zeros_like(lower_valid_mask, device=self.rank)
            # in_tar_pose_lower = torch.zeros_like(in_tar_pose_lower, device=self.rank)
            
            # breakpoint()
            with torch.no_grad():
                # breakpoint()
                upper_codes = self.upper_gesture_codec.encode(in_tar_pose_upper) # B x K=8 x T=25
                lower_codes = self.lower_gesture_codec.encode(in_tar_pose_lower) # B x K=8 x T=25
                face_codes = self.face_gesture_codec.encode(in_tar_pose_face) # B x K=4 x T=25
                # if len(seamless_indices) > 0: breakpoint() # check lower codes for beatx samples
                lower_codes[seamless_indices] = self.modelout_ignore_index # set to pad token for seamless interaction samples
                face_codes[seamless_indices] = self.modelout_ignore_index # set to pad token for seamless interaction samples


            gesture_tokens = torch.cat((upper_codes, lower_codes, face_codes), dim=1) # B x K=20 x T=25
            pad_loss_mask = (gesture_tokens != self.modelout_ignore_index).float() # B x K x T
            
            # gradually increase lower body part drop prob after pretrain_warmup_epochs*2
            if epoch > self.args.pretrain_warmup_epochs*2:
                # breakpoint()
                masked_lower_codes = lower_codes.clone()
                masked_face_codes = face_codes.clone()
                drop_prob = min((epoch/(self.args.pretrain_warmup_epochs * 4)) * self.args.lower_bodypart_dropprob, self.args.lower_bodypart_dropprob)
                if its == 0 and drop_prob < self.args.lower_bodypart_dropprob:
                    logger.info(f"Epoch {epoch}: Using lower body part drop prob {drop_prob:.4f}")
                valid_indices = beatx_indices
                num_valid = len(valid_indices)
                num_to_drop = int(num_valid * drop_prob)
                if num_to_drop > 0:
                    # breakpoint() # check lower drop indices and face drop indices
                    lower_drop_indices = valid_indices[torch.randperm(num_valid)[:num_to_drop]]
                    face_drop_indices = valid_indices[torch.randperm(num_valid)[:num_to_drop]]
                    masked_lower_codes[lower_drop_indices] = self.modelout_ignore_index
                    masked_face_codes[face_drop_indices] = self.modelout_ignore_index
            else:
                masked_lower_codes = lower_codes.clone()
                masked_face_codes = face_codes.clone()
            
            
            # audio codes: B x K=8 x T=125 
            # text codes: B x K=1 x T=125
            conditions = (audio_codes, text_codes) 
            
            

            # flatten batches adn time in forward pass
            in_gesture_tokens = torch.cat((upper_codes, masked_lower_codes, masked_face_codes), dim=1) 
            
            # breakpoint() # check the mask
            lower_cross_attn_mask = torch.zeros_like(gesture_tokens).to(torch.bool) # B x K=20 x T
            lower_cross_attn_mask[
                :, 
                self.upper_gesture_codec.num_codebooks : self.upper_gesture_codec.num_codebooks + self.lower_gesture_codec.num_codebooks, 
                :] = True # only apply cross-attention on upper and face tokens

            model_out = self.model(
                in_gesture_tokens, 
                audio_codes=audio_codes,
                text_codes=text_codes, 
                sum_condition=tar_spk,
                ca_depth_padding_mask=lower_cross_attn_mask if self.args.drop_lower_crossattn else None,
            ) # B x K=16 x T=25 x cardinality
            if self.args.vad_guidance:
                model_out, vad_logits = model_out
                
            
            # flatten labels and logits
            # breakpoint() # change shape pf temp logits and num temp classifiers
            # print(model_out)

            # TODO: Change loss 
            logits = model_out #.logits

            ce_loss = 0
            B, K, T, card = logits.shape
            upper_loss = 0
            lower_loss = 0
            face_loss = 0
            
            # breakpoint() # check K 
            for k in range(K):
                loss_k = F.cross_entropy(
                    logits[:, k].reshape(B*T, card),
                    gesture_tokens[:, k].reshape(B*T), 
                    reduction="none"
                )
                # breakpoint() # check pad_mask
                loss_k = loss_k * pad_loss_mask[:, k].reshape(B*T) 
                loss = loss_k.sum() / (pad_loss_mask[:, k].sum() + 1e-12)
                loss_k = loss * (1/K)

                if k < 8:  # upper body + hands
                    upper_loss += loss_k.item()
                elif k >=8 and k < 16:   # face + expressions
                    lower_loss += loss_k.item()
                else:  # lower body
                    face_loss += loss_k.item()
                    loss_k = loss_k * self.args.face_loss_weight # upweight face loss
                
                ce_loss += loss_k

            ce_loss = ce_loss #/ K
            upper_loss = upper_loss #/ (K/2)
            lower_loss = lower_loss #/ (K/2)
            face_loss = face_loss #/ (K/4)

            g_loss_final += ce_loss
            self.tracker.update_meter("ce_loss", "train", ce_loss.item())
            self.tracker.update_meter("ce_upper", "train", upper_loss)
            self.tracker.update_meter("ce_face", "train", face_loss)
            self.tracker.update_meter("ce_lower", "train", lower_loss)
            
            perplexity = self.calculate_perplexity(
                logits, 
                gesture_tokens, 
                pad_token_id=self.modelout_ignore_index
            )
            self.tracker.update_meter("perplexity", "train", perplexity)

            # ---------------- VAD Loss ---------------- #
            if self.args.vad_guidance and epoch > self.args.pretrain_warmup_epochs:
                # breakpoint()
                # compute vad loss
                vad_loss = F.binary_cross_entropy_with_logits(
                    vad_logits,
                    tar_vad,
                    reduction="none"
                ) # B x T

                vad_loss[beatx_indices] *= 1.0
                vad_loss[seamless_indices] *= 0.1

                vad_loss = vad_loss * vad_loss_mask.unsqueeze(-1).to(vad_loss.dtype) # B x T
                vad_loss = vad_loss.sum() / (vad_loss_mask.sum() + 1e-12)
                vad_loss = vad_loss * self.args.vad_loss_weight
                g_loss_final += vad_loss
                self.tracker.update_meter("vad_loss", "train", vad_loss.item())

            # ---------------- GAN Loss ---------------- #
            # # decode from predicted tokens
            # if self.args.gan_loss_weight > 0 and epoch > self.args.pretrain_warmup_epochs*2:
            #     upper_logits = logits[:, :self.upper_gesture_codec.num_codebooks, :, :] # B x K=8 x T=25 x cardinality
            #     lower_logits = logits[:, self.upper_gesture_codec.num_codebooks:, :, :] # B x K=8 x T=25 x cardinality
            #     with torch.no_grad():
            #         pred_upper_tokens = torch.argmax(upper_logits, dim=-1) # B x K=8 x T=25
            #         pred_lower_tokens = torch.argmax(lower_logits, dim=-1) # B x K=8 x T=25

            #         pred_upper_pose = self.upper_gesture_codec.decode(pred_upper_tokens) # B x T x upper_dim
            #         pred_lower_pose = self.lower_gesture_codec.decode(pred_lower_tokens) # B x T x lower_dim
            

            # # ---------------- Contrastive Loss ---------------- #
            # if self.args.contrastive_loss_weight > 0 and epoch > self.args.pretrain_warmup_epochs:
            #     lambda_contrastive = (its / len(self.train_loader)) * self.args.contrastive_loss_weight if epoch - self.args.pretrain_warmup_epochs == 1 else self.args.contrastive_loss_weight
            #     if its % self.args.log_period == 0 and lambda_contrastive < self.args.contrastive_loss_weight and self.global_rank == 0:
            #             # breakpoint()
            #             logger.info(f"lambda_contrastive: {lambda_contrastive:.4f} at epoch {epoch}, iter {its}")

            #     # breakpoint() # check slicing
            #     upper_logits = logits[:, :self.upper_gesture_codec.num_codebooks, :, :] # B x K=8 x T=25 x cardinality
            #     lower_logits = logits[:, self.upper_gesture_codec.num_codebooks:self.upper_gesture_codec.num_codebooks + self.lower_gesture_codec.num_codebooks, :, :] # B x K=8 x T=25 x cardinality
            #     face_logits = logits[:, self.upper_gesture_codec.num_codebooks + self.lower_gesture_codec.num_codebooks:, :, :] # B x K=4 x T=25 x cardinality
            #     with torch.no_grad():
            #         # breakpoint()
            #         # assign pad logits to -Inf before argmax
            #         upper_logits[..., self.modelout_ignore_index] = -1e9
            #         face_logits[..., self.modelout_ignore_index] = -1e9
            #         lower_logits[..., self.modelout_ignore_index] = -1e9

            #         # now take argmax
            #         pred_upper_tokens = torch.argmax(upper_logits, dim=-1) # B x K=8 x T=25
            #         pred_face_tokens = torch.argmax(face_logits, dim=-1) # B x K=4 x T=25
            #         pred_lower_tokens = torch.argmax(lower_logits, dim=-1) # B x K=8 x T=25

            #         assert (pred_upper_tokens != self.modelout_ignore_index).all()
            #         assert (pred_face_tokens != self.modelout_ignore_index).all()
            #         assert (pred_lower_tokens != self.modelout_ignore_index).all()

            #     if len(beatx_indices) > 0:
            #         with torch.no_grad():

            #             beatx_gt_upper_tokens = upper_codes[beatx_indices]
            #             beatx_gt_face_tokens = face_codes[beatx_indices]
            #             beatx_gt_lower_tokens = lower_codes[beatx_indices]

            #             beatx_pred_upper_tokens = pred_upper_tokens[beatx_indices]
            #             beatx_pred_face_tokens = pred_face_tokens[beatx_indices]
            #             beatx_pred_lower_tokens = pred_lower_tokens[beatx_indices]

            #             beatx_pred_upper_q_latent = self.upper_gesture_codec.decode_latent(beatx_pred_upper_tokens) # B x T x latent_dim
            #             beatx_pred_face_q_latent = self.face_gesture_codec.decode_latent(beatx_pred_face_tokens) # B x T x latent_dim
            #             beatx_pred_lower_q_latent = self.lower_gesture_codec.decode_latent(beatx_pred_lower_tokens) # B x T x latent_dim

            #             beatx_gt_upper_q_latent = self.upper_gesture_codec.decode_latent(beatx_gt_upper_tokens) # B x T x latent_dim
            #             beatx_gt_face_q_latent = self.face_gesture_codec.decode_latent(beatx_gt_face_tokens) # B x T x latent_dim
            #             beatx_gt_lower_q_latent = self.lower_gesture_codec.decode_latent(beatx_gt_lower_tokens) # B x T x latent_dim
            #             beatx_gt_upper_q_latent = beatx_gt_upper_q_latent.detach()
            #             beatx_gt_face_q_latent = beatx_gt_face_q_latent.detach()
            #             beatx_gt_lower_q_latent = beatx_gt_lower_q_latent.detach()

                        
            #         upperbeat_contrastive_loss = self.contrastive_loss_func.compute_contrastive_loss(
            #             beatx_gt_upper_q_latent, beatx_pred_upper_q_latent, 
                        
            #         )
            #         upperbeat_contrastive_loss = upperbeat_contrastive_loss * lambda_contrastive
            #         self.tracker.update_meter("cont_upper_loss", "train", upperbeat_contrastive_loss.item())

            #         facebeat_contrastive_loss = self.contrastive_loss_func.compute_contrastive_loss(
            #             beatx_gt_face_q_latent, beatx_pred_face_q_latent, 
                        
            #         )
            #         facebeat_contrastive_loss = facebeat_contrastive_loss * lambda_contrastive
            #         self.tracker.update_meter("cont_face_loss", "train", facebeat_contrastive_loss.item())

            #         lowerbeat_contrastive_loss = self.contrastive_loss_func.compute_contrastive_loss(
            #             beatx_gt_lower_q_latent, beatx_pred_lower_q_latent, 
                        
            #         )
            #         lowerbeat_contrastive_loss = lowerbeat_contrastive_loss * lambda_contrastive
            #         self.tracker.update_meter("cont_lower_loss", "train", lowerbeat_contrastive_loss.item())

            #         g_loss_final += (
            #             upperbeat_contrastive_loss 
            #             + facebeat_contrastive_loss
            #             + lowerbeat_contrastive_loss
            #         )

            #     # breakpoint()
            #     if len(seamless_indices) > 0:
                    
            #         with torch.no_grad():
            #             seamless_gt_upper_tokens = upper_codes[seamless_indices]

            #             # breakpoint()

            #             seamless_pred_upper_tokens = pred_upper_tokens[seamless_indices]
            #             seamless_pred_face_tokens = pred_face_tokens[seamless_indices]
            #             seamless_pred_lower_tokens = pred_lower_tokens[seamless_indices]

            #             seamless_pred_upper_q_latent = self.upper_gesture_codec.decode_latent(seamless_pred_upper_tokens) # B x T x latent_dim
            #             seamless_pred_face_q_latent = self.face_gesture_codec.decode_latent(seamless_pred_face_tokens) # B x T x latent_dim
            #             seamless_pred_lower_q_latent = self.lower_gesture_codec.decode_latent(seamless_pred_lower_tokens) # B x T x latent_dim
                        
            #             seamless_gt_upper_q_latent = self.upper_gesture_codec.decode_latent(seamless_gt_upper_tokens) # B x T x latent_dim
            #             seamless_gt_upper_q_latent = seamless_gt_upper_q_latent.detach()

            #         seamless_upper_cont_loss = self.contrastive_loss_func.compute_contrastive_loss(
            #             seamless_gt_upper_q_latent, seamless_pred_upper_q_latent, 
                        
            #         )
            #         seamless_upper_cont_loss = seamless_upper_cont_loss * lambda_contrastive
            #         self.tracker.update_meter("embody_upper_cont_loss", "train", seamless_upper_cont_loss.item())

            #         # sample lower_q_latent from gt_lower_q_latent as pseudo ground truth (handle non-equal batch size for seamless and beatx samples)
            #         if len(beatx_indices) > 0:
            #             with torch.no_grad():
            #                 pseudo_gt_lower_q_latent = beatx_gt_lower_q_latent[
            #                     torch.randint(0, beatx_gt_lower_q_latent.shape[0], (seamless_pred_lower_q_latent.shape[0],)) 
            #                 ]
            #                 pseudo_gt_lower_q_latent = pseudo_gt_lower_q_latent.detach()    
            #                 pseudo_gt_face_q_latent = beatx_gt_face_q_latent[
            #                     torch.randint(0, beatx_gt_face_q_latent.shape[0], (seamless_pred_face_q_latent.shape[0],))
            #                 ]
            #                 pseudo_gt_face_q_latent = pseudo_gt_face_q_latent.detach()

            #             seamless_lower_mmd_loss = self.contrastive_loss_func.compute_mmd(
            #                 seamless_pred_lower_q_latent, 
            #                 pseudo_gt_lower_q_latent
                            
            #             )
            #             seamless_lower_mmd_loss = seamless_lower_mmd_loss * lambda_contrastive
            #             self.tracker.update_meter("embody_lower_mmd_loss", "train", seamless_lower_mmd_loss.item())

            #             seamless_face_mmd_loss = self.contrastive_loss_func.compute_mmd(
            #                 seamless_pred_face_q_latent, 
            #                 pseudo_gt_face_q_latent
                            
            #             )
            #             seamless_face_mmd_loss = seamless_face_mmd_loss * lambda_contrastive
            #             self.tracker.update_meter("embody_face_mmd_loss", "train", seamless_face_mmd_loss.item())
            #         else:
            #             seamless_lower_mmd_loss = torch.tensor(0.0, device=self.local_rank)
            #             seamless_face_mmd_loss = torch.tensor(0.0, device=self.local_rank)

            #         g_loss_final += (
            #             seamless_upper_cont_loss 
            #             + seamless_face_mmd_loss
            #             + seamless_lower_mmd_loss
            #         )
            # # -------------------------------------------- #

            # ---------------- Constrastive Loss with Gumbel softmax ---------------- #
            if self.args.contrastive_loss_weight > 0 and epoch > self.args.pretrain_warmup_epochs:
                # Tau schedule
                tau0 = getattr(self.args, "gumbel_tau", 1.0)
                tau_min = getattr(self.args, "gumbel_tau_min", 0.4)  # 0.3–0.5 is typical
                anneal_epochs = getattr(self.args, "gumbel_tau_anneal_epochs", 5)
                prog = min(1.0, max(0.0, (epoch - self.args.pretrain_warmup_epochs) / max(1, anneal_epochs)))
                tau = tau0 - (tau0 - tau_min) * prog
                hard = True  # straight-through

                lambda_contrastive = (its / len(self.train_loader)) * self.args.contrastive_loss_weight if epoch - self.args.pretrain_warmup_epochs == 1 else self.args.contrastive_loss_weight
                if its % self.args.log_period == 0 and self.global_rank == 0:
                    logger.info(f"lambda_contrastive: {lambda_contrastive:.4f}, tau: {tau:.3f} at epoch {epoch}, iter {its}")

                upper_logits = logits[:, :self.upper_gesture_codec.num_codebooks, :, :]
                lower_logits = logits[:, self.upper_gesture_codec.num_codebooks:self.upper_gesture_codec.num_codebooks + self.lower_gesture_codec.num_codebooks, :, :]
                face_logits  = logits[:, self.upper_gesture_codec.num_codebooks + self.lower_gesture_codec.num_codebooks:, :, :]

                # differentiable predicted q-latents
                pred_upper_q_latent, _ = self._gumbel_logits_to_q_latent(upper_logits, self.upper_gesture_codec, self.modelout_ignore_index, tau=tau, hard=hard)
                pred_lower_q_latent, _ = self._gumbel_logits_to_q_latent(lower_logits, self.lower_gesture_codec, self.modelout_ignore_index, tau=tau, hard=hard)
                pred_face_q_latent,  _ = self._gumbel_logits_to_q_latent(face_logits,  self.face_gesture_codec,  self.modelout_ignore_index, tau=tau, hard=hard)

                if len(beatx_indices) > 0:
                    # GT latents (no gradients)
                    with torch.no_grad():
                        beatx_gt_upper_q_latent = self.upper_gesture_codec.decode_to_vqlatent(upper_codes[beatx_indices]).permute(0,2,1).detach()
                        beatx_gt_upper_q_latent = beatx_gt_upper_q_latent.to(pred_upper_q_latent.dtype)
                        beatx_gt_face_q_latent  = self.face_gesture_codec.decode_to_vqlatent(face_codes[beatx_indices]).permute(0,2,1).detach()
                        beatx_gt_face_q_latent = beatx_gt_face_q_latent.to(pred_face_q_latent.dtype)
                        beatx_gt_lower_q_latent = self.lower_gesture_codec.decode_to_vqlatent(lower_codes[beatx_indices]).permute(0,2,1).detach()
                        beatx_gt_lower_q_latent = beatx_gt_lower_q_latent.to(pred_lower_q_latent.dtype)

                    # slice predicted latents to BEATX rows
                    beatx_pred_upper_q_latent = pred_upper_q_latent[beatx_indices]
                    beatx_pred_face_q_latent  = pred_face_q_latent[beatx_indices]
                    beatx_pred_lower_q_latent = pred_lower_q_latent[beatx_indices]

                    # breakpoint()
                    upperbeat_contrastive_loss = self.contrastive_loss_func.compute_contrastive_loss(
                        beatx_gt_upper_q_latent, beatx_pred_upper_q_latent,
                    ) * lambda_contrastive
                    self.tracker.update_meter("cont_upper_loss", "train", upperbeat_contrastive_loss.item())

                    facebeat_contrastive_loss = self.contrastive_loss_func.compute_contrastive_loss(
                        beatx_gt_face_q_latent, beatx_pred_face_q_latent,
                    ) * lambda_contrastive
                    self.tracker.update_meter("cont_face_loss", "train", facebeat_contrastive_loss.item())

                    lowerbeat_contrastive_loss = self.contrastive_loss_func.compute_contrastive_loss(
                        beatx_gt_lower_q_latent, beatx_pred_lower_q_latent,
                    ) * lambda_contrastive
                    self.tracker.update_meter("cont_lower_loss", "train", lowerbeat_contrastive_loss.item())

                    g_loss_final += (
                        upperbeat_contrastive_loss 
                        + facebeat_contrastive_loss
                        + lowerbeat_contrastive_loss
                    )

                if len(seamless_indices) > 0:
                    # breakpoint()
                    with torch.no_grad():
                        seamless_gt_upper_q_latent = self.upper_gesture_codec.decode_to_vqlatent(upper_codes[seamless_indices]).permute(0,2,1).detach()
                        seamless_gt_upper_q_latent = seamless_gt_upper_q_latent.to(pred_upper_q_latent.dtype)

                    # slice predicted latents to Seamless rows
                    seamless_pred_upper_q_latent = pred_upper_q_latent[seamless_indices]
                    seamless_pred_face_q_latent  = pred_face_q_latent[seamless_indices]
                    seamless_pred_lower_q_latent = pred_lower_q_latent[seamless_indices]

                    seamless_upper_cont_loss = self.contrastive_loss_func.compute_contrastive_loss(
                        seamless_gt_upper_q_latent, seamless_pred_upper_q_latent,
                    ) * lambda_contrastive
                    self.tracker.update_meter("embody_upper_cont_loss", "train", seamless_upper_cont_loss.item())

                    if len(beatx_indices) > 0:
                        with torch.no_grad():
                            pseudo_gt_lower_q_latent = beatx_gt_lower_q_latent[
                                torch.randint(0, beatx_gt_lower_q_latent.shape[0], (seamless_pred_lower_q_latent.shape[0],), device=self.local_rank)
                            ].detach()
                            pseudo_gt_face_q_latent = beatx_gt_face_q_latent[
                                torch.randint(0, beatx_gt_face_q_latent.shape[0], (seamless_pred_face_q_latent.shape[0],), device=self.local_rank)
                            ].detach()
                        seamless_lower_mmd_loss = self.contrastive_loss_func.compute_mmd(
                            seamless_pred_lower_q_latent, pseudo_gt_lower_q_latent
                        ) * lambda_contrastive * 0.1 # downweight MMD loss compared to contrastive loss
                        self.tracker.update_meter("embody_lower_mmd_loss", "train", seamless_lower_mmd_loss.item())

                        seamless_face_mmd_loss = self.contrastive_loss_func.compute_mmd(
                            seamless_pred_face_q_latent, pseudo_gt_face_q_latent
                        ) * lambda_contrastive * 0.1 # downweight MMD loss compared to contrastive loss
                        self.tracker.update_meter("embody_face_mmd_loss", "train", seamless_face_mmd_loss.item())
                    else:
                        seamless_lower_mmd_loss = torch.tensor(0.0, device=self.local_rank)
                        seamless_face_mmd_loss = torch.tensor(0.0, device=self.local_rank)

                    g_loss_final += (
                        seamless_upper_cont_loss 
                        + seamless_face_mmd_loss
                        + seamless_lower_mmd_loss
                    )
                
            # ---------------- Gen reconstruction loss ---------------- #
            if self.args.genrecon_loss_weight > 0:

                # Tau schedule
                tau0 = getattr(self.args, "gumbel_tau", 1.0)
                tau_min = getattr(self.args, "gumbel_tau_min", 0.4)  # 0.3–0.5 is typical
                anneal_epochs = getattr(self.args, "gumbel_tau_anneal_epochs", 5)
                prog = min(1.0, max(0.0, (epoch - self.args.pretrain_warmup_epochs) / max(1, anneal_epochs)))
                tau = tau0 - (tau0 - tau_min) * prog
                hard = True  # straight-through

                assert self.args.contrastive_loss_weight == 0, "Only one of genrecon_loss_weight and contrastive_loss_weight can be non-zero"
                lambda_genrecon = self.args.genrecon_loss_weight
                if its % self.args.log_period == 0 and lambda_genrecon < self.args.genrecon_loss_weight and self.global_rank == 0:
                        # breakpoint()
                        logger.info(f"lambda_genrecon: {lambda_genrecon:.4f} at epoch {epoch}, iter {its}")

                # breakpoint() # check slicing
                upper_logits = logits[:, :self.upper_gesture_codec.num_codebooks, :, :]
                lower_logits = logits[:, self.upper_gesture_codec.num_codebooks:self.upper_gesture_codec.num_codebooks + self.lower_gesture_codec.num_codebooks, :, :]
                face_logits  = logits[:, self.upper_gesture_codec.num_codebooks + self.lower_gesture_codec.num_codebooks:, :, :]

                # differentiable predicted q-latents
                pred_upper_q_latent, _ = self._gumbel_logits_to_q_latent(upper_logits, self.upper_gesture_codec, self.modelout_ignore_index, tau=tau, hard=hard)
                pred_lower_q_latent, _ = self._gumbel_logits_to_q_latent(lower_logits, self.lower_gesture_codec, self.modelout_ignore_index, tau=tau, hard=hard)
                pred_face_q_latent,  _ = self._gumbel_logits_to_q_latent(face_logits,  self.face_gesture_codec,  self.modelout_ignore_index, tau=tau, hard=hard)

                if len(beatx_indices) > 0:
                    # GT latents (no gradients)
                    with torch.no_grad():
                        beatx_gt_upper_q_latent = self.upper_gesture_codec.decode_to_vqlatent(upper_codes[beatx_indices]).permute(0,2,1).detach()
                        beatx_gt_upper_q_latent = beatx_gt_upper_q_latent.to(pred_upper_q_latent.dtype)
                        beatx_gt_face_q_latent  = self.face_gesture_codec.decode_to_vqlatent(face_codes[beatx_indices]).permute(0,2,1).detach()
                        beatx_gt_face_q_latent = beatx_gt_face_q_latent.to(pred_face_q_latent.dtype)
                        beatx_gt_lower_q_latent = self.lower_gesture_codec.decode_to_vqlatent(lower_codes[beatx_indices]).permute(0,2,1).detach()
                        beatx_gt_lower_q_latent = beatx_gt_lower_q_latent.to(pred_lower_q_latent.dtype)

                    # slice predicted latents to BEATX rows
                    beatx_pred_upper_q_latent = pred_upper_q_latent[beatx_indices]
                    beatx_pred_face_q_latent  = pred_face_q_latent[beatx_indices]
                    beatx_pred_lower_q_latent = pred_lower_q_latent[beatx_indices]
                    
                    upperbeat_genrecon_loss = F.mse_loss(
                        beatx_pred_upper_q_latent, beatx_gt_upper_q_latent
                    )
                    upperbeat_genrecon_loss = upperbeat_genrecon_loss * lambda_genrecon
                    self.tracker.update_meter("mse_upper_loss", "train", upperbeat_genrecon_loss.item())

                    facebeat_genrecon_loss = F.mse_loss(
                        beatx_pred_face_q_latent, beatx_gt_face_q_latent
                    ) 
                    facebeat_genrecon_loss = facebeat_genrecon_loss * lambda_genrecon
                    self.tracker.update_meter("mse_face_loss", "train", facebeat_genrecon_loss.item())

                    
                    lowerbeat_genrecon_loss = F.mse_loss(
                        beatx_pred_lower_q_latent, beatx_gt_lower_q_latent
                    )
                    lowerbeat_genrecon_loss = lowerbeat_genrecon_loss * lambda_genrecon
                    self.tracker.update_meter("mse_lower_loss", "train", lowerbeat_genrecon_loss.item())

                    g_loss_final += (
                        upperbeat_genrecon_loss
                        + facebeat_genrecon_loss
                        + lowerbeat_genrecon_loss
                    )

                if len(seamless_indices) > 0:
                    with torch.no_grad():
                        seamless_gt_upper_q_latent = self.upper_gesture_codec.decode_latent(upper_codes[seamless_indices]).detach()

                    # slice predicted latents to Seamless rows
                    seamless_pred_upper_q_latent = pred_upper_q_latent[seamless_indices]
                    seamless_pred_face_q_latent  = pred_face_q_latent[seamless_indices]
                    seamless_pred_lower_q_latent = pred_lower_q_latent[seamless_indices]

                    seamless_upper_mse_loss = F.mse_loss(
                        seamless_pred_upper_q_latent, seamless_gt_upper_q_latent
                    )
                    seamless_upper_mse_loss = seamless_upper_mse_loss * lambda_genrecon
                    self.tracker.update_meter("embody_mse_upper_loss", "train", seamless_upper_mse_loss.item())

                    g_loss_final += (
                        seamless_upper_mse_loss 
                    )

            # -------------------------------------------- #
            
            g_loss_final.backward()

            # for name, param in self.model.named_parameters():
            #     grad = param.grad
            #     if grad is None:
            #         print(f"[rank {self.local_rank}] param {name} grad is None")
            #     else:
            #         pass
            #         # print(f"[rank {self.local_rank}] param {name}: shape={tuple(g.shape)}, dtype={g.dtype}, device={g.device}")

            # breakpoint() # check dtype of model parameters
            upcast_mixed_precision(self.model.parameters(), optim_dtype=self.optim_dtype)
            
            if self.args.grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.opt.step()
            if self.args.lr_policy == "onecyclelr":
                self.opt_s.step()
            downcast_mixed_precision(self.model.parameters(), param_dtype=self.param_dtype)
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            if its % self.args.log_period == 0: # and self.local_rank == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g) 

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
        if self.args.lr_policy != "onecyclelr":
            self.opt_s.step(epoch)
                    
    def val(self, epoch):
        if epoch < self.args.pretrain_warmup_epochs:
            return 0
        
        gc.collect()
        self.upper_gesture_codec.eval()
        self.face_gesture_codec.eval()
        self.lower_gesture_codec.eval()
        self.model.eval()
        t_start = time.time()
        # self.gesture_metrics.reset()
        # print(len(self.val_loader))
        with torch.no_grad():
            for its, dict_data in enumerate(self.val_loader):
                # if its == 1:
                #     break
                # print("val iter", its) 

                tar_pose_upper = dict_data["motion_upper"]
                tar_pose_hands = dict_data["motion_hands"]
                tar_pose_face = dict_data["motion_face"]

                tar_pose_lower = dict_data["motion_lower"]
                tar_trans = dict_data["transl"].to(self.local_rank)
                tar_contact = dict_data["contact"].to(self.local_rank)
                tar_exps = dict_data["expressions"].to(self.local_rank)

                tar_audio_tokens = dict_data["audio_tokens"]
                tar_text_tokens = dict_data["text_tokens"]
                tar_spk = dict_data["speaker_id"] 

                lower_valid_mask = dict_data["lower_valid_mask"].to(self.local_rank) 
                assert lower_valid_mask is not None, "Lower valid mask should not be None in training"

                dataset_name_per_item = dict_data['dataset_name']
                dataset_name_per_item = np.array(dataset_name_per_item)
                # get indices for seamless interaction and beatx
                # seamless_indices = np.where(dataset_name_per_item != 'BEATX')[0]
                # beatx_indices = np.where(dataset_name_per_item == 'BEATX')[0]
                seamless_indices = torch.where(lower_valid_mask == 0)[0].to(self.local_rank)
                beatx_indices = torch.where(lower_valid_mask == 1)[0].to(self.local_rank)
                # breakpoint()

                
                # subtract the first frame translation
                tar_trans[:, :, 0] = tar_trans[:, :, 0] - tar_trans[:, 0:1, 0]
                tar_trans[:, :, 2] = tar_trans[:, :, 2] - tar_trans[:, 0:1, 2]

                # upper_hands_joint_mask = self.train_data.upper_mask_for_flattened + \
                #     self.train_data.hands_mask_for_flattened
                # lower_joint_mask = self.train_data.lower_mask_for_flattened 
                # upper_lower_joint_mask = upper_hands_joint_mask + lower_joint_mask

                # -- lower motion -- #
                tar_trans_vel = other_tools.estimate_linear_velocity(
                    tar_trans, dt=1/self.args.motion_fps
                )
                tar_pose_lower = tar_pose_lower.to(self.local_rank)
                
                # breakpoint()
                bs, n, ldim = tar_pose_lower.shape
                lj = ldim //3
                tar_pose_lower = rc.axis_angle_to_matrix(tar_pose_lower.reshape(bs, n, lj, 3))
                tar_pose_lower = rc.matrix_to_rotation_6d(tar_pose_lower).reshape(bs, n, lj*6)

                # -- upper motion -- #
                tar_pose_upper = tar_pose_upper.to(self.local_rank)
                tar_pose_hands = tar_pose_hands.to(self.local_rank)
                

                bs, n, udim = tar_pose_upper.shape
                uj = udim //3
                tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, uj, 3))
                tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, uj*6)

                hj = tar_pose_hands.shape[-1]
                hj = hj // 3
                tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, hj, 3))
                tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, hj*6)

                # -- face motion -- #
                tar_pose_face = tar_pose_face.to(self.local_rank)
                bs, n, fdim = tar_pose_face.shape
                fj = fdim //3
                tar_pose_face = rc.axis_angle_to_matrix(tar_pose_face.reshape(bs, n, fj, 3))
                tar_pose_face = rc.matrix_to_rotation_6d(tar_pose_face).reshape(bs, n, fj*6)

                
                j = uj + hj + fj + lj
                
                tar_audio_tokens = tar_audio_tokens.to(self.local_rank)
                tar_text_tokens = tar_text_tokens.to(self.local_rank)
                tar_spk = tar_spk.to(self.local_rank)

                t_data = time.time() - t_start
                # breakpoint()

                audio_codes = tar_audio_tokens # B x K=8 x T=125
                text_codes = tar_text_tokens 
                
                
                # breakpoint()
                in_tar_pose_upper = torch.cat((tar_pose_upper, tar_pose_hands), dim=-1)
                in_tar_pose_lower = torch.cat((tar_pose_lower, tar_trans_vel, tar_contact), dim=-1)
                in_tar_pose_face = torch.cat((tar_pose_face, tar_exps), dim=-1) # 106
                
                upper_codes = self.upper_gesture_codec.encode(in_tar_pose_upper) # B x K=8 x T=25
                lower_codes = self.lower_gesture_codec.encode(in_tar_pose_lower) # B x K=8 x T=25
                face_codes = self.face_gesture_codec.encode(in_tar_pose_face) # B x K=4 x T=25
                lower_codes[seamless_indices] = self.modelout_ignore_index # set to pad token for seamless interaction samples
                face_codes[seamless_indices] = self.modelout_ignore_index # set to pad token for seamless interaction samples
                
                conditions = (audio_codes, text_codes) 
                
                
                gesture_tokens = torch.cat((upper_codes, lower_codes, face_codes), dim=1) # B x K=16 x T=25

                # breakpoint() # check the mask
                lower_cross_attn_mask = torch.zeros_like(gesture_tokens).to(torch.bool) # B x K=20 x T
                lower_cross_attn_mask[
                    :, 
                    self.upper_gesture_codec.num_codebooks : self.upper_gesture_codec.num_codebooks + self.lower_gesture_codec.num_codebooks, 
                    :] = True # only apply cross-attention on upper and face tokens
                # flatten batches adn time in forward pass
                in_gesture_tokens = gesture_tokens
                model_out = self.model(
                    in_gesture_tokens, 
                    audio_codes=audio_codes,
                    text_codes=text_codes, 
                    sum_condition=tar_spk,
                    ca_depth_padding_mask=lower_cross_attn_mask if self.args.drop_lower_crossattn else None,
                    ) # B x K=16 x T=25 x cardinality
                
                # TODO: Change loss
                logits = model_out #.logits

                ce_loss = 0
                B, K, T, card = logits.shape
                upper_loss = 0
                face_loss = 0
                lower_loss = 0
                
                pad_mask = (gesture_tokens != self.modelout_ignore_index).float() # B x K x T
                for k in range(K):
                    loss_k = F.cross_entropy(
                        logits[:, k].reshape(B*T, card),
                        gesture_tokens[:, k].reshape(B*T), 
                        reduction="none"
                    )
                    # breakpoint() # check pad_mask
                    loss_k = loss_k * pad_mask[:, k].reshape(B*T) 
                    loss = loss_k.sum() / (pad_mask[:, k].sum() + 1e-12)
                    loss_k = loss * (1/K)

                    if k < 8:  # upper body + hands
                        upper_loss += loss_k.item()
                    elif k >=8 and k < 16:  # face + expressions
                        lower_loss += loss_k.item()
                    else:  # lower body
                        face_loss += loss_k.item()
                    
                    ce_loss += loss_k

                ce_loss = ce_loss #/ K
                upper_loss = upper_loss #/ (K/2)
                face_loss = face_loss #/ (K/4)
                lower_loss = lower_loss #/ (K/2)

                # g_loss_final += ce_loss
                self.tracker.update_meter("ce_loss", "val", ce_loss.item())
                self.tracker.update_meter("ce_upper", "val", upper_loss)
                self.tracker.update_meter("ce_face", "val", face_loss)
                self.tracker.update_meter("ce_lower", "val", lower_loss)
                
                perplexity = self.calculate_perplexity(
                    logits, 
                    gesture_tokens, 
                    pad_token_id=self.modelout_ignore_index
                )
                self.tracker.update_meter("perplexity", "val", perplexity)

                if self.args.debug:
                    if its == 10: break
                
        if self.global_rank == 0:
            self.val_recording(epoch)
            # self.store_inputs_outputs(os.path.join(self.checkpoint_path, "visualizations/"))
            
            
    def test(self, epoch, visualize=False, max_batches=None, save=False):
        results_save_path = os.path.join(self.checkpoint_path,  f"{epoch}/") 
        if os.path.exists(results_save_path) and self.args.is_train:
            return 0
        
        if visualize:
            import soundfile as sf
            import shutil
            import tempfile
            # All renderer helpers come from the new dataloaders/utils path
            # now (matching run_gestinference.py). render_smplx_debug_video
            # + stitch_videos_hstack + mux_audio_into_video are already
            # imported at the top of this module.

            assert self.test_loader.batch_size == 1, "Visualization currently only supports batch size of 1 for easier handling of file names and outputs"


        

        
        if not os.path.exists(results_save_path):
            os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0

        causal = self.model.causal
        # breakpoint()

        gc.collect()
        self.upper_gesture_codec.eval()
        self.lower_gesture_codec.eval()
        self.model.eval()
        self.gesture_metrics.reset()

        gesture_lm_config = {
            "use_sampling": True,
            "temp_gtemporal": 0.9,
            "temp_gdepth": 0.9,
            "top_p_gtemporal": 0.8,
            "top_p_gdepth": 0.95,
            "check": True,
        }
        # breakpoint()

        
        
        if causal:
            logger.info(f"Using causal model sampling")
        else:
            raise NotImplementedError("Non-causal model sampling is not implemented yet")
            
        logger.info(f"Length of test set: {len(self.test_data)} samples in {len(self.test_loader)} batches")
        
        with torch.no_grad():
            for its, dict_data in enumerate(self.test_loader):
                if max_batches is not None and its >= max_batches:
                    break

                # if its == int(0.25 * len(self.test_loader)):
                #     break

                tar_spk = dict_data["speaker_id"] 
                sample_names = dict_data["filechunk_id"]
                file_name = dict_data["file_id"]
                # breakpoint()
                # if "_1_" not in sample_names[0]:
                #     continue
                # if "C0" not in sample_names[0]:
                #     continue
                # if "lawrence" not in file_name[0]:
                #     continue
                
                # breakpoint()
                gesture_spk_condition = torch.full((1, 1), tar_spk[0], device=self.local_rank, dtype=torch.long)
                # breakpoint() # check the flowgen settings
                glmgen = GestureLMGen(
                    self.model,
                    condition_tensors=gesture_spk_condition, 
                    cfg_coef=self.args.cfg_scale, #2.3,
                    **gesture_lm_config
                )
                # breakpoint()

                

                tar_pose_upper = dict_data["motion_upper"]
                tar_pose_face = dict_data["motion_face"]
                tar_pose_hands = dict_data["motion_hands"]

                tar_pose_lower = dict_data["motion_lower"]
                tar_trans = dict_data["transl"].to(self.local_rank)
                tar_contact = dict_data["contact"].to(self.local_rank)
                tar_exps = dict_data["expressions"].to(self.local_rank)

                tar_audio_tokens = dict_data["audio_tokens"]
                tar_text_tokens = dict_data["text_tokens"]
                tar_beta = dict_data["beta"].to(self.local_rank)

                tar_raw_text = dict_data["raw_text"]
                tar_raw_audio = dict_data["raw_audio"]

                
                lower_valid_mask = dict_data["lower_valid_mask"].to(self.local_rank)
                assert lower_valid_mask is not None, "Lower valid mask should not be None in training"

                # With --dataset_ratio *_fulllength and the full-sequence eval
                # HDF5, the loader yields whole clips whose frame count is not
                # guaranteed to be a multiple of self.args.frame_chunk_size,
                # but the gesture codec's SEANet encoder asserts
                # `T % stride == 0` (conv.py:247). Training used fixed
                # `pose_length=250` chunks (always divisible) so the codec
                # trainers handle this with a one-liner truncation; mirror
                # that here before any reshape touches `n`.
                # breakpoint() # check the shape before truncation
                remain = tar_pose_upper.shape[1] % self.args.frame_chunk_size
                if remain:
                    n_trunc = tar_pose_upper.shape[1] - remain
                    tar_pose_upper = tar_pose_upper[:, :n_trunc]
                    tar_pose_hands = tar_pose_hands[:, :n_trunc]
                    tar_pose_face = tar_pose_face[:, :n_trunc]
                    tar_pose_lower = tar_pose_lower[:, :n_trunc]
                    tar_trans = tar_trans[:, :n_trunc]
                    tar_contact = tar_contact[:, :n_trunc]
                    tar_exps = tar_exps[:, :n_trunc]
                    tar_beta = tar_beta[:, :n_trunc]
                    lower_valid_mask = lower_valid_mask[:n_trunc]

                # subtract the first frame translation
                tar_trans[:, :, 0] = tar_trans[:, :, 0] - tar_trans[:, 0:1, 0]
                tar_trans[:, :, 2] = tar_trans[:, :, 2] - tar_trans[:, 0:1, 2]

                upper_hands_joint_mask = self.test_data.upper_mask_for_flattened + \
                    self.test_data.hands_mask_for_flattened
                lower_joint_mask = self.test_data.lower_mask_for_flattened 
                face_joint_mask = self.test_data.face_mask_for_flattened
                # upper_lower_joint_mask = upper_hands_joint_mask + lower_joint_mask

                # -- lower motion -- #
                tar_trans_vel = other_tools.estimate_linear_velocity(
                    tar_trans, dt=1/self.args.motion_fps, 
                )
                tar_pose_lower = tar_pose_lower.to(self.local_rank)
                tar_pose_lower_aa = tar_pose_lower.clone()
                # # breakpoint()
                bs, n, ldim = tar_pose_lower.shape
                lj = ldim //3 
                tar_pose_lower = rc.axis_angle_to_matrix(tar_pose_lower.reshape(bs, n, lj, 3))
                tar_pose_lower = rc.matrix_to_rotation_6d(tar_pose_lower).reshape(bs, n, lj*6)

                # # -- upper motion -- #
                
                tar_pose_upper = tar_pose_upper.to(self.local_rank)
                tar_pose_hands = tar_pose_hands.to(self.local_rank)
                tar_pose_upperhands_aa = torch.cat((tar_pose_upper, tar_pose_hands), dim=-1)
                

                bs, n, udim = tar_pose_upper.shape
                uj = udim //3
                tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, uj, 3))
                tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, uj*6)

                hj = tar_pose_hands.shape[-1]
                hj = hj // 3
                tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, hj, 3))
                tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, hj*6)
                uhj = uj + hj

                # -- face motion -- #
                tar_exps = tar_exps.to(self.local_rank)
                tar_pose_face = tar_pose_face.to(self.local_rank)
                tar_pose_face_aa = tar_pose_face.clone()
                bs, n, fdim = tar_pose_face.shape
                fj = fdim //3
                tar_pose_face = rc.axis_angle_to_matrix(tar_pose_face.reshape(bs, n, fj, 3))
                tar_pose_face = rc.matrix_to_rotation_6d(tar_pose_face).reshape(bs, n, fj*6)


                
                tar_audio_tokens = tar_audio_tokens.to(self.local_rank)
                tar_text_tokens = tar_text_tokens.to(self.local_rank)
                tar_spk = tar_spk.to(self.local_rank)

                # breakpoint()
                audio_codes = tar_audio_tokens # B x K=8 x T=125
                text_codes = tar_text_tokens # B x K=1 x T=125

                audio_text_codes = torch.cat((text_codes, audio_codes), dim=1) # B x K=9 x T=125


                in_tar_pose_upper = torch.cat((tar_pose_upper, tar_pose_hands), dim=-1)
                in_tar_pose_face = torch.cat((tar_pose_face, tar_exps), dim=-1) # 106
                in_tar_pose_lower = torch.cat((tar_pose_lower, tar_trans_vel, tar_contact), dim=-1)
                
                upper_codes_gt = self.upper_gesture_codec.encode(in_tar_pose_upper) # B x K=8 x T=25
                # CLAUDE REMOVE START --- IGNORE ---
                lower_codes_gt = self.lower_gesture_codec.encode(in_tar_pose_lower) # B x K=8 x T=25
                face_codes_gt = self.face_gesture_codec.encode(in_tar_pose_face) # B x K=4 x T=25
                # CLAUDE REMOVE END --- IGNORE ---
                
                lower_cross_attn_mask = torch.zeros(
                    bs, 
                    self.upper_gesture_codec.num_codebooks + self.lower_gesture_codec.num_codebooks + self.face_gesture_codec.num_codebooks , 
                    1, 
                    device=audio_codes.device
                ).to(torch.bool) # B x K=20 x T
                # lower_cross_attn_mask[:, -self.lower_gesture_codec.num_codebooks:, :] = True 

                # breakpoint()
                lower_cross_attn_mask[
                    :, 
                    self.upper_gesture_codec.num_codebooks : self.upper_gesture_codec.num_codebooks + self.lower_gesture_codec.num_codebooks:, 
                    :] = True # only apply cross-attention on upper and face tokens
    
                
    
                
                # breakpoint() # check the shapes of upperhands, lower pos both rec and tar
                if not "fulllength" in self.args.dataset_ratio:
                    remain = n%self.args.pose_length
                    tar_pose_upperhands_aa = tar_pose_upperhands_aa[:, :n-remain, :]
                    tar_pose_lower = tar_pose_lower[:, :n-remain, :]
                    tar_trans = tar_trans[:, :n-remain, :]
                # tar_trans_vel = tar_trans_vel[:, :n-remain, :]
                # breakpoint()
                final_pos = tar_trans[:, 0]

                framechunk_size = self.args.frame_chunk_size
                num_frames = int((audio_text_codes.shape[2] / self.model.query2mem_scale) * framechunk_size)
                # breakpoint()
                with self.upper_gesture_codec.streaming(batch_size=bs) as sg_codec, \
                    self.face_gesture_codec.streaming(batch_size=bs) as sf_codec, \
                    self.lower_gesture_codec.streaming(batch_size=bs) as sl_codec:

                    out_upperhands = None
                    out_face_motion = None
                    out_exps = None
                    out_lower = None
                    out_trans = None
                    out_uppertokens = None
                    
                    # num_frames = int((audio_text_codes.shape[2] / self.model.query2mem_scale) * framechunk_size)
                    # breakpoint()               
                    if "fulllength" in self.args.dataset_ratio and num_frames < tar_pose_upperhands_aa.shape[1]:
                        # breakpoint()
                        tar_pose_upperhands_aa = tar_pose_upperhands_aa[:, :num_frames, :]
                        tar_pose_lower_aa = tar_pose_lower_aa[:, :num_frames, :]
                        tar_pose_face_aa = tar_pose_face_aa[:, :num_frames, :]
                        tar_trans = tar_trans[:, :num_frames, :]
                        tar_exps = tar_exps[:, :num_frames, :]
                        tar_beta = tar_beta[:, :num_frames, :]

                    assert num_frames == tar_pose_upperhands_aa.shape[1] == tar_pose_lower_aa.shape[1] == tar_trans.shape[1]
                    audio_num_frames = audio_text_codes.shape[2]
                    # self.upper_gesture_vae.reset_streaming()
                    # self.lower_gesture_vae.reset_streaming()
                    # flowgen.reset_streaming()
                    step_time_average = 0
                    with glmgen.streaming(batch_size=bs) as sflowgen:
                        # with self.model.streaming(batch_size=audio_codes.shape[0]):
                        tic = time.time()
                        for chunk_start in tqdm(range(0, audio_num_frames, self.model.query2mem_scale), desc="Generating gesture chunks"):
                            chunk_end = min(chunk_start + self.model.query2mem_scale, audio_num_frames)
                            audiotext_chunk = audio_text_codes[:, :, chunk_start:chunk_end]
                            # breakpoint() # check the chunking of audio and text codes # cheeck self.gesture_codecs[0].num_codebooks
                            chunk_len = chunk_end - chunk_start

                            
                            # lower_cross_attn_mask = \
                            #     torch.ones(bs, self.lower_gesture_codec.num_codebooks ,1, device=audiotext_chunk.device).to(torch.bool) # B x K=16 x T
                            # # lower_cross_attn_mask[:, self.upper_gesture_codec.num_codebooks:, :] = True 
                            # # breakpoint()
                            step_tic = time.time()
                            gmoshi_tokens = glmgen.step(
                                condition=audiotext_chunk,
                                ca_query_padding_mask=lower_cross_attn_mask if self.args.drop_lower_crossattn else None,
                            )
                            assert gmoshi_tokens.shape[-1] == 1, "Only support single clip generation"
                            # breakpoint()

                            # upper_tokens = gmoshi_tokens[:, :self.upper_gesture_codec.num_codebooks]
                            # face_tokens = gmoshi_tokens[:, self.upper_gesture_codec.num_codebooks:self.upper_gesture_codec.num_codebooks + self.face_gesture_codec.num_codebooks]
                            # lower_tokens = gmoshi_tokens[:, -self.lower_gesture_codec.num_codebooks:]

                            upper_tokens = gmoshi_tokens[:, :self.upper_gesture_codec.num_codebooks, :] 
                            lower_tokens = gmoshi_tokens[:, self.upper_gesture_codec.num_codebooks:self.upper_gesture_codec.num_codebooks + self.lower_gesture_codec.num_codebooks, :] 
                            face_tokens = gmoshi_tokens[:, self.upper_gesture_codec.num_codebooks + self.lower_gesture_codec.num_codebooks:, :] 
                            



            
                            
                            # breakpoint()
                            assert (upper_tokens != self.modelout_ignore_index).all(), "Generated upper tokens contain pad token!"
                            assert (lower_tokens != self.modelout_ignore_index).all(), "Generated lower tokens contain pad token!"
                            assert (face_tokens != self.modelout_ignore_index).all(), "Generated face tokens contain pad token!"

                            # breakpoint() # check the generated tokens
                            # out_uppertokens = upper_tokens if out_uppertokens is None else torch.cat((out_uppertokens, upper_tokens), dim=2)

                            upper_motion = self.upper_gesture_codec.decode(upper_tokens)
                            face_motion = self.face_gesture_codec.decode(face_tokens)
                            lowertrans_motion = self.lower_gesture_codec.decode(lower_tokens)
                            # TEMPORARY: set upper motion to zeros
                            # upper_motion = torch.zeros(bs, framechunk_size, uhj*6, device=lowertrans_motion.device)

                            

                            rec_pose_upperhands = upper_motion
                            rec_pose_face = face_motion[:, :, :6]  
                            rec_exps = face_motion[:, :, 6:]
                            rec_pose_lower = lowertrans_motion[:, :, :lj*6]
                            # rec_trans = lowertrans_motion[:, :, lj*6:lj*6+3]
                            rec_trans_vel = lowertrans_motion[:, :, lj*6:lj*6+3]
                            # breakpoint()

                            assert rec_pose_upperhands.shape[1] == rec_pose_lower.shape[1] == rec_pose_face.shape[1] == framechunk_size
                            n = rec_pose_lower.shape[1]

                            rec_pose_upperhands = rec_pose_upperhands.reshape(bs, n, uhj, 6)
                            rec_pose_face = rec_pose_face.reshape(bs, n, fj, 6)
                            rec_pose_lower = rec_pose_lower.reshape(bs, n, lj, 6)

                            rec_pose_upperhands = rc.rotation_6d_to_matrix(rec_pose_upperhands)#
                            rec_pose_upperhands = rc.matrix_to_axis_angle(rec_pose_upperhands).reshape(bs, n, uhj*3)

                            rec_pose_face = rc.rotation_6d_to_matrix(rec_pose_face)#
                            rec_pose_face = rc.matrix_to_axis_angle(rec_pose_face).reshape(bs, n, fj*3)
                            # TEMPORARY:
                            # lower_motion = torch.zeros_like(lower_motion)
                            # rec_pose_upperhands = torch.zeros_like(rec_pose_upperhands)
                            # motion_trans = torch.zeros_like(motion_trans) 

                            rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)#
                            rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs, n, lj*3)

                            rec_trans, final_pos = other_tools.velocity2position_mixeddiff(
                                rec_trans_vel, 1/self.args.motion_fps, init_pos=final_pos
                            )
                            # breakpoint() # check shapes of all recs

                            if chunk_start != 0:
                                out_upperhands = torch.cat((out_upperhands, rec_pose_upperhands), dim=1)
                                out_face_motion = torch.cat((out_face_motion, rec_pose_face), dim=1)
                                out_exps = torch.cat((out_exps, rec_exps), dim=1)
                                out_lower = torch.cat((out_lower, rec_pose_lower), dim=1)
                                out_trans = torch.cat((out_trans, rec_trans), dim=1)
                            else:
                                out_upperhands = rec_pose_upperhands
                                out_face_motion = rec_pose_face
                                out_exps = rec_exps
                                out_lower = rec_pose_lower
                                out_trans = rec_trans

                            step_toc = time.time()
                            step_time_average += (step_toc - step_tic)

                        toc = time.time()
                        logger.info(f"Sampling time for {num_frames/self.args.motion_fps} sec of motion: {toc - tic} sec")
                        logger.info(f"Average step time: {step_time_average / (audio_num_frames / self.model.query2mem_scale)} sec")


                rec_pose_upperhands = out_upperhands
                rec_pose_face = out_face_motion
                rec_exps = out_exps
                rec_pose_lower = out_lower
                rec_trans = out_trans
                assert num_frames == rec_pose_upperhands.shape[1]
                assert num_frames == rec_exps.shape[1]
                assert num_frames == rec_pose_face.shape[1]
                assert num_frames == rec_pose_lower.shape[1]
                assert num_frames == rec_trans.shape[1]

                # breakpoint() # check the shapes of upperhands, lower pos both rec and tar

                rec_pose_upperhands = rec_pose_upperhands.reshape(bs*num_frames, uhj*3)
                rec_pose_face = rec_pose_face.reshape(bs*num_frames, fj*3)
                rec_pose_lower = rec_pose_lower.reshape(bs*num_frames, lj*3)
                rec_trans = rec_trans.reshape(bs*num_frames, 3)
                rec_exps = rec_exps.reshape(bs*num_frames, -1)

                # breakpoint()
                tar_pose_upperhands = tar_pose_upperhands_aa.reshape(bs*num_frames, uhj*3)
                tar_pose_face = tar_pose_face_aa.reshape(bs*num_frames, fj*3)
                tar_pose_lower = tar_pose_lower_aa.reshape(bs*num_frames, lj*3)
                tar_trans = tar_trans.reshape(bs*num_frames, 3)
                tar_exps = tar_exps.reshape(bs*num_frames, -1)

                tar_pose_upperhands = self.test_data.inverse_selection_tensor(
                    tar_pose_upperhands, 
                    upper_hands_joint_mask, 
                    tar_pose_upperhands.shape[0]
                )
                rec_pose_upperhands = self.test_data.inverse_selection_tensor(
                    rec_pose_upperhands, 
                    upper_hands_joint_mask, 
                    rec_pose_upperhands.shape[0]
                )
                tar_pose_face = self.test_data.inverse_selection_tensor(
                    tar_pose_face, 
                    face_joint_mask, 
                    tar_pose_face.shape[0]
                )
                rec_pose_face = self.test_data.inverse_selection_tensor(
                    rec_pose_face, 
                    face_joint_mask, 
                    rec_pose_face.shape[0]
                )
                tar_pose_lower = self.test_data.inverse_selection_tensor(
                    tar_pose_lower, 
                    lower_joint_mask, 
                    tar_pose_lower.shape[0]
                )
                rec_pose_lower = self.test_data.inverse_selection_tensor(
                    rec_pose_lower, 
                    lower_joint_mask, 
                    rec_pose_lower.shape[0]
                )
                
                # breakpoint() 
                tar_pose = tar_pose_upperhands + tar_pose_face + tar_pose_lower
                rec_pose = rec_pose_upperhands + rec_pose_face + rec_pose_lower
                # rec_trans = rec_trans - rec_trans
                # tar_trans = tar_trans - tar_trans

                # breakpoint() # check the shapes
                tar_beta = tar_beta.reshape(bs*num_frames, -1)
                # breakpoint()

                

                # Per-batch-item metric update. All tensors at this point
                # are shape `(bs * num_frames, ...)`; we slice one sample
                # at a time so this loop is correct for bs > 1.
                for b in range(bs):
                    sl = slice(b * num_frames, (b + 1) * num_frames)
                    metric_dict = {
                        "rec_pose": rec_pose[sl],
                        "rec_exps": rec_exps[sl],
                        "rec_trans": rec_trans[sl],
                        "tar_pose": tar_pose[sl],
                        "tar_exps": tar_exps[sl],
                        "tar_beta": tar_beta[sl][0],
                        "tar_trans": tar_trans[sl],
                        "file_id": file_name[b],
                    }
                    self.gesture_metrics.update(metric_dict)
                

                rec_pose = rec_pose.cpu().numpy()
                tar_pose = tar_pose.cpu().numpy()
                tar_trans = tar_trans.cpu().numpy()
                rec_trans = rec_trans.cpu().numpy()
                tar_exps = tar_exps.cpu().numpy()
                rec_exps = rec_exps.cpu().numpy()
                tar_beta = tar_beta.cpu().numpy()
                # breakpoint()
                

                total_length += rec_pose.shape[0]

                # --- save + (optional) visualize per batch item --- #
                # All numpy arrays here are shape `(bs * num_frames, ...)`;
                # slice one sample at a time so this is correct for bs > 1.
                njoints = len(self.test_data.smplx_joint_names)
                upper_codes_gt_np = upper_codes_gt.cpu().numpy()
                for b in range(bs):
                    sl = slice(b * num_frames, (b + 1) * num_frames)
                    rec_pose_b = rec_pose[sl].reshape(num_frames, njoints, 3)
                    tar_pose_b = tar_pose[sl].reshape(num_frames, njoints, 3)
                    rec_trans_b = rec_trans[sl].reshape(num_frames, 3)
                    tar_trans_b = tar_trans[sl].reshape(num_frames, 3)
                    rec_exps_b = rec_exps[sl].reshape(num_frames, -1)
                    tar_exps_b = tar_exps[sl].reshape(num_frames, -1)
                    tar_beta_b = tar_beta[sl][0]  # one (300,) per item

                    sample_save_path = os.path.join(results_save_path, sample_names[b])
                    # CLAUDE REMOVE START --- IGNORE ---
                    os.makedirs(sample_save_path, exist_ok=True)
                    np.savez(os.path.join(sample_save_path, 'upper_tokens.npz'),
                        upper_tokens = upper_codes_gt.cpu().numpy()
                    )
                    np.savez(os.path.join(sample_save_path, 'lower_tokens.npz'),
                        lower_tokens = lower_codes_gt.cpu().numpy()
                    )
                    np.savez(os.path.join(sample_save_path, 'face_tokens.npz'),
                        face_tokens = face_codes_gt.cpu().numpy()
                    )
                    # CLAUDE REMOVE END --- IGNORE ---
                    if save or visualize:
                        os.makedirs(sample_save_path, exist_ok=True)

                    if save:
                        np.savez(
                            os.path.join(sample_save_path, 'gt.npz'),
                            betas=tar_beta_b,
                            poses=tar_pose_b,
                            expressions=tar_exps_b,
                            trans=tar_trans_b,
                            model='smplx',
                            gender='NEUTRAL_2020',
                            mocap_frame_rate=30,
                        )
                        np.savez(
                            os.path.join(sample_save_path, 'pred.npz'),
                            betas=tar_beta_b,
                            poses=rec_pose_b,
                            expressions=rec_exps_b,
                            trans=rec_trans_b,
                            model='smplx',
                            gender='NEUTRAL_2020',
                            mocap_frame_rate=30,
                        )
                        np.savez(
                            os.path.join(sample_save_path, 'upper_tokens.npz'),
                            upper_tokens=upper_codes_gt_np[b],
                        )

                    # --- render --- #
                    if visualize:
                        logger.info(f"visualizing {sample_names[b]}")
                        # Re-center each trajectory at origin (matches BEATX
                        # chunk preprocessing + run_gestinference's viz_trans).
                        # render_smplx_debug_video's camera + floor framing
                        # assume SMPL-X v_template feet at y~=-1.05; without
                        # this the floor + body appear misaligned.
                        tar_trans_viz = tar_trans_b - tar_trans_b[0:1, :]
                        rec_trans_viz = rec_trans_b - rec_trans_b[0:1, :]

                        final_path = os.path.join(sample_save_path, "gt_pred_compared_audio.mp4")
                        with tempfile.TemporaryDirectory(prefix="test_sbs_") as tmpdir:
                            gt_path = os.path.join(tmpdir, "gt.mp4")
                            pred_path = os.path.join(tmpdir, "pred.mp4")
                            stitched_path = os.path.join(tmpdir, "stitched.mp4")
                            audio_path = os.path.join(tmpdir, "audio.wav")

                            # GT in red, Pred in blue (mirrors render_smplx_
                            # side_by_side_video's default palette).
                            render_smplx_debug_video(
                                smplx_model=self.smplx_model,
                                poses=tar_pose_b.reshape(num_frames, -1),
                                transl=tar_trans_viz,
                                expressions=tar_exps_b,
                                betas=tar_beta_b,
                                output_path=gt_path,
                                fps=self.args.motion_fps,
                                mesh_color=(180, 54, 54, 255),
                            )
                            render_smplx_debug_video(
                                smplx_model=self.smplx_model,
                                poses=rec_pose_b.reshape(num_frames, -1),
                                transl=rec_trans_viz,
                                expressions=rec_exps_b,
                                betas=tar_beta_b,
                                output_path=pred_path,
                                fps=self.args.motion_fps,
                                mesh_color=(36, 73, 156, 255),
                            )
                            stitch_videos_hstack([gt_path, pred_path], stitched_path)
                            if not os.path.exists(stitched_path):
                                raise RuntimeError(f"hstack failed; no output at {stitched_path}")

                            sf.write(audio_path, tar_raw_audio[b], self.args.audio_fps)
                            shutil.move(stitched_path, final_path)
                            # mux_audio_into_video writes audio onto final_path
                            # IN-PLACE (via os.replace), so the result IS
                            # final_path -- nothing to clean up afterwards.
                            mux_audio_into_video(final_path, audio_path)
                        logger.info(f"Visualization saved to {final_path}")

                           
                # if its == 2:break
        end_time = time.time() - start_time
        self.gesture_metrics.compute_metrics()
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.motion_fps)} s motion")
    