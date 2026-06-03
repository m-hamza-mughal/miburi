import configargparse
import time
import json
import yaml
import os


def str2bool(v):
    """ from https://stackoverflow.com/a/43357954/1361529 """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')
        

def parse_args():
    """
    requirement for config
    1. command > yaml > default
    2. avoid re-definition 
    3. lowercase letters is better
    4. hierarchical is not necessary
    """
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add("-c", "--config", default='./configs/vqvae_test.yaml', is_config_file=True)
    parser.add("--project_name", default="a2g_0", type=str) # local device id
    parser.add("--notes", default="", type=str)
    parser.add("--trainer", default="aebody", type=str)
    # ------------- path and save name ---------------- #
    parser.add("--is_train", default=True, type=str2bool)
    parser.add("--debug", default=False, type=str2bool)
    # different between environments
    parser.add("--root_path", default="./", type=str)
    parser.add("--cache_path", default="datasets/beat_cache/beat_smplx_en_body/", type=str)
    parser.add("--out_path", default="./experiments/", type=str)
    # for pretrian weights or smplx models
    parser.add("--deps_path", default="./deps/", type=str)
    # SMPL-X batched forward path (BaseCausalCodecTrainer._smplx_forward).
    parser.add("--smplx_num_betas", default=300, type=int)
    parser.add("--smplx_num_expression_coeffs", default=100, type=int)
    parser.add("--smplx_chunk_per_batch_multiplier", default=64, type=int)
    parser.add("--smplx_skip_expressions_for_loss", default=False, type=str2bool)
    parser.add("--smplx_fast_parity_check_steps", default=0, type=int)
    parser.add("--smplx_fast_parity_tol", default=1e-5, type=float)
    parser.add("--mpjpe_eval_enabled", default=True, type=str2bool)
    # GTDM3 per-bodypart loss reweighting. Applied to per-codebook CE loss for
    # the face codebooks (k >= 16) in UpperFaceLowerGTDM3Trainer. Default 1.0
    # is a no-op; set > 1 in a config to upweight the face codebook losses.
    parser.add("--face_loss_weight", default=1.0, type=float)
    # When True (with vad_guidance), the GTemporalDepthModel3 vad_predictor
    # also consumes the last-4 face logits (flattened) alongside transformer_out.
    parser.add("--vad_use_face_logits", default=False, type=str2bool)
    # ------------------- evaluation ----------------------- #
    parser.add("--test_ckpt", default="/datasets/beat_cache/beat_4english_15_141/last.bin")
    parser.add("--visualize", default=False, type=str2bool,
               help="When running scripts/test.py, also render the per-sample "
                    "side-by-side GT/Pred mp4. Off by default; pass "
                    "--visualize True to opt in.")
    parser.add("--max_batches", default=None, type=int,
               help="Cap the number of test-loader batches processed by "
                    "scripts/test.py. Useful for spot-checking a new "
                    "visualization or rerunning metrics on a subset. "
                    "Default: process the whole test set.")
    parser.add("--save", default=False, type=str2bool,
               help="When running scripts/test.py, also write per-sample "
                    "gt.npz / pred.npz / upper_tokens.npz (and the codec "
                    "trainers' rec_out / tar_out dicts) under <exp_dir>/"
                    "test_<epoch>/<sample_id>/. Off by default so a "
                    "metrics-only run doesn't pollute disk.")
    parser.add("--latent_dim", default=512, type=int)
    parser.add("--nfeats", default=78, type=int, nargs="*")
    # parser.add("--vae_test_stride", default=10, type=int)
    parser.add("--test_period", default=20, type=int)
    parser.add("--codebook_size", default=1024, type=int)
    parser.add("--quantizer_lambda", default=1., type=float)
    parser.add("--quantizer", default="Quantizer", type=str)
    parser.add("--frame_chunk_size", default=8, type=int)
    parser.add("--ff_size", default=1024, type=int)
    parser.add("--decoder_arch", default="encoder_decoder", type=str)
    parser.add("--transformer_normalize_before", default=False, type=str2bool)
    parser.add("--transformer_activation", default="gelu", type=str)
    parser.add("--position_embedding", default="sine", type=str)
    
    parser.add("--vae_dist", default=None, type=str)
    
    
    # --------------- data ---------------------------- #
    parser.add("--additional_data", default=False, type=str2bool)
    parser.add("--dataset", default="beat2", type=str)
    parser.add("--rot6d", default=True, type=str2bool)
    parser.add("--ori_joints", default="beat_smplx_joints", type=str)
    parser.add("--tar_joints", default="beat_smplx_full", type=str)
    parser.add("--training_speakers", default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], type=int, nargs="*")
    
    parser.add("--new_cache", default=True, type=str2bool)
    parser.add("--beat_align", default=True, type=str2bool)
    parser.add("--word_cache", default=False, type=str2bool)
    parser.add("--disable_filtering", default=False, type=str2bool)
    parser.add("--clean_first_seconds", default=0, type=int)
    parser.add("--clean_final_seconds", default=0, type=int)

    parser.add("--audio_rep", default=None, type=str)
    parser.add("--audio_sr", default=24000, type=int)
    parser.add("--word_rep", default=None, type=str)
    parser.add("--emo_rep", default=None, type=str)
    parser.add("--sem_rep", default=None, type=str)
    parser.add("--prom_rep", default=None, type=str)
    parser.add("--facial_rep", default="smplxflame_30", type=str)
    parser.add("--pose_rep", default="smplxflame_30", type=str)
    parser.add("--id_rep", default="onehot", type=str)
    parser.add("--speaker_id", default="onehot", type=str)
    
    
    parser.add("--audio_fps", default=24000, type=int)
    parser.add("--motion_fps", default=25, type=int)
    
    
    # parser.add("--audio_norm", default=False, type=str2bool)
    parser.add("--facial_norm", default=False, type=str2bool)
    parser.add("--pose_norm", default=False, type=str2bool)
        
    parser.add("--pose_length", default=64, type=int)
    # parser.add("--pre_frames", default=4, type=int)
    parser.add("--datastride", default=10, type=int)
    parser.add("--data_nrand", default=2, type=int)

    
    parser.add("--multi_length_training", default=[1.0], type=float, nargs="*")
    # --------------- model ---------------------------- #
    parser.add("--pretrain", default=False, type=str2bool)
    parser.add("--model", default="vae", type=str)
    parser.add("--g_name", default="TransformerVAE", type=str)
    
    parser.add("--dropout", default=0.1, type=float)
    
    # --------------- training ------------------------- #
    parser.add("--epochs", default=120, type=int)
    # parser.add("--epoch_stage", default=0, type=int)
    parser.add("--grad_norm", default=0, type=float)
    # parser.add("--no_adv_epoch", default=999, type=int)
    parser.add("--batch_size", default=128, type=int)
    parser.add("--opt", default="adamw", type=str)
    parser.add("--lr_base", default=1e-4, type=float)
    parser.add("--opt_betas", default=[0.9, 0.95], type=float, nargs="*")
    parser.add("--weight_decay", default=0., type=float)
    # for warmup and cosine
    parser.add("--lr_min", default=1e-6, type=float)
    parser.add("--warmup_lr", default=5e-4, type=float)
    parser.add("--warmup_epochs", default=0, type=int)
    parser.add("--decay_epochs", default=9999, type=int)
    parser.add("--decay_rate", default=0.1, type=float)
    parser.add("--lr_policy", default="step", type=str)
    # for sgd
    parser.add("--momentum", default=0.8, type=float)
    parser.add("--rec_weight", default=500, type=float)
    parser.add("--vel_weight", default=0.0, type=float)
    parser.add("--acc_weight", default=0.0, type=float)
    parser.add("--kld_weight", default=0.0, type=float)
    parser.add("--comloss_weight", default=0.0, type=float)
    parser.add("--atcont", default=0.0, type=float)
    parser.add("--fusion_mode", default="sum", type=str)
    
    parser.add("--div_reg_weight", default=0.0, type=float)
    parser.add("--rec_ver_weight", default=0.0, type=float)
    parser.add("--rec_pos_weight", default=0.0, type=float)
    parser.add("--rec_aa_weight", default=0.0, type=float)
    parser.add("--rec_6d_weight", default=0.0, type=float)
    parser.add("--rec_loc_weight", default=0.0, type=float)
    parser.add("--rec_contact_weight", default=0.0, type=float)
    parser.add("--rec_trans_weight", default=0.0, type=float)
    parser.add("--rec_face_weight", default=0.0, type=float)
    parser.add("--lap_weight", default=0.0, type=float)
#    parser.add("--gan_noise_size", default=0, type=int)
    
    # --------------- device -------------------------- #
    parser.add("--random_seed", default=2021, type=int)
    parser.add("--deterministic", default=True, type=str2bool)
    parser.add("--benchmark", default=True, type=str2bool)
    parser.add("--cudnn_enabled", default=True, type=str2bool)
    # mix precision
    parser.add("--apex", default=False, type=str2bool)
    parser.add("--gpus", default=[0], type=int, nargs="*")
    parser.add_argument("--local-rank", "--local_rank", type=int)
    parser.add("--loader_workers", default=0, type=int)
    parser.add("--ddp", default=False, type=str2bool)
    parser.add("--sparse", default=1, type=int)
    #parser.add("--world_size")

    parser.add("--is_continue", default=False, type=str2bool)
    parser.add("--continue_ckpt", default=None, type=str)


    parser.add("--code_save_split", default=None, type=str)

    # --------------- extra args -------------------------- #
    parser.add("--num_frames", default=64, type=int)
    parser.add("--test_length", default=64, type=int)
    parser.add("--name", default=None, type=str)

    # logging
    parser.add("--log_period", default=10, type=int)

    # --------------- vectorquantize args ---------------- #
    parser.add("--codebook_dim", default=None, type=int) # for lower codebook dim
    parser.add("--vq_ema_decay", default=0.8, type=float)
    parser.add("--vq_kmeans_init", default=False, type=str2bool)
    parser.add("--vq_kmeans_iters", default=10, type=int)

    # l2 normalization of the codes and the encoded vectors
    parser.add("--vq_cosine_sim", default=False, type=str2bool) 
    
    # this should actively replace any codes that have an exponential moving average 
    # cluster size less than value
    parser.add("--vq_deadcodethresh", default=0, type=float) 

    # Orthogonal regularization loss
    parser.add("--vq_orthogonalreg_weight", default=0, type=float)
    # this would randomly sample from the codebook for the loss, 
    # for limiting memory usage
    parser.add("--vq_orthogonalreg_maxcodes", default=None, type=int)
    # if you have a very large codebook, and would only like to enforce the 
    # loss on the activated codes per batch
    parser.add("--vq_orthogonalreg_activecodesonly", default=False, type=str2bool)

    # Multi-headed VQ
    parser.add("--vq_num_heads", default=1, type=int)
    parser.add("--vq_sepcodebookperhead", default=False, type=str2bool)

    # codebook_diversity loss
    parser.add("--vq_diversityloss_weight", default=0, type=float)
    parser.add("--vq_diversityloss_temp", default=100, type=float)

    # --------------- residual vqvae args ---------------- #
    parser.add("--num_quantizers", default=1, type=int)
    # codebook sharing across quantizers (https://arxiv.org/abs/2203.01941)
    parser.add("--shared_codebook", default=False, type=str2bool)
    # stocahstic code sampling (https://arxiv.org/abs/2203.01941)
    parser.add("--stochastic_sample_codes", default=False, type=str2bool)
    parser.add("--stochastic_sampletemp", default=0.1, type=float)

    # grouper residual vqvae
    parser.add("--quantizer_groups", default=1, type=int)
    # implicit neural codebook
    parser.add("--implicit_codebook", default=False, type=str2bool)


    parser.add("--sem_loss", default=False, type=str2bool)

    parser.add("--resample_motionfps", default=False, type=str2bool)
    

    # teacher args
    parser.add("--gesturecodec_ckpt", default=None, type=str)
    parser.add("--emb_recon_weight", default=1.0, type=float)
    parser.add("--gid_ce_weight", default=1.0, type=float)
    parser.add("--tid_ce_weight", default=1.0, type=float)
    parser.add("--contrastive_weight", default=1.0, type=float)
    parser.add("--text_contrastive_weight", default=1.0, type=float)
    parser.add("--teacherguidance_weight", default=1.0, type=float)
    parser.add("--vilbert_contrastive_weight", default=1.0, type=float)
    parser.add("--vil_num_negatives", default=3, type=int)
    parser.add("--teacher_ckpt", default=None, type=str)

    parser.add("--transformer_layers", default=8, type=int)
    parser.add("--convblock_layers", default=2, type=int)
    parser.add("--transformer_heads", default=16, type=int)
    parser.add("--decoder_sizefactor", default=2, type=int)
    

    parser.add("--use_semantic_teacher", default=True, type=str2bool)
    
    # parser.add("--gan_loss_weight", default=1.0, type=float)
    
    

    parser.add("--gestureformer_layers", default=8, type=int)
    parser.add("--gestureformer_heads", default=16, type=int)
    parser.add("--gestureformer_depformer_heads", default=16, type=int)
    parser.add("--gestureformer_depformer_layers", default=8, type=int)
    parser.add("--upperlower_nfeats", default=258, type=int)
    parser.add("--lowertrans_nfeats", default=61, type=int)
    parser.add("--face_nfeats", default=106, type=int)
    parser.add("--lowerbodycodec_ckpt", default=None, type=str)
    parser.add("--upperbodycodec_ckpt", default=None, type=str)
    parser.add("--facecodec_ckpt", default=None, type=str)

    parser.add("--tempdepth_consistency_weight", default=0.1, type=float)

    parser.add("--mask_schedule", default="cosine", type=str)
    parser.add("--max_unmaskedepochs", default=25, type=int)

    parser.add("--inputtoken_noise_prob", default=0.0, type=float)
    parser.add("--pretrain_warmup_epochs", default=0, type=int)
    parser.add("--memory_dropout_prob", default=0.0, type=float)
    parser.add("--memory_embnoise_prob", default=0.0, type=float)

    parser.add("--num_temp_classifiers", default=2, type=int)

    parser.add("--param_dtype", default="float32", type=str)
    parser.add("--optim_dtype", default="float32", type=str)

    parser.add("--lr_cyclestepsizeup", default=1000, type=int)

    parser.add("--beatx_data_path", default="/CT/GestureSynth1/work/GestureGPT/PantoMatrix/BEAT2/beat_english_v2.0.0/", type=str)
    parser.add("--embody3d_path", default="/CT/GestureSynth2/work/Embody3D/dataset/aiagent/", type=str)
    # UNIFIEDDataset paths (new HDF5 schema under datasets/data_cache/).
    # These also serve as the "eval HDF5 path" knobs for scripts/test.py
    # -- override the YAML value at the CLI to point a test run at the
    # full-sequence eval HDF5 (see README "Evaluation HDF5" section).
    parser.add("--beatx_cache_path", default=None, type=str,
               help="Directory containing the BEATX UnifiedDataset HDF5 "
                    "(e.g. datasets/data_cache/beatx_train/ or .../beatx_eval/). "
                    "Set per-run on the CLI of scripts/test.py to swap "
                    "between the chunked-training and the full-sequence "
                    "evaluation caches without touching the YAML.")
    parser.add("--embody3d_cache_path", default=None, type=str,
               help="Directory containing the Embody3D UnifiedDataset HDF5. "
                    "Same role as --beatx_cache_path, for the other dataset.")
    parser.add("--index_cache_dir", default=None, type=str)
    parser.add("--return_joint_positions", default=False, type=str2bool)
    parser.add("--align_first_frame_yaw", default=True, type=str2bool)
    # ------------------ seamless interaction args ------------------ #
    parser.add("--motion_dir", default=None, type=str)
    parser.add("--body_part", default="full", type=str)  # full, upper, lower, hands
    parser.add("--file_list_path", default=None, type=str)
    parser.add("--varying_frame_length", default=True, type=str2bool)  # varying frame length for training

    # ------------------- flow modelling args ------------------ #
    parser.add("--cfg_dropout_prob", default=0.0, type=float)  #
    parser.add("--cfg_scale", default=4, type=float)
    parser.add("--gestureformer_flow_heads", default=16, type=int)
    parser.add("--gestureformer_flow_layers", default=8, type=int)
    parser.add("--rate_flowselfconsistency", default=0.25, type=float)
    parser.add("--flowtimestep_skewedsampling", default=False, type=str2bool)
    parser.add("--latent_dtype", default="bfloat16", type=str)


    # ------------------- VAD guidance for seamless interaction ------------------ #
    parser.add("--vad_guidance", default=False, type=str2bool)  #
    parser.add("--vad_loss_weight", default=0.0, type=float)  #

    # ------------------- Causal condition prediction for flow (Arch 1_6/9) ------------------ #
    parser.add("--causalcond_supervision_weight", default=0.0, type=float)  #

    # ------------------- curriculum training ------------------ #
    parser.add("--dataset_ratio", default="full_beatx", type=str)  # full_beatx, 80beatx_20seamless, 50beatx_50seamless
    parser.add("--mmd_weight", default=0.0, type=float)  #
    parser.add("--accel_penalty_weight", default=0.0, type=float)  #
    parser.add("--globalorient_weight", default=2.0, type=float)  #

    parser.add("--latent_diff_loss_weight", default=0.0, type=float)  #

    parser.add("--kl_warmup_epochs", default=0, type=int)
    parser.add("--kl_free_bits", default=0.0, type=float)


    parser.add("--use_wavelet", default=False, type=str2bool)

    parser.add("--drop_lower_crossattn", default=False, type=str2bool)

    parser.add("--contrastive_loss_weight", default=1.0, type=float)
    parser.add("--gan_loss_weight", default=1.0, type=float)
    parser.add("--genrecon_loss_weight", default=1.0, type=float)
    parser.add("--textaudio_emb_freeze", default=False, type=str2bool)

    
    parser.add("--lower_bodypart_dropprob", default=0.0, type=float)

    parser.add("--flow_mode", default="v", type=str)  # x1 or v
    parser.add("--time_sample_strategy", default="beta", type=str)  # uniform or beta or lognorm
    parser.add("--rate_x1predictionloss", default=0.0, type=float)  #
    parser.add("--codebook_classifier_rate", default=0.0, type=float)  #

    # ------------------- Meanflow args ------------------ #
    parser.add("--meanflowloss", default="adaptive_l2_loss", type=str)  # l2_loss or adaptive_l2_loss
    parser.add("--w_cfg_scale", default=None, type=float)  #
    parser.add("--v_for_uncond", default=True, type=str2bool)  #
    parser.add("--jvp_api", default='autograd', type=str)  #
    parser.add("--meanflow_ratio", default=0.5, type=float)  #
    parser.add("--trainstep_compile", default=False, type=str2bool)  #
    parser.add("--gradsynccheck_epoch", default=100, type=int)  #
    
    
    args = parser.parse_args()
    
    args.name = args.project_name
    
    is_train = args.is_train

    args.num_frames = args.pose_length
    args.test_length = args.pose_length

    if is_train:
        time_local = time.localtime()
        name_expend = "%02d%02d_%02d%02d_"%(time_local[1], time_local[2],time_local[3], time_local[4])
        
        if args.debug:
            args.name = name_expend + "debug_" + args.name
        else:
            args.name = name_expend + args.name
        
    return args