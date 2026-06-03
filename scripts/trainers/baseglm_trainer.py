import smplx
import os
import json
import torch
import copy
# import logging
import math
from abc import abstractmethod
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


from .utils.optim_factory import create_optimizer
from .utils.scheduler_factory import create_scheduler
from .utils.tools import save_checkpoints, load_checkpoints
from .utils.distributed import get_rank, get_world_size
from . import dataloaders as dataset
from miburi.models import loaders, GestureMimiCodec

# logger = logging.getLogger()



class BaseGLMTrainer(object):
    def __init__(self, args):
        self.args = args
        if args.ddp: dist.barrier()
        self.local_rank = int(os.environ.get("LOCAL_RANK", "N/A")) if args.ddp else 0
        self.global_rank = get_rank() if args.ddp else 0
        
        self.checkpoint_path = args.out_path + args.name + args.notes + "/"
        if not args.is_train:
            # breakpoint()
            self.checkpoint_path = os.path.dirname(args.test_ckpt) + "/"
        
        if self.global_rank==0 and self.args.is_train: # and args.code_save_split is None:
            self.writer = SummaryWriter(log_dir=self.checkpoint_path)
        
        # breakpoint() # check codecs and their difference
        
        self.upper_gesture_codec, self.face_gesture_codec, self.lower_gesture_codec = \
            self.get_dep_model(args)
        
        
        mimi_frame_rate = loaders.FRAME_RATE
        # breakpoint() # check frame rate and codec difference
        self.codec_difference = int(mimi_frame_rate / self.upper_gesture_codec.frame_rate)

        self.model = self.get_model(args)
        self.audio_codec_nulltoken = -1 #self.model.audio_procemb[0].num_embeddings - 1
        self.text_codec_nulltoken = -1 # self.model.text_procemb.num_embeddings - 1
        
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
                # self.model.init_weights()
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

        # self.model = self.model.to(self.rank)
        self.upper_gesture_codec = self.upper_gesture_codec.to(self.local_rank)
        self.lower_gesture_codec = self.lower_gesture_codec.to(self.local_rank)
        self.face_gesture_codec = self.face_gesture_codec.to(self.local_rank)

        if args.ddp:
            assert get_world_size() > 1, "DDP requires more than one process"
            process_group = torch.distributed.new_group()
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model, process_group)
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank,
                             broadcast_buffers=False, find_unused_parameters=True)
            # self.mimi = DDP(self.mimi, device_ids=[self.rank], output_device=self.rank,
            #                  broadcast_buffers=False, find_unused_parameters=False)
            # self.upper_gesture_codec = DDP(self.upper_gesture_codec, device_ids=[self.rank], output_device=self.rank,
            #                  broadcast_buffers=False, find_unused_parameters=False)
            # self.lower_gesture_codec = DDP(self.lower_gesture_codec, device_ids=[self.rank], output_device=self.rank,
            #                  broadcast_buffers=False, find_unused_parameters=False)
            # self.text_procemb = DDP(self.text_procemb, device_ids=[self.rank], output_device=self.rank,
            #                  broadcast_buffers=False, find_unused_parameters=False)
            # self.audio_procemb = DDP(self.audio_procemb, device_ids=[self.rank], output_device=self.rank,
            #                     broadcast_buffers=False, find_unused_parameters=False)
        
        for p in self.model.parameters():
            if torch.isnan(p).any():
                raise ValueError("nan in model parameters")
        
        self.model.train()
        self.upper_gesture_codec.eval()
        self.lower_gesture_codec.eval()
        self.face_gesture_codec.eval()

        self.tracker = None
        self.dataset_file = dataset
        dataset_class = getattr(self.dataset_file, args.dataset.upper() + "Dataset")

        if args.is_train:
            self.train_data = dataset_class(
                args, 
                "train", 
                only_motion=False, 
                dataset_ratio=args.dataset_ratio,
                debug=args.debug,
                varying_frame_length=False, # TODO: maybe experiment later 
                ret_rawaudio=False,
                ret_vad=args.vad_guidance,
            )

            self.val_data = dataset_class(
                args, 
                "val", 
                only_motion=False,
                dataset_ratio=args.dataset_ratio,
                debug=args.debug,
                varying_frame_length=False, # TODO: maybe experiment later
                ret_rawaudio=False,
                ret_vad=args.vad_guidance,
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
                in_order=False,
            )
            # self.train_loader = self.train_loader
            
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
                in_order=False,
            )
            logger.info(f"[GPU{self.global_rank}:{self.local_rank}] Init val dataloader success")

            if self.global_rank == 0 and getattr(self.train_data, "speaker_id_to_index", None):
                spkmap_path = os.path.join(self.checkpoint_path, "speaker_id_to_index.json")
                os.makedirs(self.checkpoint_path, exist_ok=True)
                with open(spkmap_path, "w") as f:
                    json.dump(self.train_data.speaker_id_to_index, f, indent=2)
                logger.info(f"[GPU{self.global_rank}:{self.local_rank}] Wrote speaker map -> {spkmap_path}")

        if self.local_rank == 0:
            self.test_data = dataset_class(
                args, 
                "test", 
                only_motion=False, 
                debug=args.debug,
                dataset_ratio=args.dataset_ratio,
                varying_frame_length=False,
                ret_rawaudio=False if args.is_train else True,  # No raw audio in training
                ret_vad=True if args.is_train and args.vad_guidance else False,
            )
            self.test_loader = torch.utils.data.DataLoader(
                self.test_data,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.loader_workers,
                drop_last=False, 
                sampler=None,
                collate_fn=self.test_data.collate_fn,
            )
            logger.info(f"[GPU{self.global_rank}:{self.local_rank}] Init test dataloader success")

        
        # if self.local_rank == 0:
        if self.global_rank == 0: logger.info(self.model)
        model_device = next(self.model.parameters()).device
        logger.info(f"[GPU{self.global_rank}:{self.local_rank}] init model success {model_device}") 
           
        
        if args.is_train:
            self.opt = create_optimizer(args, self.model)
            self.opt_s = create_scheduler(args, self.opt)
        
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
        
        # --- for debugging purposes # TODO: remove later
        # self.train_data = self.test_data
        # self.val_data = self.test_data
        # self.train_loader = self.test_loader
        # self.val_loader = self.test_loader
        # self.train_length = len(self.train_loader)

        # breakpoint()
        # model_module = __import__(f"models.{args.model}", fromlist=["something"])
        
        # if args.ddp:
        #     self.model = getattr(model_module, args.g_name)(args).to(self.rank)
        #     process_group = torch.distributed.new_group()
        #     self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model, process_group)
        #     self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank,
        #                      broadcast_buffers=False, find_unused_parameters=False)
        # else:
        # self.model = torch.nn.DataParallel(getattr(model_module, args.g_name)(args), args.gpus).cuda()
        
        # # Hook function to print tensor values
        # self.end_offsets = []
        # self.k_sa = []
        # self.attn_sa = []
        # def KV_forward_hook(module, inp, oup):
        #     if module._streaming_state is not None:
        #         # torch.cuda.synchronize()  # Ensure CUDA operations complete before printing
        #         self.end_offsets.append(module._streaming_state.kv_cache.cache[0][0, 0, :, 0])
        #         self.k_sa.append(module.temp_k_tensor[0,0,:,0])
        #         self.attn_sa.append(module.temp_attn) 

        # self.attn_biases = []
        # self.delta_downsampled = []
        # self.posks = []
        # self.posqs = []
        # self.posqs_before = []
        # self.k_tensors = []
        # def CA_forward_hook(module, inp, oup):
        #     if module.temp_attn is not None:
        #         # torch.cuda.synchronize()  # Ensure CUDA operations complete before printing
        #         self.attn_biases.append(module.temp_attn)
        #         self.delta_downsampled.append(module.temp_delta)
        #         self.posks.append(module.temp_posk)
        #         self.posqs.append(module.temp_posq)
        #         # self.posqs_before.append(module.temp_posq_before)
        #         self.k_tensors.append(module.temp_k_tensor[0, 0, :, 0])

        # breakpoint()
        

        # self.latent_proj = get_latent_proj(args)
        # self.latent_proj = self.latent_proj.cuda()

        # self.kv_handle_1 = self.model.decoder.layers[0].cross_attn.register_forward_hook(KV_forward_hook)
        # self.kv_handle_2 = self.model.decoder.layers[1].cross_attn.register_forward_hook(KV_forward_hook)
        
        # self.ca_handle_1 = self.model.decoder.layers[0].cross_attn.register_forward_hook(CA_forward_hook)
        # self.ca_handle_2 = self.model.decoder.layers[1].cross_attn.register_forward_hook(CA_forward_hook)
        # --- for debugging purposes # TODO: remove later

    # inverse selection functions here or in utils? #TODO

    # @staticmethod
    def get_dep_model(self, args):
        upper_gesture_codec_kwargs = loaders.get_uppergesturecodec_kwargs()
        lower_gesture_codec_kwargs = loaders.get_uppergesturecodec_kwargs()
        face_gesture_codec_kwargs = loaders.get_facegesturecodec_kwargs()

        upper_gesture_codec = GestureMimiCodec(
            num_frames=args.num_frames,
            frame_chunk_size=args.frame_chunk_size,
            nfeats=args.upperlower_nfeats,
            motion_fps=args.motion_fps,
            num_heads=args.transformer_heads,
            # num_layers=args.transformer_layers,
            transformer_layers=args.transformer_layers,
            convblock_layers=args.convblock_layers,
            # decoder_sizefactor = args.decoder_sizefactor,
            **upper_gesture_codec_kwargs
        )
        load_checkpoints(upper_gesture_codec, args.upperbodycodec_ckpt)
        upper_gesture_codec.eval()
        for p in upper_gesture_codec.parameters():
            p.requires_grad = False

        logger.info(f"[GPU{self.global_rank}:{self.local_rank}] Upper gesture codec loaded")

        lower_gesture_codec = GestureMimiCodec(
            num_frames=args.num_frames,
            frame_chunk_size=args.frame_chunk_size,
            nfeats=args.lowertrans_nfeats,
            motion_fps=args.motion_fps,
            num_heads=args.transformer_heads,
            # num_layers=args.transformer_layers,
            transformer_layers=args.transformer_layers,
            convblock_layers=args.convblock_layers,
            # decoder_sizefactor = args.decoder_sizefactor,
            **lower_gesture_codec_kwargs
        )
        load_checkpoints(lower_gesture_codec, args.lowerbodycodec_ckpt)
        lower_gesture_codec.eval()
        for p in lower_gesture_codec.parameters():
            p.requires_grad = False

        logger.info(f"[GPU{self.global_rank}:{self.local_rank}] Lower gesture codec loaded")

        if args.facecodec_ckpt is not None:
            face_gesture_codec = GestureMimiCodec(
                num_frames=args.num_frames,
                frame_chunk_size=args.frame_chunk_size,
                nfeats=args.face_nfeats,
                motion_fps=args.motion_fps,
                num_heads=args.transformer_heads // 2,
                # num_layers=args.transformer_layers,
                transformer_layers=args.transformer_layers // 2,
                convblock_layers=args.convblock_layers,
                # decoder_sizefactor = args.decoder_sizefactor,
                **face_gesture_codec_kwargs
            )
            load_checkpoints(face_gesture_codec, args.facecodec_ckpt)
            face_gesture_codec.eval()
            for p in face_gesture_codec.parameters():
                p.requires_grad = False

            logger.info(f"[GPU{self.global_rank}:{self.local_rank}] Face gesture codec loaded")
            return upper_gesture_codec, face_gesture_codec, lower_gesture_codec

        
        return upper_gesture_codec, lower_gesture_codec


    
    
    @abstractmethod
    def get_model(self, args):
        pass

    def train_recording(self, epoch, its, t_data, t_train, mem_cost, lr_g, lr_d=None):
        pstr = f"[GPU{self.global_rank}:{self.local_rank}] [{epoch:03}][{its:03}/{self.train_length:03d}]"
        for name, states in self.tracker.loss_meters.items():
            metric = states['train']
            if metric.count > 0:
                pstr += f"{name}: {metric.avg:.4f}\t"
                
                if self.global_rank == 0: self.writer.add_scalar(f"train/{name}", metric.avg, epoch*self.train_length+its)
        pstr += f"glr: {lr_g:.1e}\t"
        if self.global_rank == 0: self.writer.add_scalar("lr/glr", lr_g, epoch*self.train_length+its)
        if lr_d is not None:
            pstr += f"dlr: {lr_d:.1e}\t"
            if self.global_rank == 0: self.writer.add_scalar("lr/dlr", lr_d, epoch*self.train_length+its)
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
                    if self.global_rank == 0: self.writer.add_scalars(f"val/{name}", {name+"_val":metric.avg, name+"_train":states['train'].avg}, epoch*self.train_length)
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
