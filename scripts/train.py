# import logging
from loguru import logger
import sys
import time
import warnings
import os
import torch.multiprocessing as mp
import torch.distributed as dist


import trainers
from trainers.utils import config
from trainers.utils import tools as other_tools

from trainers.utils.distributed import (
    BACKEND,
    avg_aggregate,
    get_rank,
    get_world_size,
    is_torchrun,
    set_device,
)

# logger = logging.getLogger()

# @logger.catch
def main_worker(args):
    other_tools.set_random_seed(args)
    
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    # Initialize the process group using environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if "LOCAL_RANK" in os.environ:
        # torchrun-launched (multi-GPU DDP)
        set_device()
        logger.info("Going to init comms...")
        dist.init_process_group(backend=BACKEND)
    else:
        # Plain `python scripts/train.py` (single GPU, no DDP)
        logger.info("LOCAL_RANK not set: single-process mode, skipping dist.init_process_group.")
        set_device()

    world_size = get_world_size()
    global_rank = get_rank()

    if world_size > 1:
        args.ddp = True  # Ensure ddp flag is set
    else:
        args.ddp = False

    logger.info(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not Set')}")
    
    other_tools.set_args_and_logger(args, global_rank)
    if global_rank == 0:
        other_tools.print_exp_info(args)
    
    # return one intance of trainer
    logger.info(f"Rank {global_rank} trainer initializing...")
    trainer = getattr(trainers, args.trainer+"Trainer")(args)
    if args.is_continue:
        # logger.info("Continue training ...")
        # other_tools.load_checkpoints(trainer.model, args.continue_ckpt, after_distributed=True)
        start_epoch = int(os.path.basename(args.continue_ckpt).split("_")[1].replace(".safetensors", "")) + 1
        if global_rank == 0: logger.info(f"Start from epoch {start_epoch}")
    else:
        start_epoch = 0
        if global_rank == 0: logger.info("Training from scratch ...")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs+1):
        if args.ddp: 
            trainer.val_loader.sampler.set_epoch(epoch)
        
        # trainer.val(epoch)
        
        epoch_time = time.time()-start_time
        
        if epoch != args.epochs:
            if args.ddp: 
                trainer.train_loader.sampler.set_epoch(epoch)
            trainer.tracker.reset()
            trainer.train(epoch)
        if args.debug:
            if global_rank == 0:
                trainer.val(epoch)
                other_tools.save_checkpoints(os.path.join(trainer.checkpoint_path, f"last_{epoch}"), trainer.model, opt=None, epoch=None, lrs=None, save_dtype=args.param_dtype)
                other_tools.load_checkpoints(trainer.model, os.path.join(trainer.checkpoint_path, f"last_{epoch}.safetensors"), after_distributed=True)
            # trainer.test(epoch)
        
        if (epoch) % args.test_period == 0:
            if global_rank == 0:
                # if epoch % (args.test_period * 2) == 0 and epoch != 0: dist.barrier()
                logger.info(f"[GPU {global_rank}] Saving checkpoints:")   
                other_tools.save_checkpoints(os.path.join(trainer.checkpoint_path, f"last_{epoch}"), trainer.model, opt=None, epoch=None, lrs=None, save_dtype=args.param_dtype)
                # trainer.test(epoch)
                args.test_ckpt = os.path.join(trainer.checkpoint_path, f"last_{epoch}.safetensors")
                other_tools.update_args_file(args, rank=global_rank)
        
        if epoch % (args.test_period) == 0 and epoch != 0:
            logger.info(f"GPU {global_rank} Validation:")
            trainer.val(epoch)

            # dist.barrier()
        
        if trainer.global_rank == 0:
            logger.info(f"Time info >>>>  elapsed: {epoch_time/60:.2f} mins\t" + f"remain: {(args.epochs/(epoch+1e-7)-1)*epoch_time/60:.2f} mins")
       
    

if __name__ == "__main__":
    

    
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    args = config.parse_args()
    args.is_train = True
    main_worker(args)