# import logging
from loguru import logger
import sys
import time
import warnings
import os
import random
import torch
# import torch.multiprocessing as mp
# import torch.distributed as dist

import trainers
from trainers.utils import config
from trainers.utils import tools as other_tools

# logger = logging.getLogger()

# @logger.catch
def main_worker(rank, world_size, args):
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    # dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    args.is_train = True
    other_tools.set_args_and_logger(args, rank)
    other_tools.set_random_seed(args)
    other_tools.print_exp_info(args)
    
    # return one intance of trainer
    
    trainer = getattr(trainers, args.trainer+"Trainer")(args)
    logger.info("Visualizing codebook ...")
    logger.info(f"Checkpoint: {args.test_ckpt}")

    other_tools.load_checkpoints(trainer.model, args.test_ckpt, args.g_name)

    ckpt_epoch = os.path.basename(args.test_ckpt).split("_")[1].replace(".bin", "")
    
    # trainer.val(15)
    checkpoint_dir = os.path.dirname(args.test_ckpt)
    # trainer.visualize_codebook(checkpoint_dir, ckpt_epoch)
    # trainer.visualize_textcodebook()
    trainer.visualize_trained_embeddings(checkpoint_dir, int(ckpt_epoch))

    
    # trainer.generate_codebook_motions(checkpoint_dir)
    logger.info("Testing done")
    

if __name__ == "__main__":
    os.environ["MASTER_ADDR"]='127.0.0.1'
    

    # generate a random port 
    os.environ["MASTER_PORT"]=str(random.randint(2000, 2959))
    #os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    args = config.parse_args()
    args.is_train = False
    # if args.ddp:
    #     mp.set_start_method("spawn", force=True)
    #     mp.spawn(
    #         main_worker,
    #         args=(len(args.gpus), args,),
    #         nprocs=len(args.gpus),
    #             )
    # else:
    main_worker(0, 1, args)