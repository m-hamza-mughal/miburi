import functools
from loguru import logger
import math
from typing import Callable, Union

import safetensors
import torch
import torch.distributed.fsdp.wrap as torch_wrap
from miburi.models.lm import LMModel
from miburi.models.loaders import CheckpointInfo, _is_safetensors
from miburi.modules.transformer import StreamingTransformerLayer
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

# from .args import TrainArgs
from .distributed import get_rank, get_world_size
from .tools import load_checkpoints



def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def get_fsdp_policy() -> Callable[[torch.nn.Module], bool]:
    """
    This function instantiates the FSDP wrap policy.
    - Each Transformers block becomes its own FSDP group so that only a single
      Transformer block is sharded at a time
    """

    # Each transformer block becomes a FSDP group, each being sharded separately
    transformer_block_wrap_policy = functools.partial(
        torch_wrap.transformer_auto_wrap_policy,
        transformer_layer_cls=(StreamingTransformerLayer,),
    )

    return transformer_block_wrap_policy


def log_train_params(model: Union[torch.nn.Module, FullyShardedDataParallel]):
    world_size = get_world_size()

    num_params = world_size * sum(p.numel() for p in model.parameters())
    num_train_params = world_size * sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    main_logger_info(
        f"{num_train_params:,.0f} out of {num_params:,.0f} parameters are finetuned "
        f"({num_train_params / num_params * 100:.2f}%)."
    )

def get_fsdp_model(
    args, model, continue_path=None,
) -> FullyShardedDataParallel:
    """
    Initializes and returns a FullyShardedDataParallel (FSDP) model.
    Works reliably with or without checkpoints.
    """
    
    if args.param_dtype == "bfloat16":
        param_dtype = torch.bfloat16
    elif args.param_dtype == "float32":
        param_dtype = torch.float32
    else:
        param_dtype = torch.float32

    # Simple parameter initialization function
    def param_init_fn(module):
        """Initialize parameters from meta tensors to real tensors"""
        module.to_empty(device=torch.cuda.current_device(), recurse=False)
        module.to(param_dtype)

    # Load checkpoint only on rank 0 if provided
    checkpoint_state_dict = None
    if continue_path is not None and get_rank() == 0:
        main_logger_info(f"Loading checkpoints from {continue_path}")
        try:
            checkpoint_state_dict = safetensors.torch.load_file(continue_path, device="cpu")
            checkpoint_state_dict = {k: v.to(param_dtype) for k, v in checkpoint_state_dict.items()}
            main_logger_info("Checkpoint loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Initializing from scratch.")
            checkpoint_state_dict = None
    
    # Ensure all ranks are synchronized
    torch.distributed.barrier()
    
    # Handle single GPU case
    if get_world_size() == 1:
        if any(p.is_meta for p in model.parameters()):
            for module in model.modules():
                param_init_fn(module)
        
        # Initialize weights using model's method
        if hasattr(model, 'init_weights') and checkpoint_state_dict is None:
            model.init_weights()
        
        # Load checkpoint if available
        if checkpoint_state_dict is not None:
            model.load_state_dict(checkpoint_state_dict, strict=False)
        
        return model.cuda()

    # Create FSDP model
    main_logger_info(f"Sharding model over {get_world_size()} GPUs...")
    
    wrapped_model = FullyShardedDataParallel(
        model,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        # backward_prefetch=BackwardPrefetch.BACKWARD_POST,
        limit_all_gathers=True,
        device_id=torch.cuda.current_device(),
        # auto_wrap_policy=get_fsdp_policy(),
        sync_module_states=True,
        # param_init_fn=param_init_fn,  # Minimal initialization
        use_orig_params=True,
    )

    # Initialize weights ONLY if no checkpoint is being loaded
    # if checkpoint_state_dict is None and hasattr(wrapped_model, 'init_weights'):
        # with wrapped_model.summon_full_params(wrapped_model):
            # wrapped_model.init_weights()
    
    # Load checkpoint after FSDP initialization if available
    if checkpoint_state_dict is not None:
        with wrapped_model.summon_full_params(wrapped_model):
            wrapped_model.load_state_dict(checkpoint_state_dict, strict=False)
    
    torch.distributed.barrier()  # Sync after initialization/loading

    main_logger_info("Model sharded successfully!")
    log_train_params(wrapped_model)
    
    return wrapped_model