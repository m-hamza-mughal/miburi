from loguru import logger
import os
from functools import lru_cache
from typing import List, Union

import torch
import torch.distributed as dist

# logger = logging.getLogger("distributed")

BACKEND = "nccl"


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


@lru_cache()
def get_rank() -> int:
    if is_distributed():
        return dist.get_rank()
    return 0


@lru_cache()
def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    return 1


def visible_devices() -> List[int]:
    return [int(d) for d in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]


def set_device():
    """Bind the calling process to a CUDA device.

    Works in two modes:
    - torchrun-launched: LOCAL_RANK is set; binds to that rank's GPU.
    - single-process (`python scripts/train.py`): binds to cuda:0.
    """
    if "LOCAL_RANK" in os.environ:
        # torchrun path
        logger.info(f"torch.cuda.device_count: {torch.cuda.device_count()}")
        logger.info(
            f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
        )
        logger.info(f"local rank: {int(os.environ['LOCAL_RANK'])}")
        logger.info(f"global rank: {int(os.environ['RANK'])}")

        assert torch.cuda.is_available()

        if "CUDA_VISIBLE_DEVICES" in os.environ:
            assert len(visible_devices()) == torch.cuda.device_count()

        if torch.cuda.device_count() == 1:
            torch.cuda.set_device(0)
            return

        local_rank = int(os.environ["LOCAL_RANK"])
        logger.info(f"Set cuda device to {local_rank}")
        assert 0 <= local_rank < torch.cuda.device_count(), (
            local_rank,
            torch.cuda.device_count(),
        )
        torch.cuda.set_device(local_rank)
        return

    # Single-process path: plain `python scripts/train.py`
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        logger.info(
            "Single-process mode: bound to cuda:0 "
            f"(torch.cuda.device_count={torch.cuda.device_count()})"
        )
    else:
        logger.warning("Single-process mode: no CUDA available; running on CPU.")


def avg_aggregate(metric: Union[float, int]) -> Union[float, int]:
    if not is_distributed():
        return float(metric)
    buffer = torch.tensor([metric], dtype=torch.float32, device="cuda")
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    return buffer[0].item() / get_world_size()


def is_torchrun() -> bool:
    return "TORCHELASTIC_RESTART_COUNT" in os.environ
