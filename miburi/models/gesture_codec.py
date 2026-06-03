# Part of this file is adapted from compression.py in the moshi repository.
# released under the following license.
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
# import logging
import random
from loguru import logger
import typing as tp

import torch
from torch import nn
import torch.nn.functional as F


from ..modules.resample import ConvDownsample1d, ConvTrUpsample1d
from ..modules.seanet import SEANetDecoder, SEANetDecoder2, SEANetEncoder, SEANetEncoder2
from ..modules.conv import StreamingConvTranspose1d, StreamingConv1d
from ..modules.streaming import StreamingContainer
from ..modules.transformer import ProjectedTransformer
from ..modules.wavelet import UnPatcher1D, Patcher1D

from ..modules.transformer import StreamingTransformer, StreamingTransformerDecoderLayer, create_sin_embedding
from ..modules.streaming import StreamingModule, State, StateT
from ..utils.compile import no_compile, CUDAGraphed
from .compression import CompressionModel

from ..quantization import QuantizedResult, ResidualVectorQuantizer, ResidualFiniteScalarQuantizer, SplitResidualVectorQuantizer


@dataclass
class _GestureMimiCodecState(State):
    graphed_tr_enc: CUDAGraphed | None
    graphed_tr_dec: CUDAGraphed | None
    graphed_conv_enc: CUDAGraphed | None
    graphed_conv_dec: CUDAGraphed | None




class GestureMimiCodec(CompressionModel[_GestureMimiCodecState]):
    """Gesture codec model that uses SEANet for downsampling/upsampling and Transformer for processing.
    
    Architecture:
    - Encoder: SEANetEncoder -> ProjectedTransformer
    - Decoder: ProjectedTransformer -> SEANetDecoder
    - Quantizer: ResidualVQ or SplitResidualVQ
    
    Args:
        latent_dim (int): Dimension of the latent space.
        frame_chunk_size (int): Number of frames in each chunk.
        num_frames (int): Number of frames in the motion data.
        nfeats (int): Number of features in the motion data.
        causal (bool): Whether the model is causal.
        vq_args (dict): Arguments for the quantizer.
        freeze_encoder (bool): Whether to freeze the encoder.
        **kwargs: Additional arguments.
    """
    def __init__(
        self, 
        latent_dim: int = 512,
        frame_chunk_size: int = 16,
        num_frames: int = 250, 
        nfeats: int = 337,
        causal: bool = False,
        vq_args = None,
        freeze_encoder: bool = False,
        downsampling_ratios: list[int] = [5, 2],
        n_filters: int = 64,
        
        use_wavelet = False,
        latent_dtype = "float32",
        **kwargs) -> None:

        super().__init__()

        self.latent_dim = latent_dim
        self.frame_chunk_size = frame_chunk_size  #
        self.num_frames = num_frames
        self._motion_fps = kwargs.pop("motion_fps", 25)
        self._channels = nfeats
        self._frame_rate = self._motion_fps / self.frame_chunk_size
        self.causal = causal
        self.latent_dtype = torch.bfloat16 if latent_dtype == "bfloat16" else torch.float32

        # Transformer parameters
        num_heads = kwargs.pop("num_heads", 8)
        transformer_layers = kwargs.pop("transformer_layers", 4)
        convblock_layers = kwargs.pop("convblock_layers", 2)
        
        # The downsampling factor from input sequence to latent
        self.downsampling_ratios = downsampling_ratios  # 250 frames -> 25 frames
        nfeats = nfeats[0] if isinstance(nfeats, list) else nfeats
        
        # SEANet encoder for downsampling motion to latent sequence
        self.seanet_encoder = SEANetEncoder(
            channels=nfeats,
            dimension=latent_dim,
            n_filters=n_filters,
            n_residual_layers=convblock_layers,
            ratios=self.downsampling_ratios,  # 5×2=10 downsampling
            activation="ELU",
            norm="weight_norm",
            kernel_size=7,
            residual_kernel_size=3,
            dilation_base=2,
            causal=causal,
            pad_mode="constant",
            true_skip=True
        )
        
        # Transformer for processing latent sequence
        self.transformer_encoder = ProjectedTransformer(
            input_dimension=latent_dim,
            output_dimensions=[latent_dim],
            d_model=latent_dim,
            num_heads=num_heads,
            num_layers=transformer_layers,
            causal=causal,
            conv_layout=True,
            **kwargs
        )
        
        # Transformer decoder
        self.transformer_decoder = ProjectedTransformer(
            input_dimension=latent_dim,
            output_dimensions=[latent_dim],
            d_model=latent_dim,
            num_heads=num_heads,
            num_layers=transformer_layers,
            causal=causal,
            conv_layout=True,
            **kwargs
        )
        
        # SEANet decoder for upsampling back to motion
        self.seanet_decoder = SEANetDecoder(
            channels=nfeats,
            dimension=latent_dim,
            n_filters=n_filters,
            n_residual_layers=convblock_layers,
            ratios=self.downsampling_ratios,  # 5×2=10 upsampling
            activation="ELU",
            norm="weight_norm",
            kernel_size=7,
            residual_kernel_size=3,
            dilation_base=2,
            causal=causal,
            pad_mode="constant",
            true_skip=True
        )
        
        # Vector quantizer
        assert vq_args is not None, "VQ args must be provided"
        n_q_semantic = vq_args.pop("n_q_semantic", 0)
        quantizer_type = vq_args.pop("quantizer_type", "vq")
        if n_q_semantic > 0:
            assert quantizer_type == "vq", "Only SplitResidualVQ is supported for n_q_semantic > 0"
            self.quantizer = SplitResidualVectorQuantizer(n_q_semantic=n_q_semantic,
                                                         **vq_args)
        else:
            if quantizer_type == "vq":
                self.quantizer = ResidualVectorQuantizer(**vq_args)
            elif quantizer_type == "fsq":
                self.quantizer = ResidualFiniteScalarQuantizer(**vq_args)
            else:
                raise ValueError(f"Unknown quantizer type: {quantizer_type}")


        self.use_wavelet = use_wavelet
        if use_wavelet:
            self.in_wavlettransform = Patcher1D(patch_size=1, patch_method="haar")
            self.out_wavlettransform = UnPatcher1D(patch_size=1, patch_method="haar")
        
        # Freeze encoder if needed
        if freeze_encoder:
            for param in self.seanet_encoder.parameters():
                param.requires_grad = False
            for param in self.transformer_encoder.parameters():
                param.requires_grad = False

        # if not freeze_encoder:
        #     logger.info("Initializing weights for GestureMimiCodec.")
        #     self.init_weights()

    def _init_streaming_state(self, batch_size: int) -> _GestureMimiCodecState:
        device = next(self.parameters()).device
        disable = device.type != 'cuda'
        graphed_tr_enc = CUDAGraphed(self.transformer_encoder, disable=disable)
        graphed_tr_dec = CUDAGraphed(self.transformer_decoder, disable=disable)
        graphed_conv_enc = CUDAGraphed(self.seanet_encoder, disable=disable)
        graphed_conv_dec = CUDAGraphed(self.seanet_decoder, disable=disable)
        return _GestureMimiCodecState(
            batch_size, 
            device, 
            graphed_tr_enc=graphed_tr_enc, 
            graphed_tr_dec=graphed_tr_dec,
            graphed_conv_enc=graphed_conv_enc,
            graphed_conv_dec=graphed_conv_dec
        )
    
    # @property
    # def _context_for_encoder_decoder(self):
    #     if self.torch_compile_encoder_decoder:
    #         return nullcontext()
    #     else:
    #         return no_compile()
    
    @property
    def channels(self) -> int:
        return self._channels
    
    
    @property
    def motion_dim(self) -> int:
        return self._channels
    
    @property
    def frame_rate(self) -> float:
        return self._frame_rate
    
    @property
    def frame_size(self) -> int:
        return self.frame_chunk_size
    
    @property
    def motion_fps(self) -> int:
        return self._motion_fps
    
    @property
    def sample_rate(self) -> int:
        return self.motion_fps
    
    @property
    def num_codebooks(self):
        """Number of quantizer codebooks available."""
        return self.quantizer.num_codebooks
    
    @property
    def cardinality(self):
        """Cardinality of each codebook."""
        return self.quantizer.cardinality
    
    @property
    def total_codebooks(self) -> int:
        """Total number of codebooks."""
        return self.quantizer.num_codebooks
    
    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        raise NotImplementedError()
    
    def _encode_to_unquantized_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input motion sequence to unquantized latent sequence.
        
        Args:
            x: Input tensor of shape [batch, time, features].
            
        Returns:
            Latent tensor of shape [batch, latent_dim, time//10].
        """
        state = self._streaming_state
        
        # Reshape for SEANet encoder: [batch, features, time]
        x = x.transpose(1, 2)
        
        if self.use_wavelet:
            x = self.in_wavlettransform(x)

        # SEANet encoding (downsampling): [batch, latent_dim, time//10]
        z = self.seanet_encoder(x)
        
        # Reshape for transformer: [batch, time//10, latent_dim]
        # z = z.transpose(1, 2)
        
        # Process with transformer
        if state is None:
            (z,) = self.transformer_encoder(z)
        else:
            assert state.graphed_tr_enc is not None
            (z,) = state.graphed_tr_enc(z)

        z = z.to(self.latent_dtype)
        
        return z
    
    def _cast_for_module(self, x: torch.Tensor, module: nn.Module) -> torch.Tensor:
        # Find a dtype from params or buffers; default to float32 if none found
        target_dtype = None
        for t in module.parameters(recurse=True):
            target_dtype = t.dtype
            break
        if target_dtype is None:
            for t in module.buffers():
                target_dtype = t.dtype
                break
        if target_dtype is None:
            target_dtype = torch.float32
        return x if x.dtype == target_dtype else x.to(target_dtype)
    
    def forward(self, x: torch.Tensor):
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch, time, features].
            
        Returns:
            Reconstructed tensor of shape [batch, time, features],
            quantization results,
            and loss mask.
        """
        # Encode
        z = self._encode_to_unquantized_latent(x)
        
        z_e = z.clone()
        # Quantize
        z_for_q = self._cast_for_module(z, self.quantizer)
        q_res = self.quantizer(z_for_q, self.frame_rate)
        z_q = q_res.x
        
        # Decode
        x_hat = self._decode_base(z_q)
        
        # Full time mask (no masking)
        loss_timemask = torch.ones(x.shape[1], device=x.device)
        return x_hat, q_res, z_e, loss_timemask
    
    def _decode_base(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent sequence to motion sequence.
        
        Args:
            z: Latent tensor of shape [batch, latent_dim, time//10].
            
        Returns:
            Decoded tensor of shape [batch, time, features].
        """
        state = self._streaming_state

        # z = z.to(torch.float32)
        
        # Process with transformer decoder
        if state is None:
            (z,) = self.transformer_decoder(z)
        else:
            assert state.graphed_tr_dec is not None
            (z,) = state.graphed_tr_dec(z)
        
        # Reshape for SEANet decoder: [batch, latent_dim, time//10]
        # z = z.transpose(1, 2)
        
        # SEANet decoding (upsampling): [batch, features, time]
        # with self._context_for_encoder_decoder:
        x_hat = self.seanet_decoder(z)

        if self.use_wavelet:
            x_hat = self.out_wavlettransform(x_hat)
        
        # Reshape to output format: [batch, time, features]
        x_hat = x_hat.transpose(1, 2)
        
        return x_hat
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input to discrete codes.
        
        Args:
            x: Input tensor of shape [batch, time, features].
            
        Returns:
            Quantized codes.
        """
        z = self._encode_to_unquantized_latent(x)
        z_for_q = self._cast_for_module(z, self.quantizer)
        codes = self.quantizer.encode(z_for_q)
        return codes
    
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode discrete codes back to motion sequence.
        
        Args:
            codes: Quantized codes.
            
        Returns:
            Decoded tensor of shape [batch, time, features].
        """
        z_q = self.quantizer.decode(codes)
        x_hat = self._decode_base(z_q)
        return x_hat
    
    def encode_to_latent(self, x: torch.Tensor, quantize: bool = True) -> torch.Tensor:
        """Encode to latent representation, optionally with quantization.
        
        Args:
            x: Input tensor of shape [batch, time, features].
            quantize: Whether to quantize the latent representation.
            
        Returns:
            If quantize=True, returns (codes, quantized_latents)
            If quantize=False, returns unquantized latents
        """
        z = self._encode_to_unquantized_latent(x)
        if not quantize:
            return z
        else:
            z_for_q = self._cast_for_module(z, self.quantizer)
            codes = self.quantizer.encode(z_for_q)
            return codes, self.decode_latent(codes)
    
    def encode_ze_to_codes(self, z_e: torch.Tensor) -> torch.Tensor:
        z_for_q = self._cast_for_module(z_e, self.quantizer)
        codes = self.quantizer.encode(z_for_q)
        return codes, self.decode_latent(codes)

    def decode_latent(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode from discrete codes to continuous latent space.
        
        Args:
            codes: Quantized codes.
            
        Returns:
            Continuous latent representation.
        """
        return self.quantizer.decode(codes)
    
    def decode_to_vqlatent(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode discrete codes to quantized latent representation.
        
        Args:
            codes: Quantized codes.
            
        Returns:
            Quantized latent representation.
        """
        return self.quantizer.decode_to_latent(codes)
    
    def reconstuct_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct latent representation from encoder through quantization and decoding.
        
        Args:
            z: Unquantized latent tensor.
            
        Returns:
            reconstructed motion.
        """
        z_for_q = self._cast_for_module(z, self.quantizer)
        q_res = self.quantizer(z_for_q, self.frame_rate)
        z_q = q_res.x
        
        # Decode
        x_hat = self._decode_base(z_q)
        return x_hat

    def init_weights(self):
        """Initialize weights of the model."""
        for m in self.modules():
            
            if isinstance(m, nn.Linear):
                if m.weight is not None and m.weight.dim() >= 2:
                    nn.init.xavier_uniform_(m.weight)
                elif m.weight is not None:
                    # For 1D weights, use a simpler initialization
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                if m.weight is not None and m.weight.dim() >= 2:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif m.weight is not None:
                    # For 1D weights, use a simpler initialization
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

