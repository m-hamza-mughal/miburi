from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import partial
import logging
import typing as tp
import torch
from torch import nn

from ..utils.sampling import sample_token
from ..utils.compile import CUDAGraphed
from ..utils.quantize import replace_linear_with_qlinear
from ..modules.streaming import StreamingContainer, StreamingModule, State
from ..modules.transformer import StreamingTransformer, create_norm_fn, StreamingTransformerDecoderLayer
from .lm_utils import (_init_layer,
                       ScaledEmbedding,
                       ScaledEmbeddingwithPadEmbedding)


logger = logging.getLogger(__name__)


class GTemporalDepthModel3(StreamingContainer):
    def __init__(
        self,
        n_q: int = 16,
        card: int = 512,
        dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        hidden_scale: int = 4,
        query2mem_scale: int = 5,
        num_temp_classifiers: int = 1,

        norm: str = "layer_norm",
        norm_emb: bool = False,
        bias_proj: bool = False,
        cond_dim: int = 512,

        context: tp.Optional[int] = 25,
        memory_context: tp.Optional[int] = 125,
        causal: bool = True,

        depformer_heads: int = 8,
        depformer_layers: int = 4,
        depformer_dim: int = 256,
        depformer_dim_feedforward: int | list[int] | None = None,
        depformer_multi_linear: bool = False,
        depformer_weights_per_step: bool = False,
        depformer_low_rank_embeddings: int | None = None,
        depformer_pos_emb: str = "sin",
        
        quantize: bool = False,
        device=None,
        dtype=None,
        gradient_checkpointing: bool = False,

        text_procemb: ScaledEmbedding | None = None,
        audio_procemb: ScaledEmbedding | None = None,
        textaudio_emb_freeze: bool = False,
        gesture_codec_layers: nn.ModuleList | None = None,

        body_parts: int = 2,
        bp_dist: list[int]| None = [0]*8 + [1]*4 + [2]*8,  # Example for 20 codebooks with 3 body parts 8 upper, 4 face, 8 lower # len is K

        vad_guidance: bool = False,
        vad_use_face_logits: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.n_q = n_q
        self.card = card
        self.dim = dim
        self.context = context
        self.causal = causal
        self.query2mem_scale = query2mem_scale
        self.gesture_codec_layers = gesture_codec_layers
        
        assert len(gesture_codec_layers) == n_q, f"Expected {n_q} gesture codec layers, got {len(gesture_codec_layers)}"
        
        num_audioemb, dim_audioemb = audio_procemb[0].shape
        self.audio_procemb = nn.ModuleList([
            ScaledEmbedding(
                num_embeddings=num_audioemb,
                embedding_dim=dim_audioemb,
                norm=False,
                device=device,
                dtype=dtype,
                zero_idx=-1,
                _weight=audio_procemb[embi],
                _freeze=textaudio_emb_freeze,
            ) for embi in range(len(audio_procemb))
        ])
        num_textemb, dim_textemb = text_procemb.shape
        self.text_procemb = ScaledEmbedding(
            num_embeddings=num_textemb,
            embedding_dim=dim_textemb,
            norm=False,
            device=device,
            dtype=dtype,
            zero_idx=-1,
            _weight=text_procemb,
            _freeze=textaudio_emb_freeze
        )


        EmbeddingFactory = partial(
            ScaledEmbedding,
            norm=norm_emb,
            device=device,
            dtype=dtype,
            zero_idx=self.initial_token_id,
        )
        PadEmbeddingFactory = partial(
            ScaledEmbeddingwithPadEmbedding,
            norm=norm_emb,
            device=device,
            dtype=dtype,
            zero_idx=self.initial_token_id,
            pad_idx=self.pad_token_id,
        )
        self.temporal_gemb = nn.ModuleList(
            [PadEmbeddingFactory(
                self.card, 
                gesture_codec_layers[i]._codebook.dim,
                _weight=gesture_codec_layers[i]._codebook.embedding,
                _freeze=False,
            ) for i in range(n_q)] # self.card
        )
        
        self.temporal_gproj = nn.ModuleList(
            [nn.Linear(gesture_codec_layers[i]._codebook.dim, dim, bias=False) for i in range(n_q)]
        )
        self.num_spks = 100+1
        self.spk_tempemb = EmbeddingFactory(self.num_spks, dim)

        depformer_prefix = "depformer_"
        main_kwargs = {
            k: v for k, v in kwargs.items() if not k.startswith(depformer_prefix)
        }
        self.temporal_transformer = StreamingTransformer(
            d_model=dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=int(hidden_scale * dim),
            norm=norm,
            device=device,
            dtype=dtype,
            quantize=quantize,
            context=context, # 125 for temporal
            memory_context=memory_context, # 125 for temporal
            upsample_factor=query2mem_scale, # 1
            causal=causal,
            crossattn_causal=causal,
            checkpointing=gradient_checkpointing,
            layer_class=StreamingTransformerDecoderLayer,
            num_memories=2,
            **main_kwargs,
        )
        self.out_norm = create_norm_fn(norm, dim)

        
        assert num_temp_classifiers == 1,"for this model num_temp_classifiers should be 1"
        self.num_temp_classifiers = num_temp_classifiers
        self.temporal_classifier = nn.Linear(dim, self.card + 1, bias=bias_proj)
        
        self.depformer_multi_linear = depformer_multi_linear
        
        kwargs_dep = main_kwargs.copy()
        kwargs_dep.update(
            {
                k.removeprefix(depformer_prefix): v
                for k, v in kwargs.items()
                if k.startswith(depformer_prefix)
            }
        )
        kwargs_dep["positional_embedding"] = depformer_pos_emb
        # kwargs_dep["context"] = None
        if depformer_weights_per_step:
            kwargs_dep["weights_per_step"] = n_q-1

        
        
        if depformer_multi_linear:
            # One linear layer per codebook to project different informations from the main model.
            num_in = n_q - 1
            self.depformer_in = nn.ModuleList(
                [nn.Linear(dim, depformer_dim, bias=False) for _ in range(num_in)]
            )
        else:
            self.depformer_in = nn.ModuleList(
                [nn.Linear(dim, depformer_dim, bias=False)]
            )
        
        
        EmbeddingFactory = partial(EmbeddingFactory, low_rank=depformer_low_rank_embeddings)
        PadEmbeddingFactory = partial(PadEmbeddingFactory, low_rank=depformer_low_rank_embeddings)

        self.body_parts = body_parts
        if bp_dist is not None:
            self.body_part_emb = EmbeddingFactory(body_parts, depformer_dim)
            self.bp_dist = torch.tensor(bp_dist, dtype=torch.long)
            assert len(self.bp_dist) == n_q, f"Expected body part distance list of length {n_q}, got {len(self.bp_dist)}"
        else:
            self.body_part_emb = None
            self.bp_dist = None

        self.spk_depemb = nn.ModuleList(
            [EmbeddingFactory(self.num_spks, depformer_dim) for _ in range(n_q-1)]
        ) 
        
        self.depformer_gemb = nn.ModuleList(
            [PadEmbeddingFactory(
                self.card, 
                gesture_codec_layers[i]._codebook.dim,
                _weight=gesture_codec_layers[i]._codebook.embedding,
                _freeze=False,
            ) for i in range(0, n_q-1)] # self.card
        )
        self.depformer_gproj = nn.ModuleList(
            [nn.Linear(gesture_codec_layers[i]._codebook.dim, depformer_dim, bias=False) for i in range(0, n_q-1)]
        )
        if depformer_dim_feedforward is None:
            depformer_dim_feedforward = int(hidden_scale * depformer_dim)
        self.depth_transformer = StreamingTransformer(
            d_model=depformer_dim,
            num_heads=depformer_heads,
            num_layers=depformer_layers,
            dim_feedforward=depformer_dim_feedforward,
            norm=norm,
            device=device,
            dtype=dtype,
            context=n_q-1, # 20 for depth - 1 because of the init from temporal = 19
            memory_context=query2mem_scale, # 1 for depth
            upsample_factor=1, # no upsampling for depth since it is not causal
            quantize=quantize,
            causal=causal,
            crossattn_causal=False, # non-causal for depth because query2mem_scale == 1
            checkpointing=gradient_checkpointing,
            layer_class=StreamingTransformerDecoderLayer,
            num_memories=2,
            **kwargs_dep,
        )
        
        self.depformer_classifier = nn.ModuleList(
            [nn.Linear(depformer_dim, self.card + 1, bias=bias_proj) for _ in range(n_q-1)]
        )
        # Depformer follow its own cycle of streaming entirely contained in one time step
        # and should not follow the streaming of the steps dimensions.
        self.depth_transformer.set_streaming_detached(True)

        # --- condition specific projection layers

        

        self.cond_dim = cond_dim
        self.temp_condproj = nn.ModuleList(
            [nn.Linear(self.cond_dim, self.dim) for _ in range(2)]
        )
        self.dep_condproj = nn.ModuleList(
            [nn.Linear(self.cond_dim, depformer_dim) for _ in range(2)]
        )

        self.vad_guidance = vad_guidance
        # Optional toggle: feed the last 4 face logits (flattened) into the
        # VAD predictor alongside transformer_out. Only meaningful when
        # vad_guidance is also on. Default off; existing pipeline unchanged.
        self.vad_use_face_logits = bool(vad_use_face_logits) and self.vad_guidance
        if vad_use_face_logits and not self.vad_guidance:
            logger.warning(
                "vad_use_face_logits=True ignored because vad_guidance=False"
            )
        if self.vad_guidance:
            # The temporal + depformer classifiers each emit `card + 1` logits
            # (see self.temporal_classifier / self.depformer_classifier below).
            # When vad_use_face_logits is on we down-project the 4 face
            # codebooks' flattened logits (4 * (card + 1)) to a single
            # codebook's width (card + 1) before concatenating with
            # transformer_out, so the predictor input stays compact.
            face_logit_dim = card + 1
            if self.vad_use_face_logits:
                self.vad_face_proj = nn.Linear(4 * face_logit_dim, face_logit_dim)
                vad_in_dim = dim + face_logit_dim
            else:
                self.vad_face_proj = None
                vad_in_dim = dim
            self.vad_predictor = nn.Sequential(
                nn.LayerNorm(vad_in_dim),
                nn.Linear(vad_in_dim, 256),
                nn.GELU(),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, 256),
                nn.GELU(),
                nn.Linear(256, 1),
            )
        else:
            self.vad_predictor = None
            self.vad_face_proj = None

        # ---

        self.to(device=device, dtype=dtype)
        self._init_weights()
        if quantize:
            replace_linear_with_qlinear(self)
    
    @property
    def initial_token_id(self) -> int:
        """Token id for the start of sequence (gesture)."""
        return -1
    
    @property
    def pad_token_id(self) -> int:
        """Token id for padding. Should be different from initial_token_id."""
        return self.card
    
    @property
    def ungenerated_token_id(self) -> int:
        """Special value that can be provided in the prompt to indicate that this specific
        value should be predicted and sampled. This allows for partial teacher forcing, by generating
        one modality, with the other one fixed.
        """
        return -2
    
    def _get_initial_token(self) -> torch.Tensor:
        # Returns the initial token that will be fed to the model to predict the very first timestep.
        # The output shape will be [1, 1, 1].
        device = next(iter(self.parameters())).device
        # zero = torch.full(
        #     [1, 1, 1], -1, device=device, dtype=torch.long
        # )
        init_token = torch.full(
            [1, 1, 1], self.initial_token_id, device=device, dtype=torch.long
        )
        return init_token
    
        
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
    
    
    def forward(
            self, 
            codes: torch.Tensor,
            audio_codes: torch.Tensor,
            text_codes: torch.Tensor,
            sum_condition: torch.Tensor | None = None,
            ca_depth_padding_mask: torch.Tensor | None = None,
            ) -> torch.Tensor:
        """
        """
        B, K, T = codes.shape
        if ca_depth_padding_mask is not None:
            assert ca_depth_padding_mask.shape == (B, K, T), f"Expected ca_depth_padding_mask shape {(B, K, T)}, got {ca_depth_padding_mask.shape}."
        assert K == self.n_q, f"Expected {self.n_q} codebooks, got {K}."
        initial = self._get_initial_token().expand(B, K, -1)
        
        # 
        input_sequence = torch.cat([initial, codes], dim=2) # [B, K, T+1]

        
        # sum_condition is of shape [B,]
        temporal_sum_condition = sum_condition.unsqueeze(1).expand(-1, T) # [B, T]

        # process audio and text conditions
        audio_condition, text_condition = self.process_conditions(audio_codes, text_codes) # B x K=1 x T=125 x dim
            
        # reshape audio and text codes 
        audio_condition = audio_condition.squeeze(1) # B x T=125 x dim
        text_condition = text_condition.squeeze(1) # B x T=125 dim

        transformer_out, temp_logits = self.forward_temporal(
            input_sequence[:, :, :-1], # [B, K, T]
            audio_condition=audio_condition,
            text_condition=text_condition,
            sum_condition=temporal_sum_condition,
        )
        # print(temp_logits.shape)
        
        assert transformer_out.shape[0] == input_sequence.shape[0]
        assert transformer_out.shape[1] == input_sequence.shape[2] - 1
        # dep_initial = self._get_initial_token().expand(B, -1, T)
        # dep_initial = input_sequence[:, :, 1:]
        dep_inpseq = input_sequence[:, :, 1:] # [B, K, T]
        depth_padding_mask = (dep_inpseq == self.pad_token_id)

        dep_sum_condition = temporal_sum_condition # [B, T]
        depth_logits = self.forward_depth_training(
            dep_inpseq[:, :-1, :], # [B, K-1, T]
            transformer_out=transformer_out,
            audio_condition=audio_condition,
            text_condition=text_condition,
            sum_condition=dep_sum_condition,
            depth_padding_mask=depth_padding_mask[ :, :-1, :], # [B, K-1, T]
            ca_query_padding_mask=ca_depth_padding_mask[:, 1:, :] if ca_depth_padding_mask is not None else None, # [B, K-1, T] # for cross attention, excluding first upper token
        )
        # temp_logits : [B, 1, T, card]
        # depth_logits : [B, K-1, T, card]
        combined_logits = torch.cat([temp_logits, depth_logits], dim=1)

        if self.training and self.vad_guidance:
            if self.vad_use_face_logits:
                # Last 4 codebooks = face. Flatten (4 codebooks, card+1 logits
                # each) to (4 * (card + 1)) per timestep, then down-project to
                # (card + 1) so the vad_predictor input stays compact.
                assert combined_logits.shape[1] >= 4, (
                    f"vad_use_face_logits requires K>=4, got K={combined_logits.shape[1]}"
                )
                face_logits = combined_logits[:, -4:, :, :]  # [B, 4, T, card+1]
                face_logits_flat = face_logits.permute(0, 2, 1, 3).reshape(
                    face_logits.shape[0], face_logits.shape[2], -1,
                )  # [B, T, 4*(card+1)]
                face_logits_proj = self.vad_face_proj(face_logits_flat)  # [B, T, card+1]
                vad_input = torch.cat([transformer_out, face_logits_proj], dim=-1)
            else:
                vad_input = transformer_out
            vad_logits = self.vad_predictor(vad_input).squeeze(-1)  # [B, T]
            return combined_logits, vad_logits

        return combined_logits

    def forward_temporal(
            self, 
            sequence: torch.Tensor,
            audio_condition: torch.Tensor,
            text_condition: torch.Tensor,
            sum_condition: torch.Tensor | None = None,
        ):
        """
        Args:
            codes (torch.Tensor): Input codes of shape [B, K, T].
            condition_tensors (ConditionTensors): Condition tensors for the model.
        """
        
        B, K, T = sequence.shape
        assert K == self.n_q, f"Expected {self.n_q} codebooks, got {K}."
        input_sequence = sequence
        input_ = None
        for cb_index in range(self.n_q):
            gesture_cbemb = self.temporal_gemb[cb_index](input_sequence[:, cb_index, :])
            gesture_cbemb = self.temporal_gproj[cb_index](gesture_cbemb)
            input_ = gesture_cbemb if input_ is None else input_ + gesture_cbemb
        
        # input is [B, T, D]
        # sum_condition is [B, T] which is the spk input
        if sum_condition is not None:
            sum_condition = self.spk_tempemb(sum_condition) # [B, T, D]
            input_ = input_ + sum_condition.to(input_)
        
        # conditions for cross attention
        audio_emb = audio_condition.to(input_) # B x S=125 x dim
        text_emb = text_condition.to(input_) # B x S=125 x dim
        audio_emb = self.temp_condproj[0](audio_emb)
        text_emb = self.temp_condproj[1](text_emb)
        condition_tensors = [audio_emb, text_emb]

        transformer_out = self.temporal_transformer(
            input_,
            memories=condition_tensors,
        )
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)
        assert isinstance(transformer_out, torch.Tensor)

        
        logits = self.temporal_classifier(transformer_out)
        logits = logits.unsqueeze(1) # [B, 1, T, card]
        return transformer_out, logits


    
    def forward_depth_training(
            self, 
            codes: torch.Tensor, 
            transformer_out: torch.Tensor,
            audio_condition: torch.Tensor,
            text_condition: torch.Tensor,
            sum_condition: torch.Tensor | None = None,  
            depth_padding_mask: torch.Tensor = None,   
            ca_query_padding_mask: torch.Tensor = None       
        ):
        
        """
        Args:
            codes (torch.Tensor): Input codes of shape [B, K-1, T].
            condition_tensors (ConditionTensors): Condition tensors for the model.
        """
        B, Km1, T = codes.shape
        assert Km1 == self.n_q-1, f"Expected {self.n_q-1} depth inputs, got {Km1}."
        cond_seq = audio_condition.shape[1]
        assert cond_seq == T * self.query2mem_scale, \
            f"Expected {T * self.query2mem_scale} for audio condition, got {cond_seq}."
        
        
        depformer_inputs = []
        for cb_index in range(Km1):
            if self.depformer_multi_linear:
                transformerout_cb = self.depformer_in[cb_index](transformer_out)
            else:
                transformerout_cb = self.depformer_in[0](transformer_out)
            
            depformer_cbemb = self.depformer_gemb[cb_index](codes[:, cb_index, :])
            depformer_cbemb = self.depformer_gproj[cb_index](depformer_cbemb)

            depformer_inputs.append(depformer_cbemb + transformerout_cb)
        depformer_inputs = torch.stack(depformer_inputs, dim=2) # [B, T, K-1, D]
        depformer_inputs = depformer_inputs.view(B * T, Km1, -1) # [B*T, K-1, D]

        # create padding mask for depth transformer
        
        assert depth_padding_mask is not None, "depth_padding_mask should be provided"
        depth_padding_mask = depth_padding_mask.permute(0, 2, 1).reshape(B * T, Km1) # [B*T, K-1]
        if ca_query_padding_mask is not None:
            assert ca_query_padding_mask.shape == (B, Km1, T), f"Expected ca_query_padding_mask shape {(B, Km1, T)}, got {ca_query_padding_mask.shape}."
            ca_query_padding_mask = ca_query_padding_mask.permute(0, 2, 1).reshape(B * T, Km1) # [B*T, K-1]

        # condition_projections B x S=125 x cond_dim
        audio_emb = audio_condition.reshape(B, T, self.query2mem_scale, self.cond_dim)
        text_emb = text_condition.reshape(B, T, self.query2mem_scale, self.cond_dim)
        audio_emb = audio_emb.view(B * T, self.query2mem_scale, -1).to(depformer_inputs.dtype)
        text_emb = text_emb.view(B * T, self.query2mem_scale, -1).to(depformer_inputs.dtype)
        audio_emb = self.dep_condproj[0](audio_emb)
        text_emb = self.dep_condproj[1](text_emb)
        condition_tensors = [audio_emb, text_emb]

        if sum_condition is not None:
            sum_condition_per_K = []
            for cb_index in range(Km1):
                scb = self.spk_depemb[cb_index](sum_condition) # [B, T, D]
                sum_condition_per_K.append(scb)
            sum_condition_per_K = torch.stack(sum_condition_per_K, dim=2) # [B, T, K-1, D]
            sum_condition_per_K = sum_condition_per_K.view(B * T, Km1, -1) # [B*T, K-1, D]
            sum_condition = sum_condition_per_K

            depformer_inputs = depformer_inputs + sum_condition.to(depformer_inputs)

        if self.body_part_emb is not None:
            body_part_indices = self.bp_dist[1:] 
            body_part_indices = body_part_indices.unsqueeze(0).to(depformer_inputs.device)  # [1, Km1]
            body_part_emb = self.body_part_emb(body_part_indices)  # [1, Km1, D]
            body_part_emb = body_part_emb.expand(B * T, -1, -1)  # [B*T, Km1, D]
            depformer_inputs = depformer_inputs + body_part_emb.to(depformer_inputs)
            
        
        depformer_out = self.depth_transformer(
            depformer_inputs,
            memories=condition_tensors,
            key_padding_mask=depth_padding_mask,
            ca_query_padding_mask=ca_query_padding_mask,
        )
        all_logits = []
        for cb_index in range(Km1):
            logits = self.depformer_classifier[cb_index](depformer_out[:, cb_index])
            all_logits.append(logits.reshape(B, T, -1))
        all_logits = torch.stack(all_logits, dim=1) # [B, K-1, T, card]
        return all_logits
    

    def forward_depth(
            self, 
            cb_index: int,
            code: torch.Tensor, 
            transformer_out: torch.Tensor,
            audio_condition: torch.Tensor,
            text_condition: torch.Tensor,
            sum_condition: torch.Tensor | None = None, 
            ca_query_padding_mask: torch.Tensor | None = None,
            bp_dist: torch.Tensor = None,
        ) -> torch.Tensor:

        """
        Args:
            code (torch.Tensor): Input codes of shape [B, K, T].
            condition_tensors (ConditionTensors): Condition tensors for the model.
            transformer_out: [B, T, dim]
            ca_query_padding_mask: [B, K, T] # for cross attention in depth transformer
        """
        assert not self.training, "generation shouldn't be used in training mode."
        B, K, T = code.shape
        assert K == 1, "code input to depth model should be passed 1 by 1"
        assert T == 1, "code input to depth model should be passed 1 by 1"
        assert transformer_out.shape[1] == 1, "transformer_out should be passed 1 by 1"
        last_token_input: tp.Optional[torch.Tensor] = None
        depformer_input = transformer_out

        if self.depformer_multi_linear:
            depformer_input = self.depformer_in[cb_index](transformer_out)
        else:
            depformer_input = self.depformer_in[0](transformer_out)

        # if cb_index == 0:
        #     last_token_input = self.depinit_emb(code[:, 0])
        # else:
        #     last_token_input = self.depformer_emb[cb_index-1](code[:, 0])
        last_token_input = self.depformer_gemb[cb_index](code[:, 0])
        last_token_input = self.depformer_gproj[cb_index](last_token_input)

        assert last_token_input is not None
        depformer_input = depformer_input + last_token_input
        assert depformer_input.shape[1] == 1, "depformer_input should be passed 1 by 1"

        if ca_query_padding_mask is not None:
            assert ca_query_padding_mask.shape == (B, K, T), f"Expected ca_query_padding_mask shape {(B, K, T)} with K=1, got {ca_query_padding_mask.shape}."
            ca_query_padding_mask = ca_query_padding_mask[:, :, 0] # [B, 1, 1] -> [B, 1]

        # condition_projections B x S=5 x cond_dim  
        audio_emb = audio_condition
        text_emb = text_condition
        assert audio_emb.shape[0] == B and audio_emb.shape[1] == T * self.query2mem_scale, \
            f"Expected {B}x{T * self.query2mem_scale} for audio condition, got {audio_emb.shape[0]}x{audio_emb.shape[1]} for audio condition."
        assert text_emb.shape[0] == B and text_emb.shape[1] == T * self.query2mem_scale, \
            f"Expected {B}x{T * self.query2mem_scale} for text condition, got {text_emb.shape[0]}x{text_emb.shape[1]} for text condition."
        assert audio_emb.shape[2] == self.cond_dim and text_emb.shape[2] == self.cond_dim, \
            f"Expected {self.cond_dim} dim for audio condition, got {audio_emb.shape[2]} and {self.cond_dim} for text condition, got {text_emb.shape[2]}."
        audio_emb = audio_emb.to(depformer_input.dtype)
        text_emb = text_emb.to(depformer_input.dtype)
        audio_emb = self.dep_condproj[0](audio_emb)
        text_emb = self.dep_condproj[1](text_emb)
        condition_tensors = [audio_emb, text_emb]

        if sum_condition is not None:
            assert sum_condition.shape[0] == B and sum_condition.shape[1] == T, \
                f"Expected {B}x{T} for sum_condition, got {sum_condition.shape[0]}x{sum_condition.shape[1]}."
            scb = self.spk_depemb[cb_index](sum_condition) # [B, T, D] # should be 1, 1, D
            depformer_input = depformer_input + scb.to(depformer_input) 

        if self.body_part_emb is not None:
            body_part_emb = self.body_part_emb(bp_dist[cb_index]).unsqueeze(0).unsqueeze(0)  # [1, 1, D]
            body_part_emb = body_part_emb.expand(B*T, -1, -1)  # [B*T, 1, D]
            depformer_input = depformer_input + body_part_emb.to(depformer_input)

        depformer_out = self.depth_transformer(
            depformer_input,
            memories=condition_tensors,
            ca_query_padding_mask=ca_query_padding_mask,
        ) # [B, T=1, D]
        logits = self.depformer_classifier[cb_index](depformer_out).unsqueeze(1) # B x K=1 x T=1 x card
        return logits
    
    

    def _init_weights(self):
        """Initialization of the transformer module weights.
        Mostly truncated gaussian, with `std = 1 / sqrt(dim_in)`.
        Embeddings are also initialized with `1 / sqrt(dim)` rather than `1`.
        Some layers are not going to be properly initialized:
            - in_proj in MHA.
        This is to match how our models were trained so far.
        """

        for emb_layer in self.temporal_gproj:
            _init_layer(emb_layer)
        for emb_layer in self.depformer_gproj:
            _init_layer(emb_layer)
        # _init_layer(self.depinit_emb)
        for emb_layer in self.temporal_gemb:
            _init_layer(emb_layer.pad_embedding)
        for emb_layer in self.depformer_gemb:
            _init_layer(emb_layer.pad_embedding)

        for tr_layer in self.temporal_transformer.layers:
            tr_layer.apply(_init_layer)
        for tr_layer in self.depth_transformer.layers:
            tr_layer.apply(_init_layer)

        _init_layer(self.temporal_classifier)
        for dep_linear in self.depformer_classifier:
            _init_layer(dep_linear)
        
        for dep_linear in self.depformer_in:
            _init_layer(dep_linear)
        for cond_linear in self.temp_condproj:
            _init_layer(cond_linear)
        for cond_linear in self.dep_condproj:
            _init_layer(cond_linear)




@dataclass
class _GLMGenState(State):
    cache: torch.Tensor
    initial: torch.Tensor
    graphed_temp: CUDAGraphed
    graphed_depth: CUDAGraphed
    condition_sum: torch.Tensor | None = None
    offset: int = 0
    exit_stack: ExitStack = field(default_factory=ExitStack)
    reset_callback: tp.Callable[[], None] | None = None

    def reset(self):
        self.offset = 0
        if self.reset_callback is not None:
            self.reset_callback()

    def __enter__(self):
        self.exit_stack.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit_stack.__exit__(exc_type, exc_value, traceback)


class GestureLMGen(StreamingModule[_GLMGenState]):
    def __init__(
        self,
        glm_model: GTemporalDepthModel3,
        use_sampling: bool = True,
        temp_gtemporal: float = 0.8,
        temp_gdepth: float = 0.7,
        top_k_gtemporal: int = 25,
        top_k_gdepth: int = 25,
        top_p_gtemporal: float = 1.0,
        top_p_gdepth: float = 1.0,
        cfg_coef: float = 1.,
        check: bool = False,
        condition_tensors: torch.Tensor | None = None,
    ):
        assert not glm_model.training, "generation shouldn't be used in training mode."
        super().__init__()

        self.glm_model = glm_model
        self.glm_model.set_streaming_detached(True)
        self.use_sampling = use_sampling
        self.temp_depth = temp_gdepth
        self.temp_temporal = temp_gtemporal
        self.top_k_temp = top_k_gtemporal
        self.top_k_depth = top_k_gdepth
        self.top_p_temp = top_p_gtemporal
        self.top_p_depth = top_p_gdepth
        self.cfg_coef = cfg_coef
        self.check = check
        
        self.condition_tensors = condition_tensors
        

        self.text_procemb = glm_model.text_procemb 
        self.audio_procemb = glm_model.audio_procemb 
        self.text_procemb = self.text_procemb.eval()
        self.audio_procemb = self.audio_procemb.eval()

        self.audio_codec_nulltoken = -1 # self.audio_procemb[0].num_embeddings - 1 # -1
        self.text_codec_nulltoken = -1 # self.text_procemb.num_embeddings - 1 # -1



        for p in glm_model.parameters():
            self.device = p.device
            break

        self.bp_dist = glm_model.bp_dist.to(self.device) if glm_model.bp_dist is not None else None

    def _init_streaming_state(self, batch_size: int) -> _GLMGenState:
        glm_model = self.glm_model
        initial = glm_model._get_initial_token()
        cache = torch.full(
            (batch_size, self.glm_model.n_q, 1),
            glm_model.ungenerated_token_id,
            device=self.device,
            dtype=torch.long,
        )

        condition_sum = self.condition_tensors.to(self.device) if self.condition_tensors is not None else None

        disable = self.device.type != 'cuda'
        graphed_temp = CUDAGraphed(glm_model.forward_temporal, disable=disable)
        graphed_depth = CUDAGraphed(self.depformer_step, disable=disable)

        state = _GLMGenState(
            batch_size, self.device, cache, initial, graphed_temp, graphed_depth,
            condition_sum=condition_sum
        )

        if self.cfg_coef != 1.:
            batch_size *= 2
            
        state.exit_stack.enter_context(self.glm_model.streaming(batch_size))
        state.reset_callback = self.glm_model.reset_streaming
        return state

    def process_conditions(self, audio_codes, text_codes):
        """
        Process the condition tensors using text_procemb and audio_procemb
        
        Args:
            audio_codes: B x K=8 x T=1
            text_codes: B x K=1 x T=1
        Returns:
            audio_codes: B x T=1 x dim 
            text_codes: B x T=1 x dim
        """
        audio_embs = []
        for k in range(audio_codes.shape[1]):
            audio_emb = self.audio_procemb[k](audio_codes[:, k, :])
            audio_embs.append(audio_emb)
        audio_emb = torch.stack(audio_embs, dim=1) # B x K=8 x T=1 x dim
        audio_emb = audio_emb.sum(dim=1, keepdim=True) # B x K=1 x T=1 x dim

        if text_codes is not None:
            text_emb = self.text_procemb(text_codes)
        else:
            text_emb = None


        audio_emb = audio_emb.squeeze(1) # B x T=1 x dim
        text_emb = text_emb.squeeze(1) # B x T=1 dim
        return audio_emb, text_emb

    @torch.no_grad()
    def step(self, condition: torch.Tensor | tp.List[torch.Tensor], ca_query_padding_mask: torch.Tensor | None = None
             ) -> tp.Tuple[torch.Tensor] | None:
        """
        step for the GTemporal depth model which takes in audio/text condition and generates
        the next gesture codebook token. 
        Args:
            condition: tensor of shape [B, K, S=1] for audio and text conditions
        """
        state = self._streaming_state
        if state is None:
            raise RuntimeError("Streaming state is not initialized.")
        glm_model = self.glm_model
        

        audio_emb = condition[:, 1:] # [B, 8, S=1]
        text_emb = condition[:, :1] # [B, 1, S=1]
        

        if self.cfg_coef != 1.:
            null_audio = torch.full_like(audio_emb, self.audio_codec_nulltoken)
            
            null_text = torch.full_like(text_emb, self.text_codec_nulltoken)
            audio_emb = torch.cat([audio_emb, null_audio], dim=0)
            text_emb = torch.cat([text_emb, null_text], dim=0)
        
        audio_emb, text_emb = self.process_conditions(audio_emb, text_emb) # [B, S=1, dim]
        
        
        B, condT, _ = audio_emb.shape
        assert audio_emb.dim() == 3 and text_emb.dim() == 3, \
            f"Expected 3D tensors for audio and text conditions, got {audio_emb.dim()} and {text_emb.dim()}."
        
        if self.cfg_coef == 1:
            assert B == state.batch_size, \
                f"Expected batch size {state.batch_size}, got {B}."
        else:
            assert B == state.batch_size * 2, \
                f"Expected batch size {state.batch_size * 2} for CFG, got {B}."
        
        assert condT == self.glm_model.query2mem_scale, \
            f"Expected condT {self.glm_model.query2mem_scale}, got {condT}."
        
        CT = state.cache.shape[2] # context from prev steps

        position = state.offset % CT # 
        if state.offset == 0:
            state.cache[:, :, position] = glm_model.initial_token_id # -1
        input_ = state.cache[:, :, position:position + 1] # [B, K, 1]
        # print(input_)

        if self.check:
            assert not (input_ == glm_model.ungenerated_token_id).any(), \
                f"Expected ungenerated token id {glm_model.ungenerated_token_id}, got {input_} at {state.offset}."
        
        # also check if temp transformer is streaming or not # check sum_condition shape
        sum_condition = state.condition_sum.expand(B, 1)
        # condition_tensors = [audio_emb, text_emb]
        if self.cfg_coef != 1.:
            # duplicate the batch for CFG
            input_ = input_.repeat(2, 1, 1)


        transformer_out, temp_logits = state.graphed_temp(
            input_,
            audio_emb,
            text_emb,
            sum_condition,
        )
        # temp_logits[..., glm_model.pad_token_id] = float('-inf')  # prevent sampling pad token

        if self.cfg_coef != 1.:
            # split the batch for CFG
            temp_logits, null_temp_logits = temp_logits.chunk(2, dim=0)
            # apply classifier-free guidance
            temp_logits = null_temp_logits + self.cfg_coef * (temp_logits - null_temp_logits) 

        uppertop_next_token = sample_token(
            temp_logits.float(),
            use_sampling=self.use_sampling,
            temp=self.temp_temporal,
            top_k=self.top_k_temp,  # ignored if top_p is set
            top_p=self.top_p_temp,
        )

        assert uppertop_next_token.dim() == 3, uppertop_next_token.shape
        assert uppertop_next_token.shape[2] == 1
        assert uppertop_next_token.shape[1] == 1, "Only one text stream supported."
        uppertop_next_token = uppertop_next_token[:, 0, 0]  # shape is [B]

        ca_pad_mask = ca_query_padding_mask[:, 1:, :] if ca_query_padding_mask is not None else None
        cfg_stop_mask = ca_pad_mask[0, :, 0].cpu().tolist() if (self.cfg_coef != 1. and ca_pad_mask is not None) else None
        bp_dist = self.bp_dist
        depth_tokens = state.graphed_depth(
            uppertop_next_token,
            transformer_out,
            audio_emb,
            text_emb,
            sum_condition,
            ca_pad_mask, # [B, K-1, 1] # for cross attention, excluding first upper token
            cfg_stop_mask,
            bp_dist,
        )
        # 
        state.offset += 1
        position = state.offset % CT
        state.cache[:, 0, position] = uppertop_next_token
        state.cache[:, 1:, position] = depth_tokens
        out = state.cache[:, :, position:position + 1]  # [B, K, 1]
        return out
    
    def depformer_step(
        self, 
        uppertop_next_token: torch.Tensor,
        transformer_out: torch.Tensor,
        audio_condition: torch.Tensor,
        text_condition: torch.Tensor,
        sum_condition: torch.Tensor | None = None,
        ca_query_padding_mask: torch.Tensor | None = None,
        cfg_stop_mask: list[bool] | None = None,
        bp_dist: torch.Tensor = None,
    ) -> torch.Tensor:
        B, = uppertop_next_token.shape
        B_cfg = B
        if self.cfg_coef != 1.:
            B_cfg = 2 * B
        prev_token = uppertop_next_token
        glm_model = self.glm_model
        depformer_tokens: list[torch.Tensor] = []
        assert not glm_model.depth_transformer.is_streaming
        with glm_model.depth_transformer.streaming(B_cfg):
            assert glm_model.depth_transformer.is_streaming
            for cb_index in range(glm_model.n_q-1):
                input_ = prev_token[:, None, None] # [B, 1, 1]
                if ca_query_padding_mask is not None:
                    ca_qpad_mask = ca_query_padding_mask[:, cb_index:cb_index+1, :]  
                    cfg_stop = cfg_stop_mask[cb_index] if cfg_stop_mask is not None else None
                else:
                    ca_qpad_mask = None, # [B, 1, 1] # for cross attention in depth transformer
                    cfg_stop = None
                
                if self.cfg_coef != 1.:
                    input_ = input_.repeat(2, 1, 1)
                    ca_qpad_mask = ca_qpad_mask.repeat(2, 1, 1) if ca_qpad_mask is not None else None
                
                logits = glm_model.forward_depth(
                    cb_index,
                    input_,
                    transformer_out=transformer_out,
                    audio_condition=audio_condition,
                    text_condition=text_condition,
                    sum_condition=sum_condition,
                    ca_query_padding_mask=ca_qpad_mask,
                    bp_dist=bp_dist,
                )
                # print(cb_index, cfg_stop)
                if self.cfg_coef != 1.:
                    logits, null_logits = logits.chunk(2, dim=0)
                    if not cfg_stop: # apply classifier-free guidance
                        logits = null_logits + self.cfg_coef * (logits - null_logits)
                
                logits[..., glm_model.pad_token_id] = float('-inf')  # prevent sampling pad token
                next_token = sample_token(
                    logits.float(),
                    use_sampling=self.use_sampling,
                    temp=self.temp_depth,
                    top_k=self.top_k_depth, # ignored if top_p is set
                    top_p=self.top_p_depth,

                )
                assert next_token.shape == (B, 1, 1), next_token.shape
                next_token = next_token[:, 0, 0]
                depformer_tokens.append(next_token)
                prev_token = next_token

        assert len(depformer_tokens) == glm_model.n_q-1, \
            f"Expected {glm_model.n_q-1} depth tokens, got {len(depformer_tokens)}."
        out = torch.stack(depformer_tokens, dim=1) # [B, K-1]
        assert out.shape == (B, glm_model.n_q-1), out.shape
        return out


