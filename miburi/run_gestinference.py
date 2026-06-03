# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random
import sys
import time
import copy


import numpy as np
import sentencepiece
import torch
import sphn
import os
import yaml
import smplx

from types import SimpleNamespace


from .client_utils import log, AnyPrinter, Printer, RawPrinter
from .conditioners import ConditionAttributes, ConditionTensors
from .models import loaders, MimiModel, LMModel, LMGen, GestureLMGen, GestureMimiCodec, GTemporalDepthModel3
from .utils.motion_utils import get_bodypart_masks, get_gesture_condition_tensors, inverse_selection_smplx, inverse_selection_tensor_smplx, get_smplx_bodypart_masks, velocity2position_mixeddiff
from .utils import rotation_conversions as rc

torch.set_float32_matmul_precision('medium')

def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_condition_tensors(model_type: str, lm: LMModel, batch_size: int, cfg_coef: float) -> ConditionTensors:
    condition_tensors = {}
    if lm.condition_provider is not None:
        conditions: list[ConditionAttributes] | None = None
        if model_type == 'hibiki':
            conditions = [ConditionAttributes(text={"description": "very_good"}, wav={})] * batch_size
            if cfg_coef != 1.:
                # Extending the conditions with the negatives for the CFG.
                conditions += [ConditionAttributes(text={"description": "very_bad"}, wav={})] * batch_size
        else:
            raise RuntimeError(f"Model expects conditioning but model type {model_type} is not supported.")
        assert conditions is not None
        prepared = lm.condition_provider.prepare(conditions)
        condition_tensors = lm.condition_provider(prepared)
    return condition_tensors


@dataclass
class InferenceState:
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    glm_gen: GestureLMGen
    gesture_codecs: list[GestureMimiCodec]

    def __init__(self, model_type: str, mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, gesturelm: GTemporalDepthModel3, gesture_codecs: list[GestureMimiCodec],
                 batch_size: int, cfg_coef: float, glm_cfg_coef: float, device: str | torch.device, **kwargs):
        self.model_type = model_type
        self.mimi = mimi
        self.gesture_codecs = gesture_codecs
        self.text_tokenizer = text_tokenizer
        condition_tensors = get_condition_tensors(model_type, lm, batch_size, cfg_coef)
        gesturelm_kwargs = kwargs.pop("gesturelm_kwargs")
        self.motion_mask = gesturelm_kwargs.pop("motion_mask", None)
        # self.motion_mask[0] = torch.from_numpy(self.motion_mask[0])
        # self.motion_mask[1] = torch.from_numpy(self.motion_mask[1])
        character_id = gesturelm_kwargs.pop("character_id")
        gesture_spk_condition = get_gesture_condition_tensors(character_id)
        self.lm_gen = LMGen(lm, cfg_coef=cfg_coef, condition_tensors=condition_tensors, **kwargs)
        # text_procemb = copy.deepcopy(lm.text_emb)
        # audio_procemb = copy.deepcopy(lm.emb)
        self.glm_gen = GestureLMGen(
            gesturelm,
            # condition_procemb=[text_procemb, audio_procemb],
            condition_tensors=gesture_spk_condition, 
            cfg_coef=glm_cfg_coef, # cfg_coef,
            **gesturelm_kwargs
        )
        
        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        

        self.codec_difference = int(self.mimi.frame_rate / self.gesture_codecs[0].frame_rate)
        self.motion_frame_size = self.gesture_codecs[0].frame_chunk_size
        self.motion_joints_per_codec = [43, 9, 1] # 43 for upper body, 9 for lower body, 1 for face
        self.batch_size = batch_size
        self.mimi.streaming_forever(batch_size)
        self.gesture_codecs[0].streaming_forever(batch_size) # upper body
        self.gesture_codecs[1].streaming_forever(batch_size) # lower body
        self.gesture_codecs[2].streaming_forever(batch_size) # face
        self.lm_gen.streaming_forever(batch_size)
        self.glm_gen.streaming_forever(batch_size)

        # prepare lm delay buffer for GLM
        # self.moshiout_numtokens = lm.num_codebooks - lm.dep_q
        # self.lm_delay_buffer = torch.zeros(
        #     (batch_size, self.moshiout_numtokens, self.lm_gen.max_delay), device=self.device
        # )
        # delayed_codes = [3, 1049, 1142, 1443,  249,  394,  763, 1829,  703]
        # for cb_index in range(self.moshiout_numtokens):
        #     if self.lm_gen.delays_cuda[cb_index] < self.lm_gen.max_delay:
        #         self.lm_delay_buffer[:, cb_index, :] = delayed_codes[cb_index]

        self.lower_cross_attn_mask = torch.zeros(
            self.batch_size, 
            self.gesture_codecs[0].num_codebooks + self.gesture_codecs[1].num_codebooks + self.gesture_codecs[2].num_codebooks,
            1, 
            device=self.device
        ).to(torch.bool) # B x K=16 x T

        # only apply cross-attention on upper and face tokens
        self.lower_cross_attn_mask[
                :,
                self.gesture_codecs[0].num_codebooks : self.gesture_codecs[0].num_codebooks + self.gesture_codecs[1].num_codebooks:,
                :] = True

        # ---- session-aux state, mirroring gest-server's ServerState ----
        # The full warmup (below) decodes through all three gesture codecs
        # and accumulates into these buffers, then `_reset_session_state`
        # wipes them before the real inference loop runs. This keeps the
        # warm-codec state primed without polluting the actual output.
        self.main_pcm_buffer = None
        self.mainpcm_buffer_size = 3 * self.frame_size  # 3 frames buffer
        self.motion_frame_buffer = None
        self.transl_frame_buffer = None
        self.final_pos = torch.zeros((self.batch_size)) + torch.tensor([0, 1.3, 0])

        self.printer: AnyPrinter
        if sys.stdout.isatty():
            self.printer = Printer()
        else:
            self.printer = RawPrinter()

    def _reset_session_state(self) -> None:
        """Wipe per-file auxiliary state so the next `run()` invocation
        starts with zero context. Mirrors `ServerState._reset_session_state`
        in gest-server.py. The model streaming caches are reset separately
        via `*.reset_streaming()` in `run()`."""
        if self.main_pcm_buffer is None:
            self.main_pcm_buffer = torch.zeros(
                1, 1, self.mainpcm_buffer_size, device=self.device,
            )
        else:
            self.main_pcm_buffer.zero_()
        self.motion_frame_buffer = None
        self.transl_frame_buffer = None
        self.final_pos = torch.zeros((self.batch_size)) + torch.tensor([0, 1.3, 0])


    def warmup(self) -> None:
        """Warmup the model by running a few steps with dummy data. Mirrors
        ServerState.warmup in gest-server.py: not just LM+GLM stepping, but
        also full gesture-codec decode + rot6d -> axis-angle + inverse-
        selection + final-pos integration, so the streaming state of every
        component is primed before the real inference loop starts."""
        for _ in range(4):
            audio_chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)

            audio_codes = self.mimi.encode(audio_chunk)

            for c in range(audio_codes.shape[-1]):
                moshi_tokens = self.lm_gen.step(audio_codes[:, :, c: c + 1])
                if moshi_tokens is None:
                    continue

                main_pcm = self.mimi.decode(moshi_tokens[:, 1:])

                if self.main_pcm_buffer is None:
                    self.main_pcm_buffer = main_pcm
                else:
                    self.main_pcm_buffer = torch.cat((self.main_pcm_buffer, main_pcm), dim=2)
                if self.main_pcm_buffer.shape[2] > self.mainpcm_buffer_size:
                    self.main_pcm_buffer = self.main_pcm_buffer[:, :, -self.mainpcm_buffer_size :]

                gmoshi_tokens = self.glm_gen.step(moshi_tokens, ca_query_padding_mask=self.lower_cross_attn_mask)
                assert moshi_tokens is not None and gmoshi_tokens is not None

                upper_tokens = gmoshi_tokens[:, :self.gesture_codecs[0].num_codebooks, :]
                lower_tokens = gmoshi_tokens[:, self.gesture_codecs[0].num_codebooks:self.gesture_codecs[0].num_codebooks + self.gesture_codecs[1].num_codebooks, :]
                face_tokens = gmoshi_tokens[:, self.gesture_codecs[0].num_codebooks + self.gesture_codecs[1].num_codebooks:, :]

                upper_motion = self.gesture_codecs[0].decode(upper_tokens)
                lowertrans_motion = self.gesture_codecs[1].decode(lower_tokens)
                faceexp_motion = self.gesture_codecs[2].decode(face_tokens)

                bs = lowertrans_motion.shape[0]
                uhj = self.motion_joints_per_codec[0]
                lj = self.motion_joints_per_codec[1]
                fj = self.motion_joints_per_codec[2]
                nframes = self.motion_frame_size

                upper_motion = upper_motion.reshape(bs, nframes, uhj, 6)
                upper_motion = rc.rotation_6d_to_matrix(upper_motion)
                upper_motion = rc.matrix_to_axis_angle(upper_motion).reshape(bs, nframes, (uhj)*3)

                lower_motion = lowertrans_motion[:, :, :lj*6]
                motion_transvel = lowertrans_motion[:, :, lj*6:lj*6+3]
                lower_motion = lower_motion.reshape(bs, nframes, lj, 6)
                lower_motion = rc.rotation_6d_to_matrix(lower_motion)
                lower_motion = rc.matrix_to_axis_angle(lower_motion).reshape(bs, nframes, lj*3)
                motion_trans, nextfinalpos = velocity2position_mixeddiff(
                    motion_transvel, 1/self.gesture_codecs[1].motion_fps, init_pos=self.final_pos
                )
                self.final_pos = nextfinalpos

                face_motion = faceexp_motion[:, :, :fj*6]
                face_exps = faceexp_motion[:, :, fj*6:]
                face_motion = face_motion.reshape(bs, nframes, fj, 6)
                face_motion = rc.rotation_6d_to_matrix(face_motion)
                face_motion = rc.matrix_to_axis_angle(face_motion).reshape(bs, nframes, fj*3)

                upper_motion = upper_motion[0].cpu()
                face_motion = face_motion[0].cpu()
                lower_motion = lower_motion[0].cpu()
                face_exps = face_exps[0].cpu()
                motion_trans = motion_trans[0].cpu()

                pred_motion_upper = inverse_selection_tensor_smplx(upper_motion, self.motion_mask[0], nframes)
                pred_motion_lower = inverse_selection_tensor_smplx(lower_motion, self.motion_mask[1], nframes)
                pred_motion_face = inverse_selection_tensor_smplx(face_motion, self.motion_mask[2], nframes)
                pred_motion_combined = pred_motion_upper + pred_motion_face + pred_motion_lower

                out_motion = torch.cat([pred_motion_combined, face_exps, motion_trans], axis=-1)

                if self.motion_frame_buffer is None:
                    self.motion_frame_buffer = out_motion
                else:
                    self.motion_frame_buffer = torch.cat((self.motion_frame_buffer, out_motion), dim=0)

        self.motion_frame_buffer = self.motion_frame_buffer[len(self.motion_frame_buffer):]
        torch.cuda.synchronize()


    def run(self, in_pcms: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Returns a list of tupel `(text_tokens, audio_tokens)`"""
        out_pcms_per_item: list[list[torch.Tensor]] = [[] for _ in range(self.batch_size)]
        out_text_tokens_per_item: list[list[torch.Tensor]] = [[] for _ in range(self.batch_size)]
        out_motion_per_item: list[list[torch.Tensor]] = [[] for _ in range(self.batch_size)]
        # For the Hibiki translation model, we feed a special token for the end of the input stream,
        # which corresponds to `2048` on all the codebooks of the audio stream, and wait
        # for the EOS on the output text stream to be emitted, as indication that the model is done.
        eos_reached: list[bool] = [False] * self.batch_size
        need_eos_input: bool = True
        self.printer.log("info", "starting the inference loop")
        device = self.lm_gen.lm_model.device
        start_time = time.time()
        ntokens = 0
        first_frame = True
        # We keep only fully frames.
        self.warmup()
        # Per-file zero-context restart, mirroring gest-server.py's reset
        # at WS-connect time: reset_streaming on every model component,
        # then wipe the session-aux buffers populated by warmup.
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()
        self.glm_gen.reset_streaming()
        self.gesture_codecs[0].reset_streaming()
        self.gesture_codecs[1].reset_streaming()
        self.gesture_codecs[2].reset_streaming()
        self._reset_session_state()
        chunks = deque([
            chunk for chunk in in_pcms.split(self.frame_size, dim=2)
            if chunk.shape[-1] == self.frame_size])
        self.printer.print_header()
        # counter = 0
        moshi_token_buffer = []
        g_step_time_average = 0.0
        mg_steptime_average = 0.0
        while not all(eos_reached):
            mg_tic = time.time()
            if chunks:
                chunk = chunks.popleft()
                codes = self.mimi.encode(chunk)
            else:
                if self.model_type == 'hibiki':
                    if need_eos_input:
                        # First frame after the end of the file, we feed a code full of 2048
                        # to indicate the end of stream.
                        need_eos_input = False
                        eos_value = self.mimi.cardinality
                        codes = torch.full(
                            (self.batch_size, self.mimi.num_codebooks, 1),
                            eos_value, device=device, dtype=torch.long)
                    else:
                        silence = torch.zeros((self.batch_size, self.mimi.channels, self.frame_size), device=device)
                        codes = self.mimi.encode(silence)
                else:
                    # For other models, we stop as soon as we are reaching the end of the audio.
                    break
            
            # if first_frame:
            #     # Ensure that the first slice of codes is properly seen by the transformer
            #     # as otherwise the first slice is replaced by the initial tokens.
            #     # codes: B x mimi.num_codebooks x 1(corresponding to framesize)

            #     # we have to repeat the codes for codec difference and provide them
            #     # as conditioning to the gesture model and the resulting gmoshi tokens
            #     # will be used to start the gesture codec decoding. otherwise the gesturecodec decoding 
            #     # is weird for first few frames. 
            #     # this is a hack, but works for now. but it causes 5 first conditioning frames to be the same which is not right.

            #     lmtoks = self.lm_gen.step(codes) # TODO: check if double pass messes up things
            #     assert lmtoks is None
            #     tokens = self.lm_gen.step(codes)
            #     assert tokens is not None
            #     moshi_token_buffer = [tokens] * self.codec_difference
            #     moshi_tokens = torch.cat(moshi_token_buffer, dim=-1)

            #     # assert len(moshi_token_buffer) == self.codec_difference

            #     gmoshi_tokens = self.glm_gen.step(moshi_tokens)
            #     assert moshi_tokens is not None and gmoshi_tokens is not None
            #     upper_tokens = gmoshi_tokens[:, :self.gesture_codecs[0].num_codebooks]
            #     lower_tokens = gmoshi_tokens[:, self.gesture_codecs[0].num_codebooks:]
            #     upper_motion = self.gesture_codecs[0].decode(upper_tokens)
            #     lower_motion = self.gesture_codecs[1].decode(lower_tokens)

            #     moshi_token_buffer = []
            #     first_frame = False

            # print("+++++", codes[0, :].tolist())
            tokens = self.lm_gen.step(codes)
            # print(self.lm_gen.offsets)
            if tokens is None:
                continue
            assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
            
            
            moshi_tokens = tokens
            # in_moshi_tokens = moshi_tokens
            # print("----",)
            text = self.text_tokenizer.id_to_piece(tokens[0, 0].item())
            # print("----", ntokens+1,  tokens[0, 0].item(), text, tokens[0, 1:].tolist(), (ntokens+1)*(self.frame_size/self.mimi.sample_rate), '\n')
            # counter += 1

            out_pcm = self.mimi.decode(tokens[:, 1:]).cpu()


            g_tic = time.time()
            gmoshi_tokens = self.glm_gen.step(moshi_tokens, ca_query_padding_mask=self.lower_cross_attn_mask)
            assert gmoshi_tokens.shape[1] == self.glm_gen.glm_model.n_q

            

            upper_tokens = gmoshi_tokens[:, :self.gesture_codecs[0].num_codebooks, :] 
            lower_tokens = gmoshi_tokens[:, self.gesture_codecs[0].num_codebooks:self.gesture_codecs[0].num_codebooks + self.gesture_codecs[1].num_codebooks, :] 
            face_tokens = gmoshi_tokens[:, self.gesture_codecs[0].num_codebooks + self.gesture_codecs[1].num_codebooks:, :] 
            

            # upper_tokens = gmoshi_tokens[:, :self.gesture_codecs[0].num_codebooks]
            # face_tokens = gmoshi_tokens[:, self.gesture_codecs[0].num_codebooks: self.gesture_codecs[0].num_codebooks + self.gesture_codecs[1].num_codebooks]
            # lower_tokens = gmoshi_tokens[:, -self.gesture_codecs[2].num_codebooks:]
            upper_motion = self.gesture_codecs[0].decode(upper_tokens)
            lowertrans_motion = self.gesture_codecs[1].decode(lower_tokens)
            faceexp_motion = self.gesture_codecs[2].decode(face_tokens)


            
            
            bs = lowertrans_motion.shape[0]
            uhj = self.motion_joints_per_codec[0]
            lj = self.motion_joints_per_codec[1]
            fj = self.motion_joints_per_codec[2]
            nframes = self.motion_frame_size

            # TEMPORARY:
            # lowertrans_motion = torch.zeros_like(lowertrans_motion)
            # upper_motion = torch.zeros(bs, nframes, uhj*6, device=lowertrans_motion.device)

            upper_motion = upper_motion.reshape(bs, nframes, uhj, 6)
            upper_motion = rc.rotation_6d_to_matrix(upper_motion)
            upper_motion = rc.matrix_to_axis_angle(upper_motion).reshape(bs, nframes, (uhj)*3)
            
            # pred_posetrans_lower
            lower_motion = lowertrans_motion[:, :, :lj*6]
            motion_transvel = lowertrans_motion[:, :, lj*6:lj*6+3]
            lower_motion = lower_motion.reshape(bs, nframes, lj, 6)
            lower_motion = rc.rotation_6d_to_matrix(lower_motion)
            lower_motion = rc.matrix_to_axis_angle(lower_motion).reshape(bs, nframes, lj*3)
            motion_trans, nextfinalpos = velocity2position_mixeddiff(
                motion_transvel, 1/self.gesture_codecs[1].motion_fps, init_pos=self.final_pos
            )
            # print("--->", self.final_pos, motion_trans, nextfinalpos)
            self.final_pos = nextfinalpos

            face_motion = faceexp_motion[:, :, :fj*6]
            face_exps = faceexp_motion[:, :, fj*6:]
            face_motion = face_motion.reshape(bs, nframes, fj, 6)
            face_motion = rc.rotation_6d_to_matrix(face_motion)
            face_motion = rc.matrix_to_axis_angle(face_motion).reshape(bs, nframes, fj*3)

            

            # TEMPORARY:
            # lower_motion = torch.zeros_like(lower_motion)
            # upper_motion = torch.zeros_like(upper_motion)
            # motion_trans = torch.zeros_like(motion_trans)
            
            upper_motion = upper_motion.cpu().numpy()
            face_motion = face_motion.cpu().numpy()
            lower_motion = lower_motion.cpu().numpy()
            face_exps = face_exps.cpu().numpy()
            motion_trans = motion_trans.cpu().numpy()
            

            # Masks are now torch tensors (converted in main() so the warmup
            # can call the tensor variant). The numpy variant below uses
            # `np.where(... == 1)`, so bridge them once per loop iteration.
            mask_upper_np = self.motion_mask[0].cpu().numpy()
            mask_lower_np = self.motion_mask[1].cpu().numpy()
            mask_face_np = self.motion_mask[2].cpu().numpy()
            # if len(moshi_token_buffer) == self.codec_difference:
            for b, (one_um, one_fm, one_lm, one_exp, one_tr) in enumerate(zip(upper_motion, face_motion, lower_motion, face_exps, motion_trans)):
                if eos_reached[b]:
                    continue
                # print(one_um.shape, one_lm.shape, one_tr.shape)
                pred_motion_upper = inverse_selection_smplx(one_um, mask_upper_np, nframes)
                pred_motion_lower = inverse_selection_smplx(one_lm, mask_lower_np, nframes)
                pred_motion_face = inverse_selection_smplx(one_fm, mask_face_np, nframes)
                pred_motion_combined = pred_motion_upper + pred_motion_face + pred_motion_lower
                # one_tr = np.zeros_like(one_tr) # TEMPORARY
                out_motion_per_item[b].append(np.concatenate([pred_motion_combined, one_exp, one_tr], axis=-1))
                # if b == 0:
                    # print("++++", ntokens+1, one_um.shape, one_lm.shape, one_tr.shape)

            # if len(moshi_token_buffer) == self.codec_difference: 
            #     moshi_token_buffer = []
            g_toc = time.time()
            g_step_time_average += (g_toc - g_tic)
            


            for b, (one_text, one_pcm) in enumerate(zip(tokens[:, 0].cpu(), out_pcm)):
                if eos_reached[b]:
                    continue
                elif one_text.item() == self.text_tokenizer.eos_id():
                    if need_eos_input:
                        # We sampled the EOS before the end of the file! Not possible.
                        self.printer.log("warning", "EOS sampled too early.")
                    else:
                        eos_reached[b] = True

                out_text_tokens_per_item[b].append(one_text)
                out_pcms_per_item[b].append(one_pcm)
                if b == 0:
                    if one_text.item() not in [0, 3]:
                        text = self.text_tokenizer.id_to_piece(one_text.item())  # pyright: ignore
                        text = text.replace("▁", " ")
                        self.printer.print_token(text)

            mg_toc = time.time()
            mg_steptime_average += (mg_toc - mg_tic)


            
            ntokens += 1
        dt = time.time() - start_time
        self.printer.log("info", f"processed {ntokens} steps in {dt:.0f}s, {1000 * dt / ntokens:.2f}ms/step")
        
        mg_steptime_average = mg_steptime_average / ntokens
        g_step_time_average = g_step_time_average / ntokens
        self.printer.log("info", f"average gesture lm step time: {1000 * g_step_time_average:.2f}ms")
        self.printer.log("info", f"average Moshi+GLM time: {1000 * mg_steptime_average:.2f}ms")

        out = [
            (torch.cat(one_texts, dim=0), torch.cat(one_pcms, dim=1))
            for one_texts, one_pcms in zip(out_text_tokens_per_item, out_pcms_per_item)
        ]
        out_motion = [
            np.concatenate(one_motion, axis=0)
            for one_motion in out_motion_per_item
        ]
        # print(len(out))
        return out, out_pcms_per_item, out_motion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo to look into, defaults Moshiko. "
                             "Use this to select a different pre-trained model.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size to be used for inference.")
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument("--half", action="store_const", const=torch.float16, default=torch.bfloat16,
                        dest="dtype", help="Run inference with float16, not bfloat16, better for old GPUs.")
    parser.add_argument("--config", "--lm-config", dest="config", type=str, help="The config as a json file.")
    parser.add_argument("--cfg-coef", type=float, default=1., help="CFG coefficient.")
    
    parser.add_argument("--infile", type=str, required=True, help="Input audio file.")
    parser.add_argument(
        "--outfile", type=str, default=None,
        help="Output audio file in wav format. If omitted, defaults to "
             "<infile-stem>_output.wav next to the input file.",
    )

    parser.add_argument("--character-id", type=int, default=3, help="Character ID to use.") # default wayne, for goodspk exp: can be ['lawrence': 3, 'solomon': 4, 'wayne': 15, 'stewart': 2]
    parser.add_argument("--glm-config", type=str, required=True)
    parser.add_argument("--glm-cfg-coef", type=float, default=1.3, help="Gesture LM CFG coefficient.")
    parser.add_argument(
        "--visualize", action="store_true",
        help="After saving the .npz, also render an SMPL-X debug mp4 of the "
             "predicted motion with user+Moshi audio muxed in. Uses "
             "scripts/trainers/dataloaders/utils/visualize.py:render_smplx_debug_video. "
             "Requires `outfile`.",
    )
    parser.add_argument("--viz-width", type=int, default=640, help="Visualization video width.")
    parser.add_argument("--viz-height", type=int, default=480, help="Visualization video height.")
    parser.add_argument("--viz-fps", type=int, default=25, help="Visualization video FPS.")
    args = parser.parse_args()
    # If --outfile is not provided, derive it from --infile: same directory,
    # `<stem>_output.wav` name. The downstream `if args.outfile:` guard
    # therefore always passes when --infile is given, so saving + (optional)
    # visualization happen by default.
    if not args.outfile:
        in_path = Path(args.infile)
        args.outfile = str(in_path.with_name(f"{in_path.stem}_output.wav"))
        log("info", f"--outfile not provided; defaulting to {args.outfile}")
    seed_all(2342)

    # TODO: add weight paths 
    with open(args.glm_config, 'r') as f:
        train_config_dict = yaml.safe_load(f)
    # Convert dict to an object with attributes for easier access
    trainer_args = SimpleNamespace(**train_config_dict)

    log("info", "retrieving checkpoint")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo, args.moshi_weight, args.mimi_weight, args.tokenizer, args.config)
    log("info", "loading mimi")
    mimi = checkpoint_info.get_mimi(device=args.device)
    log("info", "mimi loaded")
    text_tokenizer = checkpoint_info.get_text_tokenizer()
    log("info", "loading moshi")
    lm = checkpoint_info.get_moshi(device=args.device, dtype=args.dtype)
    log("info", "moshi loaded")

    text_procemb = copy.deepcopy(lm.text_emb.weight.data)
    audio_procemb = [copy.deepcopy(a_emb.weight.data) for a_emb in lm.emb[:8]]
    smplx_model = smplx.create(
        trainer_args.deps_path + 'smplx_2020/', 
        model_type="smplx",
        gender="NEUTRAL_2020",
        flat_hand_mean=True,
        num_betas=300,
        num_expression_coeffs=100,
        use_pca=False,
    ).cuda().eval()
    for p in smplx_model.parameters():
        p.requires_grad = False

    body_parts = 3  # upper, face, lower
    # TODO: move to loaders or handle somewhere else
    gesture_codec_weights = [
        trainer_args.upperbodycodec_ckpt,
        trainer_args.lowerbodycodec_ckpt,
        trainer_args.facecodec_ckpt,
    ]
    upper_bodymask = get_smplx_bodypart_masks("upper")
    face_bodymask = get_smplx_bodypart_masks("face")
    lower_bodymask = get_smplx_bodypart_masks("lower")
    hand_bodymask = get_smplx_bodypart_masks("hands")
    # Match gest-server.py: convert to torch tensors so the warmup's
    # `inverse_selection_tensor_smplx` call (which asserts torch.Tensor)
    # succeeds. The old run() inference loop runs the inverse-selection on
    # CPU, so leave masks on CPU; per-call sites move tensors to CPU before
    # the inverse selection.
    upper_bodymask = torch.from_numpy(upper_bodymask)
    face_bodymask = torch.from_numpy(face_bodymask)
    lower_bodymask = torch.from_numpy(lower_bodymask)
    hand_bodymask = torch.from_numpy(hand_bodymask)
    motion_mask = [upper_bodymask + hand_bodymask, lower_bodymask, face_bodymask]
    gesture_lm_config = {
        "use_sampling": True,
        "temp_gtemporal": 0.9,
        "temp_gdepth": 0.9,
        "top_p_gtemporal": 0.8,
        "top_p_gdepth": 0.95,
        "check": True,
        "character_id": args.character_id,
        "motion_mask": motion_mask,
    }
    checkpoint_info.load_gesture_weights(gesture_codec_weights, trainer_args.test_ckpt, gesture_lm_config)
    
    gcodec_kwargs = dict(
        num_frames=trainer_args.num_frames,
        frame_chunk_size=trainer_args.frame_chunk_size,
        upperlower_nfeats=trainer_args.upperlower_nfeats,
        face_nfeats=trainer_args.face_nfeats,
        lowertrans_nfeats=trainer_args.lowertrans_nfeats,
        motion_fps=trainer_args.motion_fps,
        transformer_heads=trainer_args.transformer_heads,
        transformer_layers=trainer_args.transformer_layers,
        convblock_layers=trainer_args.convblock_layers,
    )
    gesture_codecs = checkpoint_info.get_gesture_codecs(
        device=args.device,
        codec_kwargs=gcodec_kwargs,
    )

    upper_codec_layers = copy.deepcopy(gesture_codecs[0].quantizer.vq.layers) # nn.ModuleList)
    lower_codec_layers = copy.deepcopy(gesture_codecs[1].quantizer.vq.layers) # nn.ModuleList
    face_codec_layers = copy.deepcopy(gesture_codecs[2].quantizer.vq.layers) # nn.ModuleList
    
    log("info", "gesturecodecs loaded")

    gesture_codec_layers = upper_codec_layers + lower_codec_layers + face_codec_layers

    # freeze the gesture codec layers
    for gcodec_layer in gesture_codec_layers:
        for p in gcodec_layer.parameters():
            p.requires_grad = False
    gesture_codec_layers.eval()
    gesture_lm_kwargs = dict(
        num_heads=trainer_args.gestureformer_heads,
        num_layers=trainer_args.gestureformer_layers,
        depformer_heads=trainer_args.gestureformer_depformer_heads,
        depformer_layers=trainer_args.gestureformer_depformer_layers,
        query2mem_scale=int(mimi.frame_rate / gesture_codecs[0].frame_rate),
        num_temp_classifiers=trainer_args.num_temp_classifiers,
        text_procemb=text_procemb,
        audio_procemb=audio_procemb,
        gesture_codec_layers=gesture_codec_layers,
        vad_guidance=trainer_args.vad_guidance,
        body_parts=body_parts,
        bp_dist=None, 
        textaudio_emb_freeze=trainer_args.textaudio_emb_freeze if hasattr(trainer_args, 'textaudio_emb_freeze') else False,
    )
    
    gesture_lm = checkpoint_info.get_gesture_lm(
        device=args.device,
        dtype=None, #args.dtype,
        gesture_lm_kwargs=gesture_lm_kwargs,
    )
    log("info", "gesturelm loaded")

    log("info", f"loading input file {args.infile}")
    in_pcms, _ = sphn.read(args.infile, sample_rate=mimi.sample_rate)
    in_pcms = torch.from_numpy(in_pcms).to(device=args.device)
    in_pcms = in_pcms[None, 0:1].expand(args.batch_size, -1, -1)

    state = InferenceState(
        checkpoint_info.model_type, mimi, text_tokenizer, lm, gesture_lm, gesture_codecs,
        args.batch_size, args.cfg_coef, args.glm_cfg_coef, args.device, gesturelm_kwargs=gesture_lm_config, **checkpoint_info.lm_gen_config, 
    )
    out_items, out_achunks, out_motionchunks = state.run(in_pcms)

    if args.outfile:
        outfile = Path(args.outfile)
        for index, (_, out_pcm) in enumerate(out_items):
            if len(out_items) > 1:
                outfile_ = outfile.with_name(f"{outfile.stem}-{index}{outfile.suffix}")
            else:
                outfile_ = outfile
            
            duration = out_pcm.shape[1] / mimi.sample_rate
            # print(out_pcm.shape)
            log("info", f"writing {outfile_} with duration {duration:.1f} sec.")
            sphn.write_wav(
                str(outfile_).replace(".wav", f"-character{args.character_id}.wav"), 
                out_pcm[0].numpy(), 
                sample_rate=mimi.sample_rate
            )
            # os.makedirs(os.path.join(outfile_.parent, "chunks"), exist_ok=True)
            
            # for cidx, out_pcms in enumerate(out_achunks[0]):
            #     out_pcm = out_pcms[0]
            #     duration = out_pcm.shape[0] / mimi.sample_rate
            #     print(out_pcm.shape)
            #     chunk_file = os.path.join(outfile_.parent, "chunks", f"{outfile_.stem}-{cidx}.wav")
            #     log("info", f"writing {chunk_file} with duration {duration:.1f} sec.")
            #     sphn.write_wav(chunk_file, out_pcm.numpy(), sample_rate=mimi.sample_rate)

            out_motiontrans = out_motionchunks[0]
            out_motionexp = out_motiontrans[:, :-3]
            out_motion = out_motionexp[:, :-100]
            out_exps = out_motionexp[:, -100:]
            out_trans = out_motiontrans[:, -3:]
            sample_save_path = os.path.join(outfile_.parent, f"{outfile_.stem}-character{args.character_id}.npz")
            
            
            num_frames = out_motion.shape[0]
            tar_beta = torch.zeros((num_frames, 300))
            tar_beta = tar_beta.cpu().numpy()
            out_motion = out_motion.reshape(out_motion.shape[0], -1, 3)
            
            rec_out = {
                    "global_orient": out_motion[:, 0].reshape(num_frames, -1),
                    "body_pose": out_motion[:, 1:22].reshape(num_frames, -1),
                    "left_hand_pose": out_motion[:, 25:40].reshape(num_frames, -1),
                    "right_hand_pose": out_motion[:, 40:55].reshape(num_frames, -1),
                    "transl": out_trans,
                    "betas": tar_beta,
                    "expression": out_exps,
                    "jaw_pose": out_motion[:, 22:23].reshape(num_frames, -1),
                    "leye_pose": out_motion[:, 23:24].reshape(num_frames, -1),
                    "reye_pose": out_motion[:, 24:25].reshape(num_frames, -1),
                }

            
            if args.visualize:
                # Lazy-import the renderer + its heavy deps (pyrender, cv2)
                # only when actually requested, so plain inference runs
                # don't pay the load cost.
                from scripts.trainers.dataloaders.utils.visualize import render_smplx_debug_video

                moshi_audio_file = str(outfile_).replace(".wav", f"-character{args.character_id}.wav")
                user_audio_file = args.infile
                mixed_audio_file = moshi_audio_file.replace(".wav", "-mixed.wav")
                final_video = sample_save_path.replace(".npz", ".mp4")

                # Mix user + Moshi audio into a single track before muxing
                # it onto the rendered mp4. render_smplx_debug_video's
                # audio_path arg takes one file -- we feed it the mix.
                os.system(
                    f"ffmpeg -loglevel error -i {moshi_audio_file} -i {user_audio_file} "
                    f"-filter_complex \"[0:a][1:a]amix=inputs=2:duration=longest\" "
                    f"-y {mixed_audio_file}"
                )

                # poses: (T, 165) flat SMPL-X pose. transl: (T, 3).
                # expressions: (T, 100). betas: (T, 300) zeros.
                
                viz_trans = out_trans - out_trans[0:1, :]
                rendered_path = render_smplx_debug_video(
                    smplx_model=smplx_model,
                    poses=out_motion.reshape(num_frames, -1),
                    transl=viz_trans,
                    expressions=out_exps,
                    betas=tar_beta,
                    output_path=final_video,
                    fps=args.viz_fps,
                    width=args.viz_width,
                    height=args.viz_height,
                    audio_path=mixed_audio_file,
                )
                log("info", f"saved {sample_save_path} and {rendered_path}")
                if os.path.exists(mixed_audio_file):
                    os.remove(mixed_audio_file)
            else:
                np.savez(sample_save_path,
                    betas=tar_beta[0],
                    poses=out_motion.reshape(num_frames, -1),
                    expressions=out_exps,
                    trans=out_trans,
                    model='smplx',
                    gender='neutral',
                    mocap_frame_rate = 30, #kept to be 30 because smplx blender addon expects always 30, change in blender to 25.
                )
                log("info", f"saved {sample_save_path} (pass --visualize to also render an mp4)")

            
                


if __name__ == "__main__":
    with torch.no_grad():
        main()
