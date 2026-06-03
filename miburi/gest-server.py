# Modified from 
import argparse
import asyncio
from dataclasses import dataclass
import inspect
import random
import os
from pathlib import Path
import tarfile
import time
import secrets
import sys
import aiohttp
import struct
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch
import os
import json
import yaml
import copy

from types import SimpleNamespace
from .client_utils import log
from .utils.motion_utils import get_gesture_condition_tensors, inverse_selection_tensor_smplx, get_smplx_bodypart_masks, velocity2position_mixeddiff
from .models import loaders, MimiModel, LMModel, LMGen, GestureLMGen, GestureMimiCodec, GTemporalDepthModel3
from .run_gestinference import get_condition_tensors
from .utils import rotation_conversions as rc

import time
import trimesh

import subprocess
import sys

torch.set_float32_matmul_precision('medium')

MOTION_HEADER_FMT = ">IIII"

IDLE_TAIL_THRESHOLD = 12


def build_dashboard_html(audio_path: str, motion_url: str, minimal_audio_ui: bool = False) -> str:
    # In minimal mode the Moshi UI only shows the mic+downloads column
    # (transcript and stats are CSS-hidden), so the audio pane needs much
    # less width -- give the motion viewer most of the row.
    columns = "0.75fr 1.25fr" if minimal_audio_ui else "1.00fr 1.00fr"
    return """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>MIBURI</title>
    <style>
      :root {{
        color-scheme: dark;
      }}
      * {{
        box-sizing: border-box;
      }}
      body {{
        margin: 0;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
        background: #070707;
        color: #d7d7d7;
      }}
      .shell {{
        min-height: 100vh;
        display: grid;
        grid-template-rows: auto 1fr;
      }}
      .bar {{
        padding: 10px 14px;
        border-bottom: 1px solid #242424;
        background: #111111;
      }}
      .title {{
        font-size: 18px;
        font-weight: 600;
        margin: 0;
        text-align: center;
      }}
      .subtitle {{
        margin: 4px 0 0 0;
        font-size: 12px;
        color: #a0a0a0;
        text-align: center;
      }}
      .panes {{
        display: grid;
        grid-template-columns: {columns};
        gap: 10px;
        padding: 10px;
        min-height: 0;
      }}
      .pane {{
        min-height: 0;
        border: 1px solid #242424;
        border-radius: 8px;
        overflow: hidden;
        background: #000;
      }}
      .pane-title {{
        margin: 0;
        padding: 8px 10px;
        border-bottom: 1px solid #242424;
        font-size: 12px;
        font-weight: 600;
        color: #bdbdbd;
        background: #0f0f0f;
      }}
      iframe {{
        width: 100%;
        height: calc(100vh - 92px);
        border: 0;
        display: block;
      }}
      @media (max-width: 1100px) {{
        .panes {{
          grid-template-columns: 1fr;
        }}
        iframe {{
          height: min(70vh, 680px);
        }}
      }}
    </style>
  </head>
  <body>
    <div class="shell">
      <header class="bar">
        <p class="title">MIBURI: Towards Expressive Interactive Gesture Synthesis (CVPR 2026)</p>
        <p class="subtitle">Audio (left) and Gesture Viewer (right)</p>
      </header>
      <main class="panes">
        <section class="pane">
          <iframe id="audio-iframe" src="{audio_path}" allow="microphone; autoplay"></iframe>
        </section>
        <section class="pane">
          <iframe id="motion-iframe" src="{motion_url}"></iframe>
        </section>
      </main>
    </div>
    <script>
      // Moshi's Start Over button calls `document.location.reload()` from
      // inside the audio iframe, which only reloads that iframe. We want a
      // full-page refresh so the motion iframe also re-mounts (Viser
      // session gone, old mesh gone, fresh audio WS). Trick: count iframe
      // `load` events. The first is the initial mount; any subsequent
      // load means the iframe self-reloaded (Start Over) -> reload the
      // top-level dashboard.
      (function() {{
        var audioFrame = document.getElementById("audio-iframe");
        var firstLoad = true;
        audioFrame.addEventListener("load", function() {{
          if (firstLoad) {{ firstLoad = false; return; }}
          window.location.reload();
        }});
      }})();
    </script>
  </body>
</html>
""".format(audio_path=audio_path, motion_url=motion_url, columns=columns)



def start_motion_server(
    host: str,
    vis_port: int,
    motion_port: int,
    smplx_dir: str,
    mixamo_character: str | None = None,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "miburi.motion_vis_server",
        "--host",
        host,
        "--viser-port",
        str(vis_port),
        "--port",
        str(motion_port),
        "--smplx-dir",
        smplx_dir,
    ]
    if mixamo_character:
        cmd.extend(["--mixamo-character", mixamo_character])
    return subprocess.Popen(cmd)

def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def get_smplx_mesh(smplx_model, forward_kwargs):
    # Get the SMPLX outputs
    with torch.no_grad():
        smplx_outputs = smplx_model(**forward_kwargs)

    vertices, faces = smplx_outputs["vertices"], smplx_outputs["faces"]

    mesh = trimesh.Trimesh(vertices=vertices.squeeze(0).cpu().numpy(), faces=faces.cpu().numpy())
    return mesh


def serialize_motion(frames: torch.Tensor, frame_count: int) -> bytes:
    motion_dim = frames.shape[-1] - 100 - 3
    exp_dim = 100
    transl_dim = 3
    header = struct.pack(MOTION_HEADER_FMT, frame_count, motion_dim, exp_dim, transl_dim)
    payload = frames.numpy().astype(np.float32).tobytes()
    return header + payload


@dataclass
class ServerState:
    model_type: str
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock

    def __init__(self, model_type: str, mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, gesturelm: GTemporalDepthModel3, gesture_codecs: list[GestureMimiCodec],
                 idle_upper_slice: np.ndarray,
                 idle_face_slice: np.ndarray,
                 cfg_coef: float, glm_cfg_coef:float, motion_vis_port: int, host: str,
                 device: str | torch.device,
                 idle_motion_replace: bool = True, **kwargs):
        self.model_type = model_type
        self.gesture_codecs = gesture_codecs
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        # When False, the per-frame silent-branch token substitution and
        # the speech-resume swap branches are skipped, letting the raw GLM
        # output flow through during silent stretches too.
        self.idle_motion_replace = idle_motion_replace
        condition_tensors = get_condition_tensors(model_type, lm, batch_size=1, cfg_coef=cfg_coef)
        gesturelm_kwargs = kwargs.pop("gesturelm_kwargs")
        self.motion_mask = gesturelm_kwargs.pop("motion_mask", None)

        character_id = gesturelm_kwargs.pop("character_id")
        gesture_spk_condition = get_gesture_condition_tensors(character_id)
        self.lm_gen = LMGen(lm, cfg_coef=cfg_coef, condition_tensors=condition_tensors, **kwargs)

        self.glm_gen = GestureLMGen(
            gesturelm,
            condition_tensors=gesture_spk_condition, 
            cfg_coef=glm_cfg_coef, 
            **gesturelm_kwargs
        )

        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lock = asyncio.Lock()

        self.codec_difference = int(self.mimi.frame_rate / self.gesture_codecs[0].frame_rate)
        self.motion_frame_size = self.gesture_codecs[0].frame_chunk_size
        self.motion_joints_per_codec = [43, 9, 1] # 43 for upper body, 9 for lower body, 1 for face
        
        self.batch_size = 1
        self.mimi.streaming_forever(self.batch_size)
        self.gesture_codecs[0].streaming_forever(self.batch_size) # upper body
        self.gesture_codecs[1].streaming_forever(self.batch_size) # lower body
        self.gesture_codecs[2].streaming_forever(self.batch_size) # face
        self.lm_gen.streaming_forever(self.batch_size)
        self.glm_gen.streaming_forever(self.batch_size)

        # 

        self.motion_frame_buffer = None
        self.transl_frame_buffer = None

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
        


        # Single-slice idle motion (was: per-character dict + random pick
        # + lawrence cleanup). One slice per stream, kept for the lifetime
        # of the process. The speech-resume branch rewinds the cursor to 0
        # when the unread tail is shorter than IDLE_TAIL_THRESHOLD.
        self.current_idle_motion = torch.from_numpy(idle_upper_slice).to(self.device)
        self.current_idlemotionlength = self.current_idle_motion.shape[-1]
        self.current_idle_motionpos = 0

        self.current_face_idle_motion = torch.from_numpy(idle_face_slice).to(self.device)
        self.current_face_idlemotionlength = self.current_face_idle_motion.shape[-1]
        self.current_face_idle_motionpos = 0

        

        self.main_pcm_buffer = None
        # 
        self.mainpcm_buffer_size = 3 * self.frame_size  # 3 frames buffer

        # -------
        self.final_pos = torch.zeros((self.batch_size)) + torch.tensor([0, 1.3, 0]) # final position for velocity2position conversion
        self.motion_ws_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.motion_vis_port = motion_vis_port

        self.host = host

    def _reset_session_state(self) -> None:
        """Wipe per-conversation auxiliary state so the next WS connection
        (e.g. after the client's Start Over button reloads the page) starts
        with zero context. The model streaming caches are reset separately
        via `*.reset_streaming()` in `handle_chat`; this covers everything
        else that __init__ + warmup set up."""
        # Silence-detection buffer: zero in place (rather than None) so the
        # `torch.cat(...)` in opus_loop continues to work without an init-
        # time check.
        if self.main_pcm_buffer is None:
            self.main_pcm_buffer = torch.zeros(
                1, 1, self.mainpcm_buffer_size, device=self.device,
            )
        else:
            self.main_pcm_buffer.zero_()
        # Motion frame buffers and world-position integrator.
        self.motion_frame_buffer = None
        self.transl_frame_buffer = None
        self.final_pos = torch.zeros((self.batch_size)) + torch.tensor([0, 1.3, 0])
        # Idle-motion cursors back to slice start (keep the currently
        # selected slices; positions only).
        self.current_idle_motionpos = 0
        self.current_face_idle_motionpos = 0
        # Drain queued motion packets so the new session doesn't render
        # stale frames from the previous one.
        while not self.motion_ws_queue.empty():
            try:
                self.motion_ws_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def warmup(self) -> None:
        """Warmup the model by running a few steps with dummy data."""
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
                # 
                upper_tokens = gmoshi_tokens[:, :self.gesture_codecs[0].num_codebooks, :] 
                lower_tokens = gmoshi_tokens[:, self.gesture_codecs[0].num_codebooks:self.gesture_codecs[0].num_codebooks + self.gesture_codecs[1].num_codebooks, :] 
                face_tokens = gmoshi_tokens[:, self.gesture_codecs[0].num_codebooks + self.gesture_codecs[1].num_codebooks:, :] 
                # 
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
                
                # pred_posetrans_lower
                lower_motion = lowertrans_motion[:, :, :lj*6]
                motion_transvel = lowertrans_motion[:, :, lj*6:lj*6+3]
                lower_motion = lower_motion.reshape(bs, nframes, lj, 6)
                lower_motion = rc.rotation_6d_to_matrix(lower_motion)
                lower_motion = rc.matrix_to_axis_angle(lower_motion).reshape(bs, nframes, lj*3)
                motion_trans, nextfinalpos = velocity2position_mixeddiff(
                    motion_transvel, 1/self.gesture_codecs[1].motion_fps, init_pos=self.final_pos
                )
                # print("--->", final_pos, motion_trans, nextfinalpos)
                self.final_pos = nextfinalpos

                face_motion = faceexp_motion[:, :, :fj*6]
                face_exps = faceexp_motion[:, :, fj*6:]
                face_motion = face_motion.reshape(bs, nframes, fj, 6)
                face_motion = rc.rotation_6d_to_matrix(face_motion)
                face_motion = rc.matrix_to_axis_angle(face_motion).reshape(bs, nframes, fj*3)


                upper_motion = upper_motion[0].cpu() #.numpy()
                face_motion = face_motion[0].cpu() #.numpy()
                lower_motion = lower_motion[0].cpu() #.numpy()
                face_exps = face_exps[0].cpu() #.numpy()
                motion_trans = motion_trans[0].cpu() #.numpy()


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

    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        motion_ws_url = f"ws://{self.host}:{self.motion_vis_port}/ws/motion"
        motion_session = aiohttp.ClientSession()
        motion_ws = await motion_session.ws_connect(motion_ws_url, max_msg_size=0)


        async def recv_loop():
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        log("error", f"{ws.exception()}")
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        log("error", f"unexpected message type {message.type}")
                        continue
                    message = message.data
                    if not isinstance(message, bytes):
                        log("error", f"unsupported message type {type(message)}")
                        continue
                    if len(message) == 0:
                        log("warning", "empty message")
                        continue
                    kind = message[0]
                    if kind == 1:  # audio
                        payload = message[1:]
                        opus_reader.append_bytes(payload)
                    else:
                        log("warning", f"unknown message kind {kind}")
            finally:
                close = True
                # Sentinel that unparks motion_send_loop from its blocking
                # `await self.motion_ws_queue.get()`. Without this the loop
                # never wakes after the WS closes, asyncio.gather stays
                # pending, and self.lock is held forever -- which makes the
                # *next* connection (e.g. after Start Over) hang.
                self.motion_ws_queue.put_nowait(None)
                log("info", "connection closed")

        async def opus_loop():
            nonlocal last_text_char
            all_pcm_data = None
            skip_frames = 1
            
            mframe_idx = 0
            # trans_frames = None
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()
                if pcm.shape[-1] == 0:
                    continue
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))

                

                while all_pcm_data.shape[-1] >= self.frame_size:
                    be = time.time()
                    chunk = all_pcm_data[: self.frame_size]
                    all_pcm_data = all_pcm_data[self.frame_size:]
                    chunk = torch.from_numpy(chunk)
                    chunk = chunk.to(device=self.device)[None, None]
                    codes = self.mimi.encode(chunk)
                    
                    
                    if skip_frames:
                        # The first input audio frame is ignored, as from the point of
                        # view of the model it is in the past. We still `mimi.encode` for simplicity,
                        # however as the first encoded frame has a specific structure (due to the left padding),
                        # we reset the streaming state of the encoder to reapply the padding on the next call.
                        self.mimi.reset_streaming()
                        skip_frames -= 1
                    for c in range(codes.shape[-1]):
                        tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                        if tokens is None:
                            continue
                        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1

                        moshi_tokens = tokens
                        main_pcm = self.mimi.decode(tokens[:, 1:])
                        
                        

                        gmoshi_tokens = self.glm_gen.step(moshi_tokens, ca_query_padding_mask=self.lower_cross_attn_mask)
                        
                        assert gmoshi_tokens.shape[1] == self.glm_gen.glm_model.n_q

                        upper_tokens = gmoshi_tokens[:, :self.gesture_codecs[0].num_codebooks, :]  # if main_pcm.norm() > 0.03 else idle_token
                        lower_tokens = gmoshi_tokens[:, self.gesture_codecs[0].num_codebooks:self.gesture_codecs[0].num_codebooks + self.gesture_codecs[1].num_codebooks, :]
                        face_tokens = gmoshi_tokens[:, self.gesture_codecs[0].num_codebooks + self.gesture_codecs[1].num_codebooks:, :]

                        # idle motion forcing logic

                        self.main_pcm_buffer = torch.cat((self.main_pcm_buffer, main_pcm), dim=2)
                        self.main_pcm_buffer = self.main_pcm_buffer[:, :, -self.mainpcm_buffer_size :]
                        # 

                        
                        if self.idle_motion_replace and self.main_pcm_buffer.norm() <= 0.5:

                            # Upper-stream substitution. Wrap to start if
                            # we've already played to the end (very long
                            # silence). No more "safe-checkpoint" arming
                            # since there's no swap-to-new-slice to gate.
                            idle_slice_length = upper_tokens.shape[-1]
                            if self.current_idle_motionpos >= self.current_idlemotionlength:
                                self.current_idle_motionpos = 0
                            upper_tokens = self.current_idle_motion[:, :, self.current_idle_motionpos : self.current_idle_motionpos + idle_slice_length]
                            self.current_idle_motionpos += idle_slice_length

                            # Face-stream substitution. Independent cursor,
                            # same simplified wrap rule.
                            face_idle_slice_length = face_tokens.shape[-1]
                            if self.current_face_idle_motionpos >= self.current_face_idlemotionlength:
                                self.current_face_idle_motionpos = 0
                            face_tokens = self.current_face_idle_motion[:, :, self.current_face_idle_motionpos : self.current_face_idle_motionpos + face_idle_slice_length]
                            self.current_face_idle_motionpos += face_idle_slice_length

                            


                        # Anticipatory rewind: while we're SPEAKING, if the
                        # unread tail of either idle slice is shorter than
                        # IDLE_TAIL_THRESHOLD tokens, reset that stream's
                        # cursor to 0. This prepares the next silent
                        # stretch to start playing the slice from the
                        # beginning instead of running off the end. The
                        # assignment is idempotent so the branch firing
                        # every LM step is harmless.
                        if self.idle_motion_replace and self.main_pcm_buffer.norm() > 0.5:
                            if self.current_idlemotionlength - self.current_idle_motionpos < IDLE_TAIL_THRESHOLD:
                                self.current_idle_motionpos = 0
                            if self.current_face_idlemotionlength - self.current_face_idle_motionpos < IDLE_TAIL_THRESHOLD:
                                self.current_face_idle_motionpos = 0

                        
                        # ------------------

                        upper_motion = self.gesture_codecs[0].decode(upper_tokens)
                        lowertrans_motion = self.gesture_codecs[1].decode(lower_tokens)
                        faceexp_motion = self.gesture_codecs[2].decode(face_tokens)
                        


                        main_pcm = main_pcm.cpu()
                        # print(main_pcm.norm().item())
                        opus_writer.append_pcm(main_pcm[0, 0].numpy())
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
                            _text = _text.replace("▁", " ")
                            # Start each new sentence on its own line in the
                            # Moshi UI transcript: when the previous piece
                            # ended a sentence (. ? !), strip the leading
                            # space the tokenizer adds and prefix a newline.
                            # The transcript element gets `white-space:
                            # pre-wrap` injected into index.html so the
                            # newline renders as a line break.
                            if last_text_char in ".?!" and _text:
                                _text = "\n" + _text.lstrip(" ")
                            if _text:
                                last_text_char = _text[-1]
                            msg = b"\x02" + bytes(_text, encoding="utf8")
                            log("info", f"text token '{_text}'")
                            await ws.send_bytes(msg)

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
                        # print("--->", final_pos, motion_trans, nextfinalpos)
                        self.final_pos = nextfinalpos

                        face_motion = faceexp_motion[:, :, :fj*6]
                        face_exps = faceexp_motion[:, :, fj*6:]
                        face_motion = face_motion.reshape(bs, nframes, fj, 6)
                        face_motion = rc.rotation_6d_to_matrix(face_motion)
                        face_motion = rc.matrix_to_axis_angle(face_motion).reshape(bs, nframes, fj*3)

                        upper_motion = upper_motion[0].cpu() #.numpy()
                        face_motion = face_motion[0].cpu() #.numpy()
                        lower_motion = lower_motion[0].cpu() #.numpy()
                        face_exps = face_exps[0].cpu() #.numpy()
                        motion_trans = motion_trans[0].cpu() #.numpy()


                        pred_motion_upper = inverse_selection_tensor_smplx(upper_motion, self.motion_mask[0], nframes)
                        pred_motion_lower = inverse_selection_tensor_smplx(lower_motion, self.motion_mask[1], nframes)
                        pred_motion_face = inverse_selection_tensor_smplx(face_motion, self.motion_mask[2], nframes)
                        pred_motion_combined = pred_motion_upper + pred_motion_face + pred_motion_lower
                        
                        out_motion = torch.cat([pred_motion_combined, face_exps, motion_trans], axis=-1)
                        # 
                        
                        # send to motion vis server

                        
                        serialized = serialize_motion(out_motion, self.motion_frame_size)
                        await self.motion_ws_queue.put(serialized)
                        
                        

                    log("info", f"frame handled in {1000 * (time.time() - be):.1f}ms")

                


        async def send_loop():
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send_bytes(b"\x01" + msg)

        async def motion_send_loop(motion_ws):
            # Blocking get; recv_loop's finally puts a `None` sentinel on
            # the queue when the WS closes, which unparks us so this
            # coroutine can return promptly and release `self.lock`.
            while True:
                batch = await self.motion_ws_queue.get()
                if batch is None or close:
                    return
                await motion_ws.send_bytes(batch)

        log("info", "accepted connection")
        close = False
        # Last character of the most recently sent text token; used by
        # opus_loop to decide when to start a new line in the transcript.
        last_text_char = ""
        async with self.lock:
            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            self.mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            self.glm_gen.reset_streaming()
            self.gesture_codecs[0].reset_streaming()
            self.gesture_codecs[1].reset_streaming()
            self.gesture_codecs[2].reset_streaming()
            # Plus the non-model auxiliary state (PCM silence buffer,
            # motion buffers, world-pos integrator, idle cursors, queue).
            self._reset_session_state()
            # Send the handshake.
            await ws.send_bytes(b"\x00")
            try:
                await asyncio.gather(opus_loop(), recv_loop(), send_loop(), motion_send_loop(motion_ws))
            finally:
                await motion_ws.close()
                await motion_session.close()
        log("info", "done with connection")
        return ws


def main():
    log("info", f"Running server from file: {Path(__file__).resolve()}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str, default="assets_dep/demo-static")
    parser.add_argument("--gradio-tunnel", action='store_true', help='Activate a gradio tunnel.')
    parser.add_argument("--gradio-tunnel-token",
                        help='Provide a custom (secret) token here to keep getting the same URL.')

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo to look into, defaults Moshiko. "
                             "Use this to select a different pre-trained model.")
    parser.add_argument("--lora-weight", type=str, help="Path to a local checkpoint file for LoRA.", default=None)
    parser.add_argument("--config-path", type=str, help="Path to a local config file.", default=None)
    parser.add_argument("--cfg-coef", type=float, default=1., help="CFG coefficient.")
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument("--no_fuse_lora", action="store_false", dest="fuse_lora", default=True,
                        help="Do not fuse LoRA layers intot Linear layers.")
    parser.add_argument("--half", action="store_const", const=torch.float16, default=torch.bfloat16,
                        dest="dtype", help="Run inference with float16, not bfloat16, better for old GPUs.")
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        )
    )

    parser.add_argument("--character-id", type=int, default=3, help="Character ID to use.") # default wayne, for goodspk exp: can be ['lawrence': 3, 'solomon': 4, 'wayne': 15, 'stewart': 2]
    parser.add_argument("--glm-config", type=str, required=True)
    parser.add_argument("--glm-cfg-coef", type=float, default=1.3, help="Gesture LM CFG coefficient.")
    

    parser.add_argument("--motion-vis-port", type=int, default=9900, help="Port for motion visualization server.")
    parser.add_argument("--viser-port", type=int, default=8083, help="Port for Viser web UI.")
    parser.add_argument(
        "--mixamo-character", type=str, default=None,
        help="Optional Mixamo character bundle (path to .npz, or a slug name "
             "resolved against assets_dep/mixamo_characters_release/<slug>.npz). When "
             "set, the demo renders that character driven by SMPL-X output "
             "instead of the default SMPL-X body.",
    )
    parser.add_argument(
        "--minimal-audio-ui", action="store_true",
        help="Hide the transcript text panel and Server Audio Stats panel "
             "in the embedded Moshi audio UI (purely a presentation toggle; "
             "the WebSocket protocol is unchanged).",
    )
    parser.add_argument(
        "--idle-motion-replace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="During silent stretches of Moshi's reply, substitute the "
             "GLM's upper-body + face gesture tokens with pre-recorded "
             "idle slices (and roll new random slices when speech "
             "resumes). Default: on. Pass --no-idle-motion-replace to let "
             "the raw GLM output flow through during silence too.",
    )

    args = parser.parse_args()
    seed_all(2342)

    with open(args.glm_config, 'r') as f:
        train_config_dict = yaml.safe_load(f)
    # Convert dict to an object with attributes for easier access
    trainer_args = SimpleNamespace(**train_config_dict)

    # Fast-fail on missing idle-motion files BEFORE any model load
    # (Moshi 7B + Mimi can be ~30 GB / ~minute of GPU work on first run).
    # Files: <exp_dir>/tokens_idle_slices/idle_{upper,face}_tokens_slices.npz.
    # Each file holds a single slice at key '0' (shape (1, K, T)).
    exp_dir = os.path.dirname(args.glm_config)
    idle_motion_path = os.path.join(
        exp_dir, "tokens_idle_slices", "idle_upper_tokens_slices.npz",
    )
    face_idle_motion_path = os.path.join(
        exp_dir, "tokens_idle_slices", "idle_face_tokens_slices.npz",
    )
    if not os.path.exists(idle_motion_path):
        raise FileNotFoundError(
            f"Upper-body idle motion slice not found: {idle_motion_path}"
        )
    if not os.path.exists(face_idle_motion_path):
        raise FileNotFoundError(
            f"Face idle motion slice not found: {face_idle_motion_path}"
        )
    idle_upper_slice = np.load(idle_motion_path)["0"]
    idle_face_slice = np.load(face_idle_motion_path)["0"]
    log(
        "info",
        f"Idle slices loaded: upper={idle_upper_slice.shape}, "
        f"face={idle_face_slice.shape}",
    )

    # Speaker name<->id map dumped by the trainer at training time
    # (baseglm_trainer writes this next to config.yaml). The integer
    # IDs are dataset_ratio + HDF5 row-order dependent, so the demo
    # must read the run-specific map rather than assuming a universal one.
    speaker_map_path = os.path.join(exp_dir, "speaker_id_to_index.json")
    if not os.path.exists(speaker_map_path):
        raise FileNotFoundError(
            f"Speaker ID-to-name map not found: {speaker_map_path}. "
            f"Re-run training (the trainer now writes this file next to config.yaml) "
            f"or back-fill it from the training log."
        )
    with open(speaker_map_path, "r") as f:
        character_name_to_id = json.load(f)
    character_id_to_name = {v: k for k, v in character_name_to_id.items()}
    if args.character_id not in character_id_to_name:
        raise ValueError(
            f"--character-id {args.character_id} is not present in {speaker_map_path}. "
            f"Valid IDs for this experiment: {sorted(character_id_to_name)}."
        )
    log(
        "info",
        f"Speaker map loaded: {len(character_name_to_id)} speakers, "
        f"using '{character_id_to_name[args.character_id]}' (id {args.character_id}).",
    )

    setup_tunnel = None
    tunnel_token = ''
    if args.gradio_tunnel:
        try:
            from gradio import networking  # type: ignore
        except ImportError:
            log("error", "Cannot find gradio which is required to activate a tunnel. "
                         "Please install with `pip install gradio`.")
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        if args.gradio_tunnel_token is None:
            tunnel_token = secrets.token_urlsafe(32)
        else:
            tunnel_token = args.gradio_tunnel_token

    log("info", "retrieving checkpoint")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo, args.moshi_weight, args.mimi_weight, args.tokenizer,
        lora_weights=args.lora_weight, config_path=args.config_path)
    log("info", "loading mimi")
    mimi = checkpoint_info.get_mimi(device=args.device)
    log("info", "mimi loaded")

    text_tokenizer = checkpoint_info.get_text_tokenizer()

    log("info", "loading moshi")
    lm = checkpoint_info.get_moshi(device=args.device, dtype=args.dtype, fuse_lora=args.fuse_lora)
    log("info", "moshi loaded")

    text_procemb = copy.deepcopy(lm.text_emb.weight.data)
    audio_procemb = [copy.deepcopy(a_emb.weight.data) for a_emb in lm.emb[:8]]
    
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

    # Speaker map + idle slices were loaded + validated up-front, right
    # after the YAML config read, so a misconfigured experiment dir fails
    # the whole process before any model load runs.

    
        
    
    
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
    # 
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
        vad_use_face_logits=trainer_args.vad_use_face_logits if hasattr(trainer_args, 'vad_use_face_logits') else False,
        body_parts=body_parts,
        bp_dist=None, 
        textaudio_emb_freeze=trainer_args.textaudio_emb_freeze if hasattr(trainer_args, 'textaudio_emb_freeze') else False,
    )
    
    gesture_lm = checkpoint_info.get_gesture_lm(
        device=args.device,
        dtype=None, 
        gesture_lm_kwargs=gesture_lm_kwargs,
    )
    
    log("info", "gesturelm loaded")

    
    model_path = trainer_args.deps_path + 'smplx_2020/'
    model_path = Path(model_path).resolve()
    
    log("info", "starting motion visualization server")
    mixamo_path: str | None = args.mixamo_character
    if mixamo_path is not None and not Path(mixamo_path).is_file():
        # Treat as slug -> assets_dep/mixamo_characters_release/<slug>.npz
        slug_path = Path("assets_dep/mixamo_characters_release") / f"{mixamo_path}.npz"
        if not slug_path.is_file():
            raise FileNotFoundError(
                f"--mixamo-character {args.mixamo_character!r}: not a file and "
                f"slug not found at {slug_path}"
            )
        mixamo_path = str(slug_path)
    motion_proc = start_motion_server(
        args.host,
        args.viser_port,
        args.motion_vis_port,
        model_path,
        mixamo_character=mixamo_path,
    )
    try:
        state = ServerState(checkpoint_info.model_type, mimi, text_tokenizer, lm, gesture_lm, gesture_codecs,
                            idle_upper_slice, idle_face_slice,
                            args.cfg_coef, args.glm_cfg_coef, args.motion_vis_port, args.host, args.device,
                            idle_motion_replace=args.idle_motion_replace,
                            gesturelm_kwargs=checkpoint_info.gesture_lm_config,
                            **checkpoint_info.lm_gen_config)
        log("info", "warming up the model")
        state.warmup()
        app = web.Application()
        app.router.add_get("/api/chat", state.handle_chat)
        static_path: None | str = None

        # args.static = "none"  # disable static content serving for now
        if args.static is None:
            log("info", "retrieving the static content")
            dist_tgz = hf_hub_download("kyutai/moshi-artifacts", "dist.tgz", cache_dir="./assets_dep/kyutai_cache")
            dist_tgz = Path(dist_tgz)
            dist = dist_tgz.parent / "dist"
            if not dist.exists():
                with tarfile.open(dist_tgz, "r:gz") as tar:
                    tar.extractall(path=dist_tgz.parent)
            static_path = str(dist)
        elif args.static != "none":
            # When set to the "none" string, we don't serve any static content.
            static_path = args.static
        if static_path is not None:
            static_dir = Path(static_path)
            audio_index_path = static_dir / "index.html"
            assets_dir = static_dir / "assets"
            if not audio_index_path.exists():
                raise RuntimeError(f"Audio UI index not found at {audio_index_path}")
            if not assets_dir.exists():
                raise RuntimeError(f"Audio UI assets directory not found at {assets_dir}")

            # When --minimal-audio-ui is set, inject a tiny <style> that
            # hides the transcript box (.player-text) and the Server Audio
            # Stats panel (.player-stats) -- both are unique class hooks in
            # the bundled Moshi UI. WS protocol is unchanged; this is pure
            # presentation.
            # Collapse Moshi's inner .main-grid down to one column and
            # two rows (controls / player) so the now-empty right column
            # and the unused extra row tracks stop reserving space. Also
            # force the .player to stretch to fill its cell -- otherwise
            # the original `justify-items: center` + the inner
            # `1fr 1fr` row split make the bordered box render slightly
            # narrower than its area, producing the odd border seam.
            minimal_ui_style = (
                "<style>"
                ".player-text,.player-stats{display:none !important;}"
                ".main-grid{"
                "grid-template-columns:1fr !important;"
                "grid-template-rows:auto 1fr !important;"
                "grid-template-areas:'controls' 'player' !important;"
                "justify-items:stretch !important;"
                "align-items:stretch !important;"
                "}"
                ".player{justify-self:stretch !important;align-self:stretch !important;"
                "overflow:hidden !important;}"
                # .server-audio / .user-audio use aspect-square + h-N/6,
                # so when .player is tall+narrow they overflow horizontally
                # and the canvas pixels poke past the right border. Cap
                # them at the parent's width (height left untouched so the
                # canvas still has a drawing box).
                ".player>.server-audio,.player>.user-audio{"
                "max-width:100% !important;}"
                "</style>"
            )

            # Always-injected style: preserve newlines + wrap in the
            # transcript box so the server-inserted `\n` between sentences
            # actually renders as a line break (default white-space would
            # collapse it to a space).
            base_ui_style = (
                "<style>"
                ".player-text{white-space:pre-wrap !important;}"
                "</style>"
            )

            async def handle_dashboard(request):
                # The bundled Moshi frontend expects to run on path "/".
                # Serve it on "/" when explicitly requested via query mode.
                if request.query.get("mode") == "audio":
                    html = audio_index_path.read_text()
                    injected = base_ui_style
                    if args.minimal_audio_ui:
                        injected += minimal_ui_style
                    html = html.replace("</head>", f"{injected}</head>", 1)
                    return web.Response(
                        text=html, content_type="text/html",
                        headers={"Cache-Control": "no-store"},
                    )

                request_host = request.host.split(":", 1)[0]
                motion_ui_url = f"http://{request_host}:{args.viser_port}"
                html = build_dashboard_html(
                    "/?mode=audio", motion_ui_url,
                    minimal_audio_ui=args.minimal_audio_ui,
                )
                return web.Response(
                    text=html,
                    content_type="text/html",
                    headers={"Cache-Control": "no-store"},
                )

            async def handle_audio(_):
                raise web.HTTPFound("/?mode=audio")

            async def handle_dashboard_probe(_):
                return web.Response(
                    text="gest-server-face2 dashboard active",
                    content_type="text/plain",
                    headers={"Cache-Control": "no-store"},
                )

            log("info", f"serving combined dashboard with audio static content from {static_dir}")
            app.router.add_get("/", handle_dashboard)
            app.router.add_get("/dashboard", handle_dashboard)
            app.router.add_get("/audio", handle_audio)
            app.router.add_get("/__dashboard_probe", handle_dashboard_probe)
            app.router.add_static(
                "/assets", path=str(assets_dir), follow_symlinks=True, name="audio-assets"
            )
        protocol = "http"
        ssl_context = None
        if args.ssl is not None:
            import ssl

            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            cert_file = os.path.join(args.ssl, "cert.pem")
            key_file = os.path.join(args.ssl, "key.pem")
            ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
            protocol = "https"

        log("info", f"Access the combined dashboard at {protocol}://{args.host}:{args.port}")
        log("info", f"Dashboard probe endpoint at {protocol}://{args.host}:{args.port}/__dashboard_probe")
        log("info", f"Access the audio UI directly at {protocol}://{args.host}:{args.port}/audio")
        log("info", f"Access the motion UI directly at http://{args.host}:{args.viser_port}")
        if setup_tunnel is not None:
            tunnel_kwargs = {}
            if "share_server_tls_certificate" in inspect.signature(setup_tunnel).parameters:
                tunnel_kwargs["share_server_tls_certificate"] = None
            tunnel = setup_tunnel('localhost', args.port, tunnel_token, None, **tunnel_kwargs)
            log("info", f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.")
            log("info", "Note that this tunnel goes through the US and you might experience high latency in Europe.")
            log("warning", "Tunnel URL only proxies this server port. The motion iframe still points to the local Viser port.")
        web.run_app(app, port=args.port, ssl_context=ssl_context)
    finally:
        motion_proc.terminate()
        motion_proc.wait(timeout=5)


with torch.no_grad():
    main()
