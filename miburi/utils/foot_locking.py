"""Causal foot locking for the live demo.

MIBURI's Gesture LM predicts lower-body pose and a translation velocity
independently, so a planted foot tends to drift a few centimetres per second
across the floor. This removes that slide online, one frame at a time.

Ported from MPI's `body-models-postprocess`
(gitlab.mpi-klsb.mpg.de/body-models/body-models-postprocess), whose online
locker is Holden's causal method -- see
https://theorangeduck.com/page/inverse-kinematics-foot-locking. The reference
implementation's solver core is Rust (pyo3); this is a numpy port of the
online path only, so the demo needs no Rust toolchain. It is validated
frame-by-frame against the reference in `tests/test_foot_locking.py`.

Two departures from the reference, both forced by running live:

* Rig setup there reads MPI's internal `body-models` package. Here it is
  derived from the `smplx` model the demo already loads (`build_foot_rig`).
* Its contact detector is offline -- `np.gradient` over a whole clip plus a
  *centred* median filter. `CausalFootContacts` is a streaming equivalent:
  backward-difference velocity, causal median, and hysteresis.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

UP_AXIS = {"x": 0, "y": 1, "z": 2}

# Widening the reachable range by this much keeps the two-bone solve away
# from the singularity at full leg extension.
IK_SOFTENING = 0.005

# SMPL-X kinematic indices. Asserted against the model in build_foot_rig
# rather than trusted, since the solver requires exactly these chains.
SMPLX_ANKLES = (7, 8)
SMPLX_TOES = (10, 11)


# --------------------------------------------------------------------------
# quaternion helpers, xyzw to match the reference (glam)
# --------------------------------------------------------------------------

def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float32)


def _qnorm(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _qnorm(_qmul(a, b))


def _qconj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u = q[:3]
    t = 2.0 * np.cross(u, v)
    return (v + q[3] * t + np.cross(u, t)).astype(np.float32)


def _from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    half = 0.5 * angle
    return np.array([*(axis * math.sin(half)), math.cos(half)], dtype=np.float32)


def _unit(v: np.ndarray) -> np.ndarray:
    n2 = float(v @ v)
    if n2 <= 1e-16:
        raise ValueError("cannot normalize a zero-length vector")
    return (v / math.sqrt(n2)).astype(np.float32)


def _between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Shortest rotation taking `source` onto `target`."""
    source, target = _unit(source), _unit(target)
    product = float(source @ target)
    if product < -1.0 + 1e-7:
        # Antiparallel: any perpendicular axis is a valid half turn.
        axis = np.cross(source, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if float(axis @ axis) <= 1e-16:
            axis = np.cross(source, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        return _from_axis_angle(_unit(axis), math.pi)
    axis = np.cross(source, target)
    return _qnorm(np.array([axis[0], axis[1], axis[2], 1.0 + product], dtype=np.float32))


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return math.acos(min(1.0, max(-1.0, float(a @ b) / denom)))


def _cosine(left: float, right: float, opposite: float) -> float:
    """Interior angle opposite `opposite` in a triangle (law of cosines)."""
    value = (left * left + right * right - opposite * opposite) / (2.0 * left * right)
    return math.acos(min(1.0, max(-1.0, value)))


# --------------------------------------------------------------------------
# rig
# --------------------------------------------------------------------------

@dataclass
class Leg:
    ankle: int
    toe: int
    ankle_height: float   # rest height of the ankle above the sole
    toe_height: float     # rest height of the toe joint above the sole
    toe_end: np.ndarray   # virtual toe tip, in the toe's local frame
    toe_end_height: float


@dataclass
class FootRig:
    parents: np.ndarray   # (J,) int64, -1 at the root
    offsets: np.ndarray   # (J, 3) rest offset in the parent's frame
    legs: tuple[Leg, Leg]
    up: int
    # Original SMPL-X index of each joint in this rig. Set when the rig has
    # been reduced to the leg chains; None when it spans the whole body.
    chain: np.ndarray | None = None


def reduce_to_legs(rig: FootRig) -> FootRig:
    """Keep only the joints the solver can reach.

    Locking writes hip, knee, ankle and toe, and reads nothing outside those
    two chains, so running forward kinematics over all 22 body joints is
    wasted work -- and this runs 25 times a second inside the render loop.
    Restricting to root + both toe-ankle-knee-hip chains (9 joints) leaves
    results bit-identical while cutting the FK cost by more than half.
    """
    keep = {0}
    for leg in rig.legs:
        joint = leg.toe
        while joint >= 0:
            keep.add(int(joint))
            joint = int(rig.parents[joint])
    chain = np.array(sorted(keep), dtype=np.int64)   # sorted == topological here
    index = {int(original): new for new, original in enumerate(chain)}

    parents = np.array([-1 if rig.parents[j] < 0 else index[int(rig.parents[j])]
                        for j in chain], dtype=np.int64)
    legs = tuple(
        Leg(ankle=index[leg.ankle], toe=index[leg.toe],
            ankle_height=leg.ankle_height, toe_height=leg.toe_height,
            toe_end=leg.toe_end, toe_end_height=leg.toe_end_height)
        for leg in rig.legs
    )
    return FootRig(parents=parents, offsets=rig.offsets[chain].copy(),
                   legs=legs, up=rig.up, chain=chain)


def build_foot_rig(smplx_model, world_up_axis: str = "y", joint_count: int = 22) -> FootRig:
    """Derive the rig the locker needs from the demo's own SMPL-X model.

    The reference reads this from MPI's internal `body-models` package. All of
    it is available on the `smplx` model: the kinematic tree, the rest joints
    (via the J_regressor on the neutral template), and the skinning weights
    used to find where the sole of the foot actually is.
    """
    up = UP_AXIS[world_up_axis]
    parents = smplx_model.parents.detach().cpu().numpy().astype(np.int64)[:joint_count].copy()
    parents[0] = -1

    v_template = smplx_model.v_template.detach().cpu().numpy().astype(np.float64)
    J_regressor = smplx_model.J_regressor.detach().cpu().numpy().astype(np.float64)
    rest_joints = (J_regressor @ v_template)[:joint_count]

    # SMPL-X rest pose is identity-rotated, so a joint's offset is just its
    # displacement from its parent.
    offsets = rest_joints.copy()
    offsets[1:] = rest_joints[1:] - rest_joints[parents[1:]]

    weights = smplx_model.lbs_weights.detach().cpu().numpy()
    dominant = weights.argmax(axis=1)

    legs = []
    for side, ankle, toe in zip(("left", "right"), SMPLX_ANKLES, SMPLX_TOES):
        knee = int(parents[ankle])
        if int(parents[toe]) != ankle or knee < 0 or int(parents[knee]) < 0:
            raise ValueError(f"{side} foot is not a toe-ankle-knee-hip chain")

        foot_verts = v_template[(dominant == ankle) | (dominant == toe)]
        if len(foot_verts) == 0:
            raise ValueError(f"cannot calibrate {side} foot: no foot-dominant vertices")
        floor = float(foot_verts[:, up].min())

        toe_end = _calibrate_toe_end(rest_joints, v_template, weights, ankle, toe, up, side)
        toe_end_world = rest_joints[toe] + toe_end   # rest rotation is identity
        legs.append(Leg(
            ankle=int(ankle),
            toe=int(toe),
            ankle_height=float(rest_joints[ankle][up] - floor),
            toe_height=float(rest_joints[toe][up] - floor),
            toe_end=toe_end.astype(np.float32),
            toe_end_height=float(toe_end_world[up] - floor),
        ))

    return FootRig(
        parents=parents,
        offsets=offsets.astype(np.float32),
        legs=(legs[0], legs[1]),
        up=up,
    )


def _calibrate_toe_end(rest_joints, vertices, weights, ankle, toe, up, side) -> np.ndarray:
    """Locate a virtual toe tip from the skinned rest mesh.

    SMPL-X has no joint at the end of the toe, but locking needs one so the
    forefoot can be kept above the floor. Take the toe-dominant vertices,
    find their weighted centre, and push it out to the most distal point
    along the foot's forward direction.
    """
    dominant_toe = weights.argmax(axis=1) == toe
    if not dominant_toe.any():
        raise ValueError(f"cannot calibrate {side} toe end: no toe-dominant vertices")

    forward = rest_joints[toe] - rest_joints[ankle]
    forward[up] = 0.0
    length = float(np.linalg.norm(forward))
    if length <= 1e-8:
        raise ValueError(f"cannot calibrate {side} toe end: degenerate foot direction")
    forward = forward / length

    verts = vertices[dominant_toe]
    w = weights[dominant_toe, toe]
    centre = np.average(verts, axis=0, weights=w)
    distal = float((verts @ forward).max())
    tip = centre + (distal - float(centre @ forward)) * forward
    return (tip - rest_joints[toe]).astype(np.float32)


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------

def _forward(parents, rotations, offsets, root):
    """Global rotations and positions for the whole chain."""
    n = len(parents)
    glob = np.empty((n, 4), dtype=np.float32)
    pos = np.empty((n, 3), dtype=np.float32)
    for j in range(n):
        p = int(parents[j])
        if p < 0:
            glob[j] = rotations[j]
            pos[j] = root
        else:
            glob[j] = _compose(glob[p], rotations[j])
            pos[j] = pos[p] + _qrot(glob[p], offsets[j])
    return glob, pos


@dataclass
class _LockState:
    """Holden's inertialized lock: the foot has a target, and an offset from
    it that decays smoothly, so locking and unlocking never pop."""
    position: np.ndarray
    velocity: np.ndarray
    input_position: np.ndarray
    input_velocity: np.ndarray
    offset_position: np.ndarray
    offset_velocity: np.ndarray
    time: float
    contact: np.ndarray
    locked: bool

    @staticmethod
    def new(position: np.ndarray) -> "_LockState":
        z = np.zeros(3, dtype=np.float32)
        return _LockState(position.astype(np.float32).copy(), z.copy(),
                          position.astype(np.float32).copy(), z.copy(),
                          z.copy(), z.copy(), 0.0, z.copy(), False)


def _inertialize_weights(t: float, blend_time: float):
    bt = max(blend_time, 1e-8)
    w0 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    w1 = (t**3 - 2.0 * t**2 + t) * blend_time
    w2 = (6.0 * t**2 - 6.0 * t) / bt
    w3 = 3.0 * t**2 - 4.0 * t + 1.0
    return w0, w1, w2, w3


def _inertialize_update(state, input_position, input_velocity, dt, blend_time):
    t = min(1.0, max(0.0, (state.time + dt) / max(blend_time, 1e-8)))
    w0, w1, w2, w3 = _inertialize_weights(t, blend_time)
    state.position = input_position + state.offset_position * w0 + state.offset_velocity * w1
    state.velocity = input_velocity + state.offset_position * w2 + state.offset_velocity * w3
    state.time += dt


def _inertialize_transition(state, src_pos, src_vel, dst_pos, dst_vel, blend_time):
    t = min(1.0, max(0.0, state.time / max(blend_time, 1e-8)))
    w0, w1, w2, w3 = _inertialize_weights(t, blend_time)
    old_velocity = state.offset_velocity
    # NOTE: the reference feeds the *already updated* offset_position into the
    # velocity line. Kept deliberately -- it is what the Rust does, and the
    # port is validated against it.
    state.offset_position = src_pos + state.offset_position * w0 + old_velocity * w1 - dst_pos
    state.offset_velocity = src_vel + state.offset_position * w2 + old_velocity * w3 - dst_vel
    state.time = 0.0


def _update_lock_state(state, input_position, input_contact, contact_height, up,
                       dt, unlock_distance, lock_distance, blend_time):
    """`contact_height` is absolute world height, i.e. ground + rest height."""
    state.input_velocity = (input_position - state.input_position) / max(dt, 1e-8)
    state.input_position = input_position
    if state.locked:
        dest_pos, dest_vel = state.contact, np.zeros(3, dtype=np.float32)
    else:
        dest_pos, dest_vel = state.input_position, state.input_velocity
    _inertialize_update(state, dest_pos, dest_vel, dt, blend_time)

    distance = float(np.linalg.norm(state.position - state.input_position))
    if not state.locked and input_contact and distance < lock_distance:
        state.locked = True
        contact = state.input_position.copy()
        contact[up] = contact_height
        state.contact = contact
        _inertialize_transition(state, state.input_position, state.input_velocity,
                                state.contact, np.zeros(3, dtype=np.float32), blend_time)
    elif state.locked and (not input_contact or distance > unlock_distance):
        state.locked = False
        _inertialize_transition(state, state.contact, np.zeros(3, dtype=np.float32),
                                state.input_position, state.input_velocity, blend_time)


def _two_bone(parents, rotations, offsets, root, hip, knee, ankle, target):
    """Analytic hip/knee solve placing the ankle at `target`."""
    glob, pos = _forward(parents, rotations, offsets, root)
    a, b, c = pos[hip], pos[knee], pos[ankle]
    ab, bc, ac, at = b - a, c - b, c - a, target - a
    lab, lbc = float(np.linalg.norm(ab)), float(np.linalg.norm(bc))
    lac, lat = float(np.linalg.norm(ac)), float(np.linalg.norm(at))
    if min(lab, lbc, lac, lat) <= 1e-8:
        raise ValueError("cannot solve a degenerate leg chain")

    down = ac / lac
    side = _qrot(glob[knee], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    forward = np.cross(down, side)
    fl = float(np.linalg.norm(forward))
    if fl <= 1e-8:
        raise ValueError("knee side vector is parallel to the leg")
    axis = _unit(np.cross(down, forward / fl))

    hip_delta = _cosine(lab, lat, lbc) - _angle(ac, ab)
    knee_delta = _cosine(lab, lbc, lat) - _angle(a - b, c - b)
    aim = _between(ac, at)

    new_hip = _compose(aim, _compose(_from_axis_angle(axis, hip_delta), glob[hip]))
    rotations[hip] = _compose(_qconj(glob[int(parents[hip])]), new_hip)
    new_knee = _compose(_from_axis_angle(axis, knee_delta), glob[knee])
    rotations[knee] = _compose(_qconj(glob[hip]), new_knee)


def _solve_leg(parents, rotations, offsets, root, leg: Leg, target_toe, up, softening,
               ground: float = 0.0):
    knee = int(parents[leg.ankle])
    hip = int(parents[knee])
    _, pos = _forward(parents, rotations, offsets, root)

    target_toe = target_toe.copy()
    target_toe[up] = max(target_toe[up], ground + leg.toe_height)
    target_ankle = target_toe + (pos[leg.ankle] - pos[leg.toe])
    target_ankle[up] = max(target_ankle[up], ground + leg.ankle_height)

    # Ease off before the leg locks straight, so the knee never snaps.
    hip_position = pos[hip]
    max_extension = float(np.linalg.norm(pos[leg.ankle] - hip_position))
    target_vector = target_ankle - hip_position
    target_length = float(np.linalg.norm(target_vector))
    if target_length > max_extension - softening and target_length > 1e-8:
        excess = max(target_length - max_extension + softening, 0.0)
        saturation = 1.0 - math.exp(-excess / softening)
        clamped = max_extension - softening + softening * saturation
        target_ankle = hip_position + target_vector * (clamped / target_length)

    _two_bone(parents, rotations, offsets, root, hip, knee, leg.ankle, target_ankle)

    # Re-aim the ankle so the toe lands on target.
    glob, pos = _forward(parents, rotations, offsets, root)
    delta = _between(pos[leg.toe] - pos[leg.ankle], target_toe - pos[leg.ankle])
    rotations[leg.ankle] = _compose(_qconj(glob[knee]), _compose(delta, glob[leg.ankle]))

    # Then lift the forefoot out of the floor.
    glob, pos = _forward(parents, rotations, offsets, root)
    current_end = pos[leg.toe] + _qrot(glob[leg.toe], leg.toe_end)
    target_end = current_end.copy()
    target_end[up] = max(target_end[up], ground + leg.toe_end_height)
    toe_delta = _between(current_end - pos[leg.toe], target_end - pos[leg.toe])
    rotations[leg.toe] = _compose(_qconj(glob[leg.ankle]),
                                  _compose(toe_delta, glob[leg.toe]))


class OnlineFootLocker:
    """Stateful causal foot locker. Feed it one frame at a time.

    Only leg rotations are altered; the root translation is left exactly as
    the model produced it, so this composes with MIBURI's own
    velocity-to-position integration rather than fighting it.
    """

    def __init__(self, rig: FootRig, *, fps: float,
                 contact_threshold: float = 0.75, unlock_distance: float = 0.25,
                 lock_distance: float = 0.01, blend_time: float = 0.5):
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        if not 0.0 <= lock_distance <= unlock_distance:
            raise ValueError("lock_distance must be between zero and unlock_distance")
        if blend_time < 0.0:
            raise ValueError("blend_time must be non-negative")
        self.rig = rig
        self.dt = 1.0 / fps
        self.contact_threshold = contact_threshold
        self.unlock_distance = unlock_distance
        self.lock_distance = lock_distance
        self.blend_time = blend_time
        self._states: list[_LockState | None] = [None, None]
        # Lock/unlock churn. Each transition blends the toe between its
        # locked and free positions, which is itself motion -- so frequent
        # flipping can add more slide than it removes.
        self.transitions = 0

    def reset(self) -> None:
        """Drop causal history; the rig is unaffected."""
        self._states = [None, None]

    def step(self, rotations: np.ndarray, root: np.ndarray,
             contacts: np.ndarray, ground: float = 0.0) -> np.ndarray:
        """Correct one frame.

        Args:
            rotations: (J, 4) local joint quaternions, xyzw, root first.
            root: (3,) world position of the root joint.
            contacts: (2,) left/right contact confidence in [0, 1].
            ground: world height of the floor along the up axis. The
                reference assumes zero; the demo integrates root translation
                from a predicted velocity, so its floor sits wherever that
                integration started. Rest foot heights are measured relative
                to this.
        Returns:
            (J, 4) corrected local rotations.
        """
        rotations = np.array(rotations, dtype=np.float32, copy=True)
        root = np.asarray(root, dtype=np.float32)
        contacts = np.asarray(contacts, dtype=np.float32)
        if rotations.shape != (len(self.rig.parents), 4):
            raise ValueError(f"rotations must be ({len(self.rig.parents)}, 4), "
                             f"got {rotations.shape}")
        if contacts.shape != (2,):
            raise ValueError(f"contacts must have shape (2,), got {contacts.shape}")

        _, positions = _forward(self.rig.parents, rotations, self.rig.offsets, root)
        for foot, leg in enumerate(self.rig.legs):
            source_toe = positions[leg.toe].copy()
            if self._states[foot] is None:
                self._states[foot] = _LockState.new(source_toe)
            state = self._states[foot]
            was_locked = state.locked
            _update_lock_state(
                state, source_toe, bool(contacts[foot] > self.contact_threshold),
                ground + leg.toe_height, self.rig.up, self.dt,
                self.unlock_distance, self.lock_distance, self.blend_time,
            )
            if state.locked != was_locked:
                self.transitions += 1
            _solve_leg(self.rig.parents, rotations, self.rig.offsets, root,
                       leg, state.position, self.rig.up, IK_SOFTENING, ground)
        return rotations


# --------------------------------------------------------------------------
# contacts
# --------------------------------------------------------------------------

class CausalFootContacts:
    """Streaming version of Holden's velocity-and-height contact heuristic.

    The reference detector is offline: it differentiates the whole clip with
    `np.gradient` and applies a *centred* median filter, both of which read
    future frames. This keeps the same signals and entry thresholds but only
    ever looks backwards -- a backward difference for velocity and a trailing
    median over `median_seconds`.

    It also adds hysteresis, which the offline version does not need. A
    contact that flickers is worse than no contact at all: every lock and
    unlock blends the toe between its held and free positions over
    `blend_time`, and that blend is itself motion. Measured on live MIBURI
    output, windows with 0-1 transitions removed 75-82% of planted-foot
    slide while windows with 3-8 transitions removed none and sometimes
    added some. So a foot already in contact keeps it until the evidence is
    clearly worse, by `release_factor`.
    """

    def __init__(self, *, fps: float, up: int,
                 velocity_threshold: float = 0.15,
                 height_threshold: float = 0.10,
                 release_factor: float = 2.0,
                 ground_height: float | None = None,
                 ground_window_seconds: float = 2.0,
                 toe_rest_height: float = 0.0,
                 median_seconds: float = 5.0 / 60.0):
        self.fps = fps
        self.up = up
        self.velocity_threshold = velocity_threshold
        self.height_threshold = height_threshold
        if release_factor < 1.0:
            raise ValueError("release_factor must be at least 1.0")
        self.release_factor = release_factor
        # `None` calibrates the floor from what the feet actually do. The
        # demo integrates root translation from a predicted velocity with an
        # arbitrary starting offset, so the world height of the ground is not
        # known ahead of time, and it differs again for a Mixamo character
        # whose rest frame sits at a different height from SMPL-X's.
        self.ground_height = ground_height
        # A planted toe *joint* rests this far above the sole, so the floor
        # is that much below the lowest toe height we observe.
        self.toe_rest_height = toe_rest_height
        window = max(1, round(median_seconds * fps))
        window += (window % 2 == 0)     # odd, so the median has no tie
        self._history: deque[np.ndarray] = deque(maxlen=window)
        self._previous: np.ndarray | None = None
        self._held = np.zeros(2, dtype=bool)
        self._floor: deque[float] = deque(maxlen=max(1, round(ground_window_seconds * fps)))

    def reset(self) -> None:
        self._history.clear()
        self._floor.clear()
        self._previous = None
        self._held = np.zeros(2, dtype=bool)

    @property
    def floor(self) -> float:
        """Current estimate of the world height of the ground."""
        if self.ground_height is not None:
            return self.ground_height
        if not self._floor:
            return 0.0
        return min(self._floor) - self.toe_rest_height

    def _observe_floor(self, toes: np.ndarray) -> float:
        """Track the lowest toe height seen recently as the floor estimate.

        A trailing window rather than an all-time minimum, so one crouch does
        not permanently drop the estimated floor.
        """
        if self.ground_height is None:
            self._floor.append(float(toes[:, self.up].min()))
        return self.floor

    def step(self, toe_positions: np.ndarray) -> np.ndarray:
        """One frame of (2, 3) left/right toe world positions -> (2,) contacts."""
        toes = np.asarray(toe_positions, dtype=np.float32).reshape(2, 3)
        if self._previous is None:
            speed = np.zeros(2, dtype=np.float32)      # no velocity on frame one
        else:
            speed = np.linalg.norm((toes - self._previous) * self.fps, axis=-1)
        self._previous = toes.copy()

        height = toes[:, self.up] - self._observe_floor(toes) - self.toe_rest_height

        # Schmitt trigger: it takes `velocity_threshold` to acquire a contact
        # but `release_factor` times that to give one up.
        enter = (speed < self.velocity_threshold) & (height < self.height_threshold)
        stay = ((speed < self.velocity_threshold * self.release_factor)
                & (height < self.height_threshold * self.release_factor))
        raw = np.where(self._held, stay, enter)
        self._held = raw

        self._history.append(raw.astype(np.float32))
        return np.median(np.stack(self._history), axis=0).astype(np.float32)


def toe_positions(rig: FootRig, rotations: np.ndarray, root: np.ndarray) -> np.ndarray:
    """World positions of both toe joints, for the contact detector."""
    _, positions = _forward(rig.parents, np.asarray(rotations, dtype=np.float32),
                            rig.offsets, np.asarray(root, dtype=np.float32))
    return np.stack([positions[leg.toe] for leg in rig.legs])
