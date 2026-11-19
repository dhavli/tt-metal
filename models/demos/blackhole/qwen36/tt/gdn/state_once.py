# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_MTP_GDN_STATE_ONCE=1: the MTP verify reads each row's GDN recurrent state ONCE and writes none; the commit
recomputes the accepted state (docs/MTP-BATCH.md section 16).

Why: c154 ([source validation record]) put the GDN layers + norms at
18.9 ms of a 39.7 ms verify (B=8, bucket 8, T=3) against a ~6.5 ms weight floor, and c140b measured the recurrence at
2.2 ms per extra candidate at Bw=8 and 1.2 ms at Bw=4: it scales with rows, i.e. recurrent-state traffic. The
sequential verify (TPGatedDeltaNet.forward_verify) runs decode_gated_delta_rule_packed(inplace_state=False) once per
candidate: each launch reads a full per-row state and writes a new one (hist_j); the QWEN36_MTP_STATE_SLOT commit
then reads hist_{idx[b]} and writes it over rec_state.

Programs (ttnn.generic_op, JIT kernels in kernels/state_once/, no image change):
  verify  (MODE 0)  per head instance: the pre-verify state read once (rows < Bw of rec_state, in place: no slice
                    copy), tokens 0..T-1 with the state carried in L1 in the IO dtype, o of every token written to one
                    ROW_MAJOR [T*Bw, 1, Nv, Dv] tensor. No state is written. q/k/v/beta/g come straight from the
                    QWEN36_MTP_CONV_MULTI batch-M conv_gates output in L1 (row u = t*Bw + b), so its two split
                    programs go. The commit's inputs leave as one small stick per (row, head, token): cin ROW_MAJOR
                    [T*Bw, 1, Nv, Dk+Dv+32] in DRAM = [k row | v row | beta, g, 0...] exactly as gathered.
  commit  (MODE 1)  per head instance: n = idx[b] + 1 tokens (idx = the slot page gslot: the accepted verify depth;
                    >= T means keep the row, n = 0), the base state read once, tokens 0..n-1 recomputed from the cin
                    sticks (k/v tiles rebuilt as the gather left them; no q, no o), the state after token n-1 written
                    over rec_state row b. Same kernels, same op order, same
                    rounding between tokens as the verify and as the sequential launches, so the committed bytes are
                    the sequential hist_{idx[b]} bytes. The token count is a per-row runtime VALUE read on device from
                    the idx page (reader -> compute through a one-word CB, read_tile_value), not a beta=0 / g=0 mask:
                    nothing past row b's accepted length touches its state.

State traffic per GDN layer, in units of one Bw-row state (Bw x Nv x Dk x Dv x io bytes):
  sequential + slot commit: verify T reads + T writes, commit 1 read + 1 write = 2T + 2
                            (+2 for the [0:Bw] slice copy of rec_state when Bw < Bmax)
  STATE_ONCE:               verify 1 read, commit 1 read + 1 write = 3 (SKIP rows read and write nothing)
                            (+ cin: 576 B per (row, head, token) written by the verify, n x 576 B read by the commit)
The conv taps are untouched by this lever: CONV_MULTI's window tensors and its commit (state_slot.commit_conv_multi)
stay as they are.

MUSTFAIL (QWEN36_MTP_GDN_STATE_ONCE_MUSTFAIL=1): the commit gives the LAST row of the bucket one token too many
(n+1; n-1 when n == T). The warmup self-test compares the served configuration bitwise and REFUSES the boot.

QWEN36_MTP_GDN_SO_T0=1 (needs QWEN36_MTP_GDN_STATE_ONCE=1; docs/MTP-BATCH.md section 17): token 0 of every in-bucket
row is always committed (mtp_batch_host.slot_index is strictly one-hot for b < Bw, so idx[b] in [0, T-1]), so the
verify writes the state after token 0 IN PLACE over rec_state row b (the instance read that row into L1 first, and no
other instance reads it), and the commit recomputes only tokens 1..idx[b] from the updated row: n = idx[b] tokens
(was idx[b] + 1); n == 0 leaves the row as the verify left it. The state leaves the verify in the IO dtype from the
same fp32 value and the same pack as the L1 carry, and the commit unpacks it as the verify unpacks the carry, so the
committed bytes are today's. c156a: the commit is 6.12 ms/step at Bw=8 (3.84/4.53/6.12 at Bw 2/4/8), the recompute.
Defined semantics of a keep entry (SKIP or >= T) INSIDE the bucket under SO_T0: the row keeps the state after token
0 (the verify already wrote it); in production such an entry cannot occur (slot_index refuses a row that is not
one-hot); the self-test's pages carry such entries and check exactly this (t0_reference_page). Rows >= Bw are never
visited by either program. VERIFY_DOUBLE's second verify passes t0=False (it must not apply token 0 twice).
MUSTFAIL (QWEN36_MTP_GDN_SO_T0_MUSTFAIL=1): the verify writes the state after token 1 (not 0) for the LAST row of the
bucket; the warmup self-test REFUSES the boot.

QWEN36_MTP_GDN_SO_SFPU_ROW0=1 (needs QWEN36_MTP_GDN_STATE_ONCE=1; docs/MTP-BATCH.md section 20): the three full-tile
SFPU passes of every token (inv_rms of the q and k L2 norms: +eps, rsqrt, *scale; exp of the decay g) run on SFPU
iteration 0 of face 0 only (VectorMode::None, ITERATIONS=1: the 32 lanes that hold element [0,0]) instead of all 1024
elements. Why (measured, [source validation record]): at Bw=8 T=3 the verify program
is 118 us/layer on one card, the compute is busy from 18.5 us to the end at ~15.5 us per token-instance, and the math
RISC spends 1.60 + 1.45 + 1.49 = 4.5 us of each token in those three SFPU passes (29%). Only [0,0] of the exp tile is
read (mul_tiles_bcast_scalar) and only column 0 of the inv_rms tile is read (mul_tiles_bcast_cols), multiplying q/k
tiles whose rows 1..31 are +0 (the gather zeroes them; so do the commit's stick rebuilds). The rows of the inv_rms
input that this pass no longer touches hold the row sums of those zero rows, +0, so q/k rows 1..31 stay +0 x +0 = +0,
as before (+0 x a finite factor). Element [0,0] runs the same SFPU code on the same value. The bytes of every output
are therefore today's by construction (ttsim bitwise gate + the warmup self-test against the sequential op).
Applies to the verify (MODE 0) and both commits (MODE 1: state_once.commit, commit_pack); read per call. Off: the
compute source is today's text (no define), so the program and its hash are today's.
MUSTFAIL (QWEN36_MTP_GDN_SO_SFPU_ROW0_MUSTFAIL=1): the decay skips exp (h = state * g, not state * exp(g)); the warmup
self-test REFUSES the boot. The self-test also runs that wrong variant explicitly at every boot (mustfail_sfpu_row0)
and refuses unless it differs.

QWEN36_MTP_GDN_SO_FUSE=1 (needs QWEN36_MTP_GDN_STATE_ONCE=1; docs/MTP-BATCH.md section 21): the per-token state chain of
the compute keeps its intermediates in DST (dest-reuse ELWMUL/ELWSUB/ELWADD after the matmuls) instead of packing each to
an fp32 CB and unpacking it for the next op: the state copy goes into the decay, v_read and v - v_read are one DST section (beta stays the stock op), the
rank-1 outer product and the add are one, and new_h packs straight to the fp32 CB (verify: o) and the io carry / sout
(the commit packs no fp32 new_h). Why (measured, op-anatomy-gdn-state-once-20261006T045611Z): the compute is busy back
to back at ~13-16 us per token-instance; ~0.2 us fixed per dependent op and fp32 pack at ~64 cycles/tile; the four
removed 16-tile passes (state copy, outer pack, carry copy, T0 copy) and one 4-tile pass are ~25% of a token. Every
FPU op keeps its operands in the same source register with the same values and fidelity; bitwise equality with today's
path is CHECKED (ttsim sim_mtp_gdn_state_once --fuse; the warmup self-test), not assumed: it holds only if DST -> Src
moves (MOVD2B; MOVD2A only for bf16-exact state values; Src format pinned to tf32) truncate as the unpacker's fp32 -> tf32 does and the io pack from fp32 DST equals copy_tiles' bytes.
QWEN36_MTP_GDN_SO_FUSE_PARTS (default delta,upd,pack; decay is opt-in, slower on the card): comma subset of decay,delta,upd,pack (pack needs upd).
MUSTFAIL (QWEN36_MTP_GDN_SO_FUSE_MUSTFAIL=1, needs the delta part): v - kn@h skips the subtraction; the warmup self-test REFUSES the boot; the
explicit rung mustfail_so_fuse runs at every boot with the flag on.

QWEN36_MTP_GDN_SO_VSPLIT=1|2|4 (verify only; needs STATE_ONCE + SO_QKH + SO_SPLITREAD, the served verify body; port of
the rnabeelTT gdn_decode head's QWEN35_GDN_VSPLIT, d62759f/dadfc7d): each head instance (row b, v-head h) is split over
P cores by value-column tiles when the bucket leaves cores idle. P = the largest of (cap, 4, 2) with Vt % P == 0 and
Bw*Nv*P <= the grid's cores (P150: Bw=1 -> P=4 on 96 cores, Bw=2 -> P=2 on 96 cores; Bw >= 4 -> P=1: today's program,
nothing changes). 1 = auto (cap 4), 2/4 = cap. Why (served, [source validation record]): the bw1 verify is
47 us per program on 24 of 110 cores (one instance per core, the per-token chain on Kt x Vt = 16-tile state ops); the
decay, kn @ h, delta, the rank-1 update and qn @ new_h are all column-separable (every FPU op of an output tile reads
only that tile's column of the state), so part p computes exactly today's bytes for its columns: the compute runs
unchanged with Vt -> Vt/P; q/k gathers, their norms and beta/g are duplicated per part. Reader/writer: only the part's
v tiles, state tiles, o-stick slice, SO_T0 state pages and cin-stick byte range (disjoint; together today's stick).
Bitwise vs P=1 by construction, CHECKED (ttsim sim_mtp_gdn_state_once.py --vsplit; the warmup self-test compares the split verify
with the P=1 verify, o + cin + state, on every chip). MUSTFAIL (QWEN36_MTP_GDN_SO_VSPLIT_MUSTFAIL=1): part 1 reads part
0's state columns; the self-test REFUSES. Its explicit wrong-input rung (mustfail_so_vsplit) runs at every boot.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import os
import struct
from pathlib import Path

try:
    import ttnn
except ImportError:  # host unit tests: the pure helpers below need no ttnn
    ttnn = None

ENV = "QWEN36_MTP_GDN_STATE_ONCE"
ENV_MF = "QWEN36_MTP_GDN_STATE_ONCE_MUSTFAIL"
ENV_T0 = "QWEN36_MTP_GDN_SO_T0"
ENV_T0_MF = "QWEN36_MTP_GDN_SO_T0_MUSTFAIL"
T0_BANNER = "QWEN36_MTP_GDN_SO_T0 engaged: commit tokens idx (was idx+1)"
T0_MF_BANNER = "QWEN36_MTP_GDN_SO_T0_MUSTFAIL engaged"
ENV_ROW0 = "QWEN36_MTP_GDN_SO_SFPU_ROW0"
ENV_ROW0_MF = "QWEN36_MTP_GDN_SO_SFPU_ROW0_MUSTFAIL"
ROW0_BANNER = "QWEN36_MTP_GDN_SO_SFPU_ROW0 engaged: q/k inv_rms + decay exp on SFPU lanes of [0,0] only"
ROW0_MF_BANNER = "QWEN36_MTP_GDN_SO_SFPU_ROW0_MUSTFAIL engaged"
_ROW0_DEFINE = "#define SO_SFPU_ROW0 1\n"
_ROW0_MF_DEFINE = "#define SO_SFPU_ROW0_MUSTFAIL 1\n"
ENV_FUSE = "QWEN36_MTP_GDN_SO_FUSE"
ENV_FUSE_MF = "QWEN36_MTP_GDN_SO_FUSE_MUSTFAIL"
ENV_FUSE_PARTS = "QWEN36_MTP_GDN_SO_FUSE_PARTS"
# default parts: decay is opt-in (QWEN36_MTP_GDN_SO_FUSE_PARTS): bitwise, but slower on the card (verify +3 us, packed
# commit +10%; [source validation record]), the exp(g) full-tile broadcast costs more than the copy it saves
FUSE_DEFAULT = ("delta", "upd", "pack")
FUSE_PARTS = ("decay", "delta", "upd", "pack")
FUSE_BANNER = "QWEN36_MTP_GDN_SO_FUSE engaged: state chain kept in DST (dest reuse), new_h packed straight to its consumers"
FUSE_MF_BANNER = "QWEN36_MTP_GDN_SO_FUSE_MUSTFAIL engaged"
ENV_SPLIT = "QWEN36_MTP_GDN_SO_SPLITREAD"
ENV_SPLIT_MF = "QWEN36_MTP_GDN_SO_SPLITREAD_MUSTFAIL"
SPLIT_BANNER = "QWEN36_MTP_GDN_SO_SPLITREAD engaged: verify state + ones read on the writer RISC, gathers alone on the reader"
SPLIT_MF_BANNER = "QWEN36_MTP_GDN_SO_SPLITREAD_MUSTFAIL engaged"
ENV_FID = "QWEN36_MTP_GDN_STATE_ONCE_FID"  # auto (default: mirror the served decode kernel) | hifi4 | hifi2
ENV_IOSRC = "QWEN36_MTP_GDN_SO_IOSRC"
ENV_IOSRC_MF = "QWEN36_MTP_GDN_SO_IOSRC_MUSTFAIL"
IOSRC_BANNER = ("QWEN36_MTP_GDN_SO_IOSRC engaged: the io state and the io g are read by the SFPU directly "
                "(no fp32 mirror copies)")
IOSRC_MF_BANNER = "QWEN36_MTP_GDN_SO_IOSRC_MUSTFAIL engaged: the decay rotates the state's tiles"
_IOSRC_DEFINE = "#define SO_IOSRC 1\n"
_IOSRC_MF_DEFINE = "#define SO_IOSRC_MUSTFAIL 1\n"
ENV_QKH = "QWEN36_MTP_GDN_SO_QKH"
ENV_QKH_MF = "QWEN36_MTP_GDN_SO_QKH_MUSTFAIL"
QKH_BANNER = ("QWEN36_MTP_GDN_SO_QKH engaged: the q/k L2 norms run once per instance (all T token rows at once), "
              "not once per token")
QKH_MF_BANNER = "QWEN36_MTP_GDN_SO_QKH_MUSTFAIL engaged: the last row's beta lands on the next token's row"
QKH_FUSE_NEEDS = ("delta", "upd", "pack")
ENV_VSPLIT = "QWEN36_MTP_GDN_SO_VSPLIT"
ENV_VSPLIT_MF = "QWEN36_MTP_GDN_SO_VSPLIT_MUSTFAIL"
VSPLIT_BANNER = "QWEN36_MTP_GDN_SO_VSPLIT engaged"
VSPLIT_MF_BANNER = "QWEN36_MTP_GDN_SO_VSPLIT_MUSTFAIL engaged: part 1 reads part 0's state columns"
REQUIRES = ("QWEN36_MTP_STATE_SLOT", "QWEN36_MTP_CONV_MULTI")
SKIP = 0xFFFF  # the slot page's keep-row entry (state_slot.SKIP)
IDX_LEN = 32
T_MAX = 16

_KDIR = Path(__file__).resolve().parent / "kernels" / "state_once"
_SRC = {}
_STOCK_REL = "ttnn/cpp/ttnn/operations/transformer/decode_gated_delta_rule/device/kernels/compute/decode_gated_delta_rule.cpp"
_FID_DEFINE = "#define MATH_FIDELITY ckernel::MathFidelity::"

# CB plan: the production factory's indices and sizes (the kernels share them), cb_carry 30, the per-instance token
# count 31 (compute) and 32 (writer), the reader's stick 33. (index, tiles | "bytes:N" | "stick", buffers, kind)
_CB_PLAN = (
    (0, "Kt", 2, "io"),  # q            (x2: the next token's gather overlaps this token's math)
    (1, "Kt", 2, "io"),  # k
    (2, "Vt", 2, "io"),  # v
    (3, 1, 2, "io"),  # g
    (4, 1, 2, "io"),  # beta
    (5, "kv", 1, "io"),  # state        (base state, once per instance)
    (6, 1, 1, "f32"),  # ones
    (7, "Kt", 1, "f32"),  # qsq
    (8, "Kt", 1, "f32"),  # ksq
    (9, 1, 1, "f32"),  # sc
    (28, 1, 1, "f32"),  # sc2
    (29, 1, 1, "f32"),  # sc3
    (10, "Kt", 1, "f32"),  # qn
    (11, "Kt", 1, "f32"),  # kn
    (12, "Kt", 1, "f32"),  # kcol
    (13, 1, 1, "f32"),  # gexp
    (14, "kv", 1, "f32"),  # sdec
    (15, "Vt", 1, "f32"),  # vread
    (16, "Vt", 2, "f32"),  # delta (in-place *beta ping-pong)
    (17, "kv", 1, "f32"),  # outer
    (18, "kv", 1, "io"),  # sout (commit: the committed state for the writer)
    (19, "Vt", 2, "io"),  # out (verify: o)
    (20, "Kt", 1, "f32"),  # qf
    (21, "Kt", 1, "f32"),  # kf
    (22, "Vt", 1, "f32"),  # vf
    (23, 1, 1, "f32"),  # gf
    (24, 1, 1, "f32"),  # betaf
    (25, "kv", 1, "f32"),  # sf
    (26, "kv", 1, "f32"),  # snew
    (27, 1, 1, "io"),  # scratch (verify: writer o-stick staging; commit: the reader's idx page)
    (30, "kv", 1, "io"),  # carry (compute -> compute, token t -> t+1)
    (31, "bytes:16", 2, "u32"),  # cnt  (commit: reader -> compute, one uint32 per instance; SO_T0 verify: WT)
    (32, "bytes:16", 2, "u32"),  # wcnt (commit: reader -> writer; SO_T0 verify: WT)
    (33, "stick", 1, "io"),  # stick (reader-private: the cin stick staged for its DRAM write / after its read)
)


def cin_width(Dk, Dv):
    """Elements per cin stick: [k (Dk) | v (Dv) | beta, g, 0 x 30]; x2 B = a multiple of 64 B at Dk = Dv = 128."""
    return Dk + Dv + 32


# ------------------------------------------------------------------------------------------------ pure helpers
def enabled():
    return _tt_public_getenv(ENV, "0") == "1"


def mustfail():
    v = _tt_public_getenv(ENV_MF, "0")
    if v not in ("0", "1", ""):
        raise ValueError(f"{ENV_MF}={v!r}: 0 | 1")
    return v == "1"


def check_env(env=None):
    """Refuse (RuntimeError) the flag without its prerequisites; False when unset. The commit reads the slot page
    (QWEN36_MTP_STATE_SLOT) and the verify reads CONV_MULTI's one batch-M conv_gates output."""
    env = os.environ if env is None else env
    if env.get(ENV, "0") != "1":
        if env.get(ENV_MF, "0") == "1":
            raise RuntimeError(f"{ENV_MF}=1 set without {ENV}=1")
        return False
    missing = [k for k in REQUIRES if env.get(k, "0") != "1"]
    if missing:
        raise RuntimeError(
            f"{ENV}=1 refused: it needs {' and '.join(k + '=1' for k in missing)} (the commit reads the slot page "
            f"gslot and the verify reads CONV_MULTI's batch-M conv_gates output)"
        )
    return True


def token_counts(page, T, bw, mf=False, t0=False):
    """The commit's per-row token count, exactly as so_reader.cpp computes it: n = idx + 1 for idx < T, else 0 (keep
    the row); mf: row bw-1 gets n+1 (n-1 when n == T). t0 (SO_T0): then n -> max(n - 1, 0), fed from token 1 (token
    0 is already in rec_state). Rows >= bw are never visited (None)."""
    out = []
    for b in range(len(page)):
        if b >= bw:
            out.append(None)
            continue
        s = page[b]
        n = s + 1 if s < T else 0
        if mf and b == bw - 1:
            n = n + 1 if n < T else n - 1
        if t0:
            n = max(n - 1, 0)
        out.append(n)
    return out


def t0_enabled(env=None):
    """QWEN36_MTP_GDN_SO_T0: False when unset; refuses (RuntimeError) without QWEN36_MTP_GDN_STATE_ONCE=1 (and its
    prerequisites, via check_env), and the must-fail without the flag."""
    env = os.environ if env is None else env
    if env.get(ENV_T0, "0") != "1":
        if env.get(ENV_T0_MF, "0") == "1":
            raise RuntimeError(f"{ENV_T0_MF}=1 set without {ENV_T0}=1")
        return False
    if env.get(ENV, "0") != "1":
        raise RuntimeError(f"{ENV_T0}=1 refused: it needs {ENV}=1 (it changes the state-once verify and commit)")
    check_env(env)
    return True


def t0_mustfail(env=None):
    env = os.environ if env is None else env
    v = env.get(ENV_T0_MF, "0")
    if v not in ("0", "1", ""):
        raise ValueError(f"{ENV_T0_MF}={v!r}: 0 | 1")
    return v == "1"


def row0_enabled(env=None):
    """QWEN36_MTP_GDN_SO_SFPU_ROW0: False when unset; refuses (RuntimeError) without QWEN36_MTP_GDN_STATE_ONCE=1 (and its
    prerequisites, via check_env), a bad value, and the must-fail without the flag."""
    env = os.environ if env is None else env
    v = env.get(ENV_ROW0, "0")
    mf = env.get(ENV_ROW0_MF, "0")
    if v not in ("0", "1", "") or mf not in ("0", "1", ""):
        raise ValueError(f"{ENV_ROW0}={v!r} {ENV_ROW0_MF}={mf!r}: 0 | 1")
    if v != "1":
        if mf == "1":
            raise RuntimeError(f"{ENV_ROW0_MF}=1 set without {ENV_ROW0}=1")
        return False
    if env.get(ENV, "0") != "1":
        raise RuntimeError(f"{ENV_ROW0}=1 refused: it needs {ENV}=1 (it changes the state-once compute kernel)")
    check_env(env)
    return True


def row0_mustfail(env=None):
    env = os.environ if env is None else env
    return row0_enabled(env) and env.get(ENV_ROW0_MF, "0") == "1"


def split_enabled(env=None):
    """QWEN36_MTP_GDN_SO_SPLITREAD (verify only): False when unset; refuses without QWEN36_MTP_GDN_STATE_ONCE=1 (and its
    prerequisites), a bad value, and the must-fail without the flag."""
    env = os.environ if env is None else env
    v = env.get(ENV_SPLIT, "0")
    mf = env.get(ENV_SPLIT_MF, "0")
    if v not in ("0", "1", "") or mf not in ("0", "1", ""):
        raise ValueError(f"{ENV_SPLIT}={v!r} {ENV_SPLIT_MF}={mf!r}: 0 | 1")
    if v != "1":
        if mf == "1":
            raise RuntimeError(f"{ENV_SPLIT_MF}=1 set without {ENV_SPLIT}=1")
        return False
    if env.get(ENV, "0") != "1":
        raise RuntimeError(f"{ENV_SPLIT}=1 refused: it needs {ENV}=1 (it changes the state-once verify kernels)")
    check_env(env)
    return True


def split_mustfail(env=None):
    env = os.environ if env is None else env
    return split_enabled(env) and env.get(ENV_SPLIT_MF, "0") == "1"


def iosrc_enabled(env=None):
    """QWEN36_MTP_GDN_SO_IOSRC: False when unset; refuses without QWEN36_MTP_GDN_STATE_ONCE=1 (and its prerequisites), a
    bad value, and the must-fail without the flag."""
    env = os.environ if env is None else env
    v = env.get(ENV_IOSRC, "0")
    mf = env.get(ENV_IOSRC_MF, "0")
    if v not in ("0", "1", "") or mf not in ("0", "1", ""):
        raise ValueError(f"{ENV_IOSRC}={v!r} {ENV_IOSRC_MF}={mf!r}: 0 | 1")
    if v != "1":
        if mf == "1":
            raise RuntimeError(f"{ENV_IOSRC_MF}=1 set without {ENV_IOSRC}=1")
        return False
    if env.get(ENV, "0") != "1":
        raise RuntimeError(f"{ENV_IOSRC}=1 refused: it needs {ENV}=1 (it changes the state-once compute kernel)")
    check_env(env)
    return True


def iosrc_mustfail(env=None):
    env = os.environ if env is None else env
    return iosrc_enabled(env) and env.get(ENV_IOSRC_MF, "0") == "1"


def qkh_enabled(env=None):
    """QWEN36_MTP_GDN_SO_QKH: False when unset; refuses without QWEN36_MTP_GDN_STATE_ONCE=1, without the served body it
    is built on (QWEN36_MTP_GDN_SO_FUSE parts delta,upd,pack without decay, QWEN36_MTP_GDN_SO_IOSRC=1), a bad value,
    and the must-fail without the flag."""
    env = os.environ if env is None else env
    v = env.get(ENV_QKH, "0")
    mf = env.get(ENV_QKH_MF, "0")
    if v not in ("0", "1", "") or mf not in ("0", "1", ""):
        raise ValueError(f"{ENV_QKH}={v!r} {ENV_QKH_MF}={mf!r}: 0 | 1")
    if v != "1":
        if mf == "1":
            raise RuntimeError(f"{ENV_QKH_MF}=1 set without {ENV_QKH}=1")
        return False
    if env.get(ENV, "0") != "1":
        raise RuntimeError(f"{ENV_QKH}=1 refused: it needs {ENV}=1 (it changes the state-once verify kernels)")
    parts = fuse_parts(env)
    if not set(QKH_FUSE_NEEDS) <= set(parts) or "decay" in parts or not iosrc_enabled(env):
        raise RuntimeError(f"{ENV_QKH}=1 refused: it is built on the served verify body ({ENV_FUSE} parts "
                           f"{','.join(QKH_FUSE_NEEDS)} without decay, {ENV_IOSRC}=1); got parts {parts} "
                           f"{ENV_IOSRC}={env.get(ENV_IOSRC, '0')!r}")
    return True


def qkh_mustfail(env=None):
    env = os.environ if env is None else env
    return qkh_enabled(env) and env.get(ENV_QKH_MF, "0") == "1"


def qkh_prefix(Bw, Nv, qkh=False, qkh_mf=False, reader=False):
    """Text prepended to the verify's so_reader / so_writer / so_compute: '' when off (today's sources). The must-fail
    define goes to the reader only (it misplaces one beta row)."""
    if qkh_mf and not qkh:
        raise RuntimeError(f"{ENV_QKH_MF} without {ENV_QKH}")
    if not qkh:
        return ""
    out = f"#define SO_QKH 1\n#define SO_QKH_BW {int(Bw)}\n#define SO_QKH_H {int(Nv)}\n"
    return out + ("#define SO_QKH_MUSTFAIL 1\n" if (qkh_mf and reader) else "")


def vsplit_cap(env=None):
    """QWEN36_MTP_GDN_SO_VSPLIT -> the cap on P (0 when unset/0; 1 -> 4 (auto), 2, 4). Refuses a bad value, the flag
    without its served verify body (STATE_ONCE + SO_QKH + SO_SPLITREAD), and the must-fail without the flag."""
    env = os.environ if env is None else env
    v = env.get(ENV_VSPLIT, "0").strip()
    mf = env.get(ENV_VSPLIT_MF, "0")
    if v not in ("0", "1", "2", "4", "") or mf not in ("0", "1", ""):
        raise ValueError(f"{ENV_VSPLIT}={v!r} {ENV_VSPLIT_MF}={mf!r}: 0 | 1 (auto) | 2 | 4, must-fail 0 | 1")
    if v in ("0", ""):
        if mf == "1":
            raise RuntimeError(f"{ENV_VSPLIT_MF}=1 set without {ENV_VSPLIT}")
        return 0
    if env.get(ENV, "0") != "1" or not qkh_enabled(env) or not split_enabled(env):
        raise RuntimeError(f"{ENV_VSPLIT}={v} refused: it is built on the served verify body ({ENV}=1, {ENV_QKH}=1, "
                           f"{ENV_SPLIT}=1)")
    return 4 if v == "1" else int(v)


def vsplit_mustfail(env=None):
    env = os.environ if env is None else env
    return vsplit_cap(env) > 0 and env.get(ENV_VSPLIT_MF, "0") == "1"


def vsplit_parts(cap, Bw, Nv, Vt, ncores):
    """P for a verify at Bw: the largest of (4, 2) <= cap with Vt % P == 0 and Bw*Nv*P <= ncores, else 1 (today)."""
    for p in (4, 2):
        if p <= cap and Vt % p == 0 and Bw * Nv * p <= ncores:
            return p
    return 1


def vsplit_prefix(P, mf=False):
    """Text prepended to so_reader.cpp / so_writer.cpp (verify): '' at P=1 (today's sources)."""
    if P <= 1:
        if mf:
            raise RuntimeError(f"{ENV_VSPLIT_MF} without a split (P={P})")
        return ""
    return f"#define SO_VSPLIT {int(P)}\n" + ("#define SO_VSPLIT_MUSTFAIL 1\n" if mf else "")


def split_prefix(split=False, split_mf=False):
    """Text prepended to so_reader.cpp / so_writer.cpp (verify): '' when off (today's sources)."""
    if split_mf and not split:
        raise RuntimeError(f"{ENV_SPLIT_MF} without {ENV_SPLIT}")
    return ("#define SO_SPLITREAD 1\n" if split else "") + ("#define SO_SPLITREAD_MUSTFAIL 1\n" if split_mf else "")


def fuse_parts(env=None):
    """QWEN36_MTP_GDN_SO_FUSE -> the tuple of fused parts (() when unset). Refuses without QWEN36_MTP_GDN_STATE_ONCE=1
    (and its prerequisites), a bad value or part, pack without upd, and the must-fail without the flag."""
    env = os.environ if env is None else env
    v = env.get(ENV_FUSE, "0")
    mf = env.get(ENV_FUSE_MF, "0")
    if v not in ("0", "1", "") or mf not in ("0", "1", ""):
        raise ValueError(f"{ENV_FUSE}={v!r} {ENV_FUSE_MF}={mf!r}: 0 | 1")
    if v != "1":
        if mf == "1":
            raise RuntimeError(f"{ENV_FUSE_MF}=1 set without {ENV_FUSE}=1")
        if env.get(ENV_FUSE_PARTS):
            raise RuntimeError(f"{ENV_FUSE_PARTS} set without {ENV_FUSE}=1")
        return ()
    if env.get(ENV, "0") != "1":
        raise RuntimeError(f"{ENV_FUSE}=1 refused: it needs {ENV}=1 (it changes the state-once compute kernel)")
    check_env(env)
    raw = env.get(ENV_FUSE_PARTS, "")
    parts = tuple(p.strip() for p in raw.split(",") if p.strip()) if raw else FUSE_DEFAULT
    bad = [p for p in parts if p not in FUSE_PARTS]
    if bad or not parts:
        raise ValueError(f"{ENV_FUSE_PARTS}={raw!r}: comma subset of {','.join(FUSE_PARTS)}")
    if "pack" in parts and "upd" not in parts:
        raise ValueError(f"{ENV_FUSE_PARTS}={raw!r}: pack needs upd")
    return tuple(p for p in FUSE_PARTS if p in parts)


def fuse_mustfail(env=None):
    env = os.environ if env is None else env
    return bool(fuse_parts(env)) and env.get(ENV_FUSE_MF, "0") == "1"


def _norm_fuse(fuse):
    if fuse is True:
        return FUSE_PARTS
    if not fuse:
        return ()
    fuse = tuple(fuse)
    if any(p not in FUSE_PARTS for p in fuse) or ("pack" in fuse and "upd" not in fuse):
        raise ValueError(f"fuse parts {fuse!r}")
    return tuple(p for p in FUSE_PARTS if p in fuse)


def compute_prefix(fid_define, row0=False, row0_mf=False, fuse=(), fuse_mf=False, iosrc=False, iosrc_mf=False):
    """The text prepended to so_compute.cpp: the fidelity define, then (SFPU_ROW0) the row-0 define and its must-fail,
    then (SO_FUSE) one define per fused part and its must-fail, then (SO_IOSRC) its define and its must-fail. row0, fuse
    and iosrc off: exactly fid_define, i.e. today's source."""
    if row0_mf and not row0:
        raise RuntimeError(f"{ENV_ROW0_MF} without {ENV_ROW0}")
    fuse = _norm_fuse(fuse)
    if fuse_mf and not fuse:
        raise RuntimeError(f"{ENV_FUSE_MF} without {ENV_FUSE}")
    if fuse_mf and "delta" not in fuse:
        raise RuntimeError(f"{ENV_FUSE_MF}: the must-fail lives in the delta part")
    if iosrc_mf and not iosrc:
        raise RuntimeError(f"{ENV_IOSRC_MF} without {ENV_IOSRC}")
    out = fid_define + (_ROW0_DEFINE if row0 else "") + (_ROW0_MF_DEFINE if row0_mf else "")
    out += "".join(f"#define SO_FUSE_{p.upper()} 1\n" for p in fuse)
    out += (_IOSRC_DEFINE if iosrc else "") + (_IOSRC_MF_DEFINE if iosrc_mf else "")
    return out + ("#define SO_FUSE_MUSTFAIL 1\n" if fuse_mf else "")


def t0_write_depth(T, bw, mf=False):
    """SO_T0 verify, as so_reader.cpp: per in-bucket row the token after which the state goes over rec_state row b.
    0 for every row; mf: 1 for row bw-1 (the wrong depth)."""
    return [1 if (mf and b == bw - 1) else 0 for b in range(bw)]


def t0_reference_page(page, T, bw):
    """The sequential slot-commit page equivalent to an SO_T0 commit of `page`: in-bucket keep entries (SKIP, >= T)
    become 0 (the verify already applied token 0; the commit then leaves the row), everything else unchanged."""
    return [(0 if (b < bw and s >= T) else s) for b, s in enumerate(page)]


def state_units(T, bw=None, bmax=None):
    """State traffic per GDN layer per verify+commit, in units of one Bw-row recurrent state.
    -> (before, after, detail). before: the sequential verify (T reads + T writes, + the slice copy's read + write when
    Bw < Bmax) + the slot commit (1 read + 1 write). after: 1 read (verify) + 1 read + 1 write (commit)."""
    sl = 2 if (bw is not None and bmax is not None and bw < bmax) else 0
    before = 2 * T + sl + 2
    after = 3
    detail = (f"before: verify {T}R+{T}W{' + slice 1R+1W' if sl else ''}, commit 1R+1W = {before}; "
              f"after: verify 1R, commit 1R+1W = {after}")
    return before, after, detail


def op_counts(T):
    """GDN conv + recurrence programs per layer per verify, (STATE_ONCE, CONV_MULTI gather split):
    CONV_MULTI: build_windows + conv_gates + 2 split + T recurrences; STATE_ONCE: build_windows + conv_gates + 1.
    The commit is 2 programs either way (commit_once replaces commit_rec; commit_conv_multi stays)."""
    return 3, (4 + T if T > 1 else 2 + T)


# ------------------------------------------------------------------------------------------------ device side
def _src(name):
    if name not in _SRC:
        _SRC[name] = (_KDIR / name).read_text()
    return _SRC[name]


def _tile_bytes(dtype):
    if dtype == ttnn.float32:
        return 4096
    if dtype == ttnn.bfloat16:
        return 2048
    raise RuntimeError(f"{ENV}: unsupported dtype {dtype} (bf16 or fp32, as the production decode op)")


_FUSE_DROP = {"decay": 25, "delta": 15, "upd": 17}  # sf, vread, outer: unused when the part keeps them in DST


def cb_plan(dtype, Dk=128, Dv=128, fuse=(), stick_dv=None):
    """[(index, bytes, page, kind)] and the per-core total. fuse (SO_FUSE parts): the CBs a part no longer uses are
    dropped and (decay) the full exp(g) tile 34 added; () is exactly today's plan."""
    fuse = _norm_fuse(fuse)
    drop = {_FUSE_DROP[p] for p in fuse if p in _FUSE_DROP}
    extra = ((34, 1, 1, "f32"),) if "decay" in fuse else ()
    Kt, Vt = Dk // 32, Dv // 32
    n = {"Kt": Kt, "Vt": Vt, "kv": Kt * Vt}
    io = _tile_bytes(dtype)
    out = []
    for idx, tiles, nbuf, kind in tuple(e for e in _CB_PLAN if e[0] not in drop) + extra:
        if tiles == "stick":
            page = cin_width(Dk, Dv if stick_dv is None else stick_dv) * (io // 1024)
            out.append((idx, page * nbuf, page, kind))
            continue
        if isinstance(tiles, str) and tiles.startswith("bytes:"):
            page = int(tiles[6:])
            out.append((idx, page * nbuf, page, kind))
            continue
        t = n[tiles] if isinstance(tiles, str) else tiles
        page = io if kind == "io" else 4096
        out.append((idx, t * nbuf * page, page, kind))
    return out, sum(b for _, b, _, _ in out)


def _stock_kernel_file():
    for root in (_tt_public_getenv("TT_METAL_HOME"), "/opt/tt-metal-src", "/opt/tt-metal"):
        if root:
            p = Path(root) / _STOCK_REL
            if p.is_file():
                return p
    return None


def fidelity():
    """-> (descriptor MathFidelity, prepended define or '', provenance). auto: mirror the served decode kernel file
    (the -gf arm overlays it with a HiFi2 MATH_FIDELITY define over the factory's HiFi4 descriptor)."""
    want = _tt_public_getenv(ENV_FID, "auto").lower()
    if want == "auto":
        p = _stock_kernel_file()
        if p is None:
            return ttnn.MathFidelity.HiFi4, "", "factory HiFi4 (no served kernel file found)"
        for line in p.read_text().splitlines():
            s = line.strip()
            if s.startswith(_FID_DEFINE):
                return ttnn.MathFidelity.HiFi4, s + "\n", f"{s.split('::')[-1]} define in {p}"
        return ttnn.MathFidelity.HiFi4, "", f"factory HiFi4 ({p} has no fidelity define)"
    names = {"hifi4": "HiFi4", "hifi3": "HiFi3", "hifi2": "HiFi2", "lofi": "LoFi"}
    if want not in names:
        raise ValueError(f"{ENV_FID}={want!r}: auto|hifi4|hifi3|hifi2|lofi")
    return ttnn.MathFidelity.HiFi4, f"{_FID_DEFINE}{names[want]}\n", f"{names[want]} ({ENV_FID})"


def _grid(device, BH):
    """The production factory's mapping: per_core = ceil(BH / ncores), core c = (c / gy, c % gy), column-major."""
    g = device.compute_with_storage_grid_size()
    gx, gy = g.x, g.y
    ncores = gx * gy
    per = (BH + ncores - 1) // ncores
    active = (BH + per - 1) // per
    work = []
    for c in range(active):
        start = c * per
        work.append((ttnn.CoreCoord(c // gy, c % gy), start, min(per, BH - start)))
    full, rem = divmod(active, gy)
    ranges = []
    if full:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full - 1, gy - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(full, 0), ttnn.CoreCoord(full, rem - 1)))
    return ttnn.CoreRangeSet(ranges), work, per


def _shape(t):
    return [int(t.shape[i]) for i in range(len(t.shape))]


def _check_tile(t, what, dtype):
    if t.layout != ttnn.TILE_LAYOUT:
        raise RuntimeError(f"{ENV}: {what} must be TILE, got {t.layout}")
    if t.dtype != dtype:
        raise RuntimeError(f"{ENV}: {what} dtype {t.dtype} != state dtype {dtype} (the production op's one-dtype rule)")
    if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise RuntimeError(f"{ENV}: {what} must be interleaved, got {t.memory_config()}")


def _check_inputs(conv_all, beta_all, g_all, state, Nk, Nv, Dk, Dv, T, Bw):
    dt = state.dtype
    M = T * Bw
    C = _shape(conv_all)[-1]
    kd = Nk * Dk
    max_m = int(_tt_public_getenv("QWEN36_MTP_VERIFY_MAX_ROWS", "32"))
    if not (1 <= T <= T_MAX) or M > max_m:
        raise RuntimeError(f"{ENV}: T={T} Bw={Bw}: need 1 <= T <= {T_MAX} and T*Bw <= {max_m}")
    if C != 2 * kd + Nv * Dv or Dk % 32 or Dv % 32 or kd % 32 or Nv % Nk:
        raise RuntimeError(f"{ENV}: C={C} Nk={Nk} Nv={Nv} Dk={Dk} Dv={Dv}: need C == 2*Nk*Dk + Nv*Dv, 32-aligned")
    _check_tile(state, "state", dt)
    ss = _shape(state)
    if ss[0] < Bw or ss[1:] != [Nv, Dk, Dv]:
        raise RuntimeError(f"{ENV}: state shape {ss} vs Bw={Bw} Nv={Nv} Dk={Dk} Dv={Dv}")
    _check_tile(conv_all, "conv", dt)
    _check_tile(beta_all, "beta", dt)
    _check_tile(g_all, "g", dt)
    if _shape(conv_all) != [1, M, C] or _shape(beta_all) != [1, M, Nv] or _shape(g_all) != [1, M, Nv]:
        raise RuntimeError(
            f"{ENV}: conv {_shape(conv_all)} beta {_shape(beta_all)} g {_shape(g_all)} vs M={M} (T={T} x Bw={Bw})"
        )
    return dt, C, kd


def _program(mode, conv_all, beta_all, g_all, state, cin, dst, idx, Nk, Nv, Dk, Dv, scale, T, Bw, mf, fid,
             t0=False, t0_mf=False, row0=None, row0_mf=None, fuse=None, fuse_mf=None, split=None, split_mf=None,
             iosrc=None, iosrc_mf=None, qkh=None, qkh_mf=None, vsplit=None, vsplit_mf=None):
    """mode 0: verify (conv_all/beta_all/g_all -> o = dst, cin written; t0: the state after token 0 -> state rows
    < Bw); mode 1: commit (cin, idx -> state; t0: tokens 1..idx). row0 / row0_mf (QWEN36_MTP_GDN_SO_SFPU_ROW0, default:
    the env): the compute's SFPU passes on the lanes of [0,0] only (bitwise today's) / its must-fail."""
    row0 = row0_enabled() if row0 is None else row0
    row0_mf = (row0_mustfail() if row0_mf is None else row0_mf) and row0
    fuse = _norm_fuse(fuse_parts() if fuse is None else fuse)
    fuse_mf = bool((fuse_mustfail() if fuse_mf is None else fuse_mf) and fuse)
    split = bool((split_enabled() if split is None else split) and mode == 0)
    split_mf = bool((split_mustfail() if split_mf is None else split_mf) and split)
    iosrc = bool(iosrc_enabled() if iosrc is None else iosrc)
    iosrc_mf = bool((iosrc_mustfail() if iosrc_mf is None else iosrc_mf) and iosrc)
    qkh = bool((qkh_enabled() if qkh is None else qkh) and mode == 0 and T >= 2)  # T=1: nothing to hoist
    qkh_mf = bool((qkh_mustfail() if qkh_mf is None else qkh_mf) and qkh)
    vcap = (vsplit_cap() if vsplit is None else int(vsplit)) if mode == 0 else 0
    vsplit_mf = bool((vsplit_mustfail() if vsplit_mf is None else vsplit_mf) and vcap)
    dt = state.dtype
    if qkh:
        if T * Bw > 32:
            raise RuntimeError(f"{ENV_QKH}: T={T} Bw={Bw}: needs T*Bw <= 32 (one row page)")
        if set(QKH_FUSE_NEEDS) - set(_norm_fuse(fuse)) or "decay" in _norm_fuse(fuse) or not iosrc:
            raise RuntimeError(f"{ENV_QKH}: needs fuse {QKH_FUSE_NEEDS} without decay and iosrc; got "
                               f"fuse={_norm_fuse(fuse)} iosrc={iosrc}")
        if conv_all.memory_config().buffer_type != ttnn.BufferType.L1:
            raise RuntimeError(f"{ENV_QKH}: conv_all must be L1 interleaved (the reader reads 32 B row chunks)")
    if t0 and T < 2:
        raise RuntimeError(f"{ENV_T0}: T={T}: needs T >= 2 (an MTP verify)")
    if t0 and mode == 0 and state.memory_config().buffer_type != ttnn.BufferType.DRAM:
        raise RuntimeError(f"{ENV_T0}: the verify writes rec_state: it must be DRAM interleaved")
    C = 2 * Nk * Dk + Nv * Dv
    kd = Nk * Dk
    if mode == 0:
        _check_inputs(conv_all, beta_all, g_all, state, Nk, Nv, Dk, Dv, T, Bw)
    else:
        _check_tile(state, "state", dt)
        max_m = int(_tt_public_getenv("QWEN36_MTP_VERIFY_MAX_ROWS", "32"))
        if _shape(state)[0] < Bw or _shape(state)[1:] != [Nv, Dk, Dv] or T * Bw > max_m or not 1 <= T <= T_MAX:
            raise RuntimeError(f"{ENV}: state {_shape(state)} vs T={T} Bw={Bw} Nv={Nv} Dk={Dk} Dv={Dv} (max rows {max_m})")
    W = cin_width(Dk, Dv)
    if cin.layout != ttnn.ROW_MAJOR_LAYOUT or cin.dtype != dt or _shape(cin) != [T * Bw, 1, Nv, W]:
        raise RuntimeError(f"{ENV}: cin {cin.layout} {cin.dtype} {_shape(cin)} != ROW_MAJOR {dt} {[T * Bw, 1, Nv, W]}")
    if cin.memory_config().buffer_type != ttnn.BufferType.DRAM:
        raise RuntimeError(f"{ENV}: cin must be DRAM (it lives from the verify trace to the commit trace)")
    desc_fid, fid_define, fid_why = fidelity() if fid is None else fid
    io = _tile_bytes(dt)
    cin_page = W * (io // 1024)
    Kt, Vt = Dk // 32, Dv // 32
    BH = Bw * Nv
    dev = state.device()
    _g = dev.compute_with_storage_grid_size()
    P = vsplit_parts(vcap, Bw, Nv, Vt, _g.x * _g.y) if vcap else 1
    if P > 1 and not (qkh and split):
        raise RuntimeError(f"{ENV_VSPLIT}: P={P} at Bw={Bw} needs the served verify body (SO_QKH and SO_SPLITREAD on); "
                           f"got qkh={qkh} split={split}")
    if vsplit_mf and P == 1:
        vsplit_mf = False  # nothing split at this Bw: the must-fail has no part 1
    crs, work, per = _grid(dev, BH * P)
    if P > 1 and per != 1:
        raise RuntimeError(f"{ENV_VSPLIT}: {BH * P} work items do not fit one per core")
    Vc = Vt // P  # value-column tiles per core (the compute's Vt)
    plan, total = cb_plan(dt, Dk, Vc * 32, fuse, stick_dv=Dv)
    fmt = {"io": dt, "f32": ttnn.float32, "u32": ttnn.uint32}
    cbs = [
        ttnn.CBDescriptor(
            total_size=nbytes,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt[kind], page_size=page)],
        )
        for i, nbytes, page, kind in plan
    ]
    eps_bits = struct.unpack("<I", struct.pack("<f", 1e-6))[0]
    scale_bits = struct.unpack("<I", struct.pack("<f", float(scale if scale is not None else Dk**-0.5)))[0]
    if mode == 1:
        if idx.layout != ttnn.ROW_MAJOR_LAYOUT or idx.dtype != ttnn.uint32 or _shape(idx)[-1] != IDX_LEN:
            raise RuntimeError(f"{ENV}: idx must be ROW_MAJOR uint32 [1, {IDX_LEN}], got {idx.layout} {idx.dtype} {_shape(idx)}")
        try:
            ipage = int(idx.buffer_aligned_page_size())
        except Exception:
            ipage = IDX_LEN * 4
        if ipage > io:
            raise RuntimeError(f"{ENV}: idx page {ipage} B does not fit the scratch tile ({io} B)")
        accs = (state, state, state, state, idx, cin)
        addrs = [0, 0, 0, state.buffer_address(), idx.buffer_address(), cin.buffer_address()]
    else:
        ipage = 0
        accs = (conv_all, beta_all, g_all, state, state, cin)
        addrs = [conv_all.buffer_address(), beta_all.buffer_address(), g_all.buffer_address(), state.buffer_address(),
                 0, cin.buffer_address()]
    r_ct = [Kt, Vt, Nv, C // 32, 0, kd // 32, 2 * kd // 32, Nv // Nk, (Nv + 31) // 32, T, Bw, mode, 1 if mf else 0,
            ipage, cin_page, 1 if t0 else 0, 1 if (t0 and t0_mf and mode == 0) else 0]
    for t in accs:
        r_ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    w_ct = ([Kt, Vt, T, mode, 1 if (t0 and mode == 0) else 0] + list(ttnn.TensorAccessorArgs(dst).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(state).get_compile_time_args()))
    c_ct = [Kt, Vc, eps_bits, scale_bits, T, mode, 1 if t0 else 0]
    r_rt, w_rt, c_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for core, start, n in work:
        # VSPLIT (P > 1): work item s = part * BH + instance, one per core; the part goes last in the reader/writer args
        bh0, part = (start % BH, start // BH) if P > 1 else (start, None)
        tail = [part] if P > 1 else []
        r_rt[core.x][core.y] = [bh0, n] + addrs + tail
        if mode == 0:
            w_rt[core.x][core.y] = [bh0, n, dst.buffer_address(), Dv * (io // 1024), BH, state.buffer_address()] + tail
        else:
            w_rt[core.x][core.y] = [bh0, n, dst.buffer_address()]
        c_rt[core.x][core.y] = [n]

    def kd_(name, ct, rt, cfg, prefix=""):
        return ttnn.KernelDescriptor(
            kernel_source=prefix + _src(name),
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=crs,
            compile_time_args=ct,
            runtime_args=rt,
            config=cfg,
        )

    ccfg = ttnn.ComputeConfigDescriptor()
    ccfg.math_fidelity = desc_fid
    ccfg.fp32_dest_acc_en = True
    ccfg.math_approx_mode = False
    desc = ttnn.ProgramDescriptor(
        kernels=[
            kd_("so_reader.cpp", r_ct, r_rt, ttnn.ReaderConfigDescriptor(),
                prefix=split_prefix(split, split_mf) + qkh_prefix(Bw, Nv, qkh, qkh_mf, reader=True)
                + vsplit_prefix(P)),
            kd_("so_writer.cpp", w_ct, w_rt, ttnn.WriterConfigDescriptor(),
                prefix=split_prefix(split, split_mf) + qkh_prefix(Bw, Nv, qkh, qkh_mf) + vsplit_prefix(P, vsplit_mf)),
            kd_("so_compute.cpp", c_ct, c_rt, ccfg,
                prefix=compute_prefix(fid_define, row0, row0_mf, fuse, fuse_mf, iosrc, iosrc_mf)
                + qkh_prefix(Bw, Nv, qkh, qkh_mf)),
        ],
        semaphores=[],
        cbs=cbs,
    )
    if mode == 0:
        ios = [conv_all, beta_all, g_all, state, cin, dst]  # t0: state is also written (already listed)
    else:
        ios = [cin, idx, state]
    ttnn.generic_op(ios, desc)
    return {"cores": len(work), "per_core": per, "cb_kib": total / 1024, "fid": fid_why, "t0": bool(t0),
            "t0_mf": bool(t0 and t0_mf and mode == 0), "row0": bool(row0), "row0_mf": bool(row0_mf), "fuse": ",".join(fuse), "fuse_mf": fuse_mf,
            "split": split, "split_mf": split_mf, "iosrc": iosrc, "iosrc_mf": iosrc_mf, "qkh": qkh, "qkh_mf": qkh_mf,
            "vsplit": P, "vsplit_mf": bool(vsplit_mf)}


def verify(conv_all, beta_all, g_all, state, Nk, Nv, Dk, Dv, scale, T, Bw, fid=None, t0=None, t0_mf=None, row0=None,
           row0_mf=None, fuse=None, fuse_mf=None, split=None, split_mf=None, iosrc=None, iosrc_mf=None, qkh=None,
           qkh_mf=None, vsplit=None, vsplit_mf=None):
    """The verify recurrence from CONV_MULTI's batch-M conv_gates output (any interleaved placement; L1 as served).
    -> (o ROW_MAJOR [T*Bw, 1, Nv, Dv] DRAM (row u = t*Bw + b), cin ROW_MAJOR [T*Bw, 1, Nv, cin_width] DRAM (the
    commit's inputs, one stick per (u, h)), info). The state (rows < Bw of the persistent [B*, Nv, Dk, Dv] buffer) is
    read once per row and, unless t0 (QWEN36_MTP_GDN_SO_T0, default: the env), not written; t0: rows < Bw get the
    state after token 0 (t0_mf: row Bw-1 after token 1). The commit must be called with the same t0."""
    t0 = t0_enabled() if t0 is None else t0
    t0_mf = (t0_mustfail() if t0_mf is None else t0_mf) and t0
    dev = state.device()
    o = ttnn.empty([T * Bw, 1, Nv, Dv], state.dtype, ttnn.ROW_MAJOR_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    cin = ttnn.empty([T * Bw, 1, Nv, cin_width(Dk, Dv)], state.dtype, ttnn.ROW_MAJOR_LAYOUT, dev,
                     ttnn.DRAM_MEMORY_CONFIG)
    info = _program(0, conv_all, beta_all, g_all, state, cin, o, None, Nk, Nv, Dk, Dv, scale, T, Bw, False, fid,
                    t0=t0, t0_mf=t0_mf, row0=row0, row0_mf=row0_mf, fuse=fuse, fuse_mf=fuse_mf,
                    split=split, split_mf=split_mf, iosrc=iosrc, iosrc_mf=iosrc_mf, qkh=qkh, qkh_mf=qkh_mf,
                    vsplit=vsplit, vsplit_mf=vsplit_mf)
    return o, cin, info


def commit(state, cin, idx, Nk, Nv, Dk, Dv, scale, T, Bw, mf=None, fid=None, t0=None, row0=None, row0_mf=None,
           fuse=None, fuse_mf=None, iosrc=None, iosrc_mf=None):
    """In place: state[b] <- the state after idx[b] + 1 of row b's verify tokens (from cin), recomputed from state[b]
    itself, for b < Bw with idx[b] < T; other rows untouched. t0 (QWEN36_MTP_GDN_SO_T0, default: the env; must match
    the verify's): state[b] already holds the state after token 0, tokens 1..idx[b] are recomputed. The state must be
    DRAM interleaved (the slot commit's rule)."""
    mc = state.memory_config()
    if mc.buffer_type != ttnn.BufferType.DRAM:
        raise RuntimeError(f"{ENV}: the committed state must be DRAM interleaved, got {mc}")
    mf = mustfail() if mf is None else mf
    t0 = t0_enabled() if t0 is None else t0
    return _program(1, None, None, None, state, cin, state, idx, Nk, Nv, Dk, Dv, scale, T, Bw, mf, fid, t0=t0, row0=row0,
                    row0_mf=row0_mf, fuse=fuse, fuse_mf=fuse_mf, iosrc=iosrc, iosrc_mf=iosrc_mf)


# ------------------------------------------------------------------------------------------------ self-test
def _bits(t):
    import torch

    t = t.contiguous()
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32)


def _same(a, b):
    import torch

    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def selftest_pages(T, bw, bmax, seed):
    """The idx pages the self-test commits: random depths with 0 and T-1 present, a SKIP row and a non-SKIP
    out-of-range entry (T) when the bucket has room, rows >= bw SKIP (as slot_index writes them) except one
    out-of-bucket entry set to 0 (it must still be ignored: rows >= bw are never visited)."""
    import random

    rng = random.Random(seed)
    pages = []
    for rnd in range(3):
        pick = [rng.randrange(T) for _ in range(bw)]
        pick[0] = 0
        pick[-1] = T - 1 if rnd != 2 else rng.randrange(T)
        if rnd >= 1 and bw > 2:
            pick[1] = SKIP
        if rnd == 2 and bw > 3:
            pick[2] = T
        page = [SKIP] * IDX_LEN
        page[:bw] = pick
        if rnd == 2 and bw < IDX_LEN:
            page[bw] = 0
        pages.append(page)
    return pages


def selftest(state_like, Nk, Nv, Dk, Dv, scale, T, bw, seq_op, seed=0, mf_served=None, fid=None, t0=None,
             t0_mf_served=None, row0=None, row0_mf_served=None, fuse=None, fuse_mf_served=None, split=None,
             split_mf_served=None, iosrc=None, iosrc_mf_served=None, qkh=None, qkh_mf_served=None, vsplit=None,
             vsplit_mf_served=None):
    """Fail-closed comparison of STATE_ONCE against the sequential verify + slot commit, on the device this runs on,
    at the shapes and dtype of `state_like` (the layer's persistent rec_state; read for shape only, every tensor here
    is a fresh copy). Random conv_all/beta_all/g_all in the served value ranges and a random base state of Bmax rows.
    Reference: T launches of seq_op (decode_gated_delta_rule_packed(inplace_state=False), return_row_major) on the
    candidate row blocks, state chained, then state_slot.commit_rec. Candidate: verify() + commit(). Bitwise: o of
    every token; the committed state (every row: rows >= bw and SKIP / out-of-range rows must be untouched), for each
    idx page of selftest_pages(). Then the explicit must-fail commit (last row +1 token) must differ.
    mf_served: the commit's must-fail setting for the served comparisons (default: the env), so under
    QWEN36_MTP_GDN_STATE_ONCE_MUSTFAIL=1 the served checks differ and selftest_ok() refuses.
    seq_op(conv, beta, g, state) -> (o, h): the production op (ttsim probes pass a port).
    t0 (QWEN36_MTP_GDN_SO_T0, default: the env): the verify must leave rows < bw at hist_0 (rows >= bw untouched:
    verify_state_t0), each commit runs on a device copy of that post-verify state and is compared with commit_rec of
    t0_reference_page(page) (in-bucket keep entries leave the row at hist_0); then an explicit SO_T0 must-fail verify
    (last row written after token 1) + the same commit must differ (mustfail_t0_verify). t0_mf_served: the verify's
    must-fail for the served comparisons (default: the env), so QWEN36_MTP_GDN_SO_T0_MUSTFAIL=1 REFUSES.
    row0 (QWEN36_MTP_GDN_SO_SFPU_ROW0, default: the env): every candidate verify/commit runs the row-0 SFPU compute (the
    sequential reference is the stock op, full-tile SFPU), row0_mf_served its must-fail (default: the env, so
    QWEN36_MTP_GDN_SO_SFPU_ROW0_MUSTFAIL=1 REFUSES); then the explicit row-0 must-fail verify (decay without exp) on a
    fresh base must differ (mustfail_sfpu_row0), at every boot.
    fuse (QWEN36_MTP_GDN_SO_FUSE parts, default: the env): every candidate verify/commit runs the fused compute,
    fuse_mf_served its must-fail (default: the env, so QWEN36_MTP_GDN_SO_FUSE_MUSTFAIL=1 REFUSES); then the explicit
    fuse must-fail verify (subtraction skipped) must differ (mustfail_so_fuse), at every boot.
    qkh (QWEN36_MTP_GDN_SO_QKH, default: the env): every candidate verify runs the once-per-instance q/k norms against
    the per-token stock sequential op (o of every token and the cin the commits consume), qkh_mf_served its must-fail
    (default: the env, so QWEN36_MTP_GDN_SO_QKH_MUSTFAIL=1 REFUSES); then the explicit QKH must-fail verify (the last
    row's beta on the next token's row) must differ (mustfail_so_qkh), at every boot.
    vsplit (QWEN36_MTP_GDN_SO_VSPLIT cap, default: the env): every candidate verify runs split where P > 1 at this bw
    (the served program); then, when P > 1, the split verify is compared with the P=1 verify on a fresh copy of the
    base (o, cin and the SO_T0 state, every chip: verify_vsplit_vs_p1) and the explicit must-fail (part 1 reads part 0's
    state columns) must differ from the sequential o (mustfail_so_vsplit), at every boot. vsplit_mf_served: the split's
    must-fail for the served verify (default: the env, so QWEN36_MTP_GDN_SO_VSPLIT_MUSTFAIL=1 REFUSES).
    Returns {check: "exact" | "differs" | ...}."""
    import torch

    try:
        from models.demos.blackhole.qwen36.tt.gdn import state_slot as _ss
    except ImportError:
        import importlib.util as _ilu

        _spec = _ilu.spec_from_file_location("state_slot", Path(__file__).resolve().parent / "state_slot.py")
        _ss = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_ss)

    mf_served = mustfail() if mf_served is None else mf_served
    t0 = t0_enabled() if t0 is None else t0
    t0_mf_served = (t0_mustfail() if t0_mf_served is None else t0_mf_served) and t0
    row0 = row0_enabled() if row0 is None else row0
    row0_mf_served = (row0_mustfail() if row0_mf_served is None else row0_mf_served) and row0
    fuse = _norm_fuse(fuse_parts() if fuse is None else fuse)
    fuse_mf_served = bool((fuse_mustfail() if fuse_mf_served is None else fuse_mf_served) and fuse)
    split = bool(split_enabled() if split is None else split)
    split_mf_served = bool((split_mustfail() if split_mf_served is None else split_mf_served) and split)
    iosrc = bool(iosrc_enabled() if iosrc is None else iosrc)
    iosrc_mf_served = bool((iosrc_mustfail() if iosrc_mf_served is None else iosrc_mf_served) and iosrc)
    fz = dict(fuse=fuse, fuse_mf=fuse_mf_served)
    sz = dict(split=split, split_mf=split_mf_served)
    qkh = bool(qkh_enabled() if qkh is None else qkh)
    qkh_mf_served = bool((qkh_mustfail() if qkh_mf_served is None else qkh_mf_served) and qkh)
    iz = dict(iosrc=iosrc, iosrc_mf=iosrc_mf_served)
    vcap = vsplit_cap() if vsplit is None else int(vsplit)
    vsplit_mf_served = bool((vsplit_mustfail() if vsplit_mf_served is None else vsplit_mf_served) and vcap)
    qz = dict(qkh=qkh, qkh_mf=qkh_mf_served, vsplit=vcap, vsplit_mf=False)  # every rung below: the served split
    dev = state_like.device()
    _gs = dev.compute_with_storage_grid_size()
    vP = vsplit_parts(vcap, bw, Nv, Dv // 32, _gs.x * _gs.y) if (vcap and qkh and split and T >= 2) else 1
    mesh = hasattr(ttnn, "MeshDevice") and isinstance(dev, ttnn.MeshDevice)
    rep = ttnn.ReplicateTensorToMesh(dev) if mesh else None
    dt = state_like.dtype
    tdt = torch.bfloat16 if dt == ttnn.bfloat16 else torch.float32
    Bmax = _shape(state_like)[0]
    C = 2 * Nk * Dk + Nv * Dv
    M = T * bw
    DR, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    g = torch.Generator().manual_seed(seed)

    def up(t, layout=ttnn.TILE_LAYOUT, dtype=None, mc=DR):
        return ttnn.from_torch(t, dtype=dtype or dt, layout=layout, device=dev, mesh_mapper=rep, memory_config=mc)

    def chips(t):
        return [ttnn.to_torch(x) for x in (ttnn.get_device_tensors(t) if mesh else [t])]

    conv_t = torch.randn(1, M, C, generator=g).to(tdt)
    beta_t = torch.sigmoid(torch.randn(1, M, Nv, generator=g)).to(tdt)
    g_t = (-torch.nn.functional.softplus(torch.randn(1, M, Nv, generator=g)) * 0.5).to(tdt)
    s_t = (torch.randn(Bmax, Nv, Dk, Dv, generator=g) * 0.05).to(tdt)
    res = {}
    keep = []
    try:
        conv_all, beta_all, g_all = up(conv_t, mc=L1), up(beta_t, mc=L1), up(g_t, mc=L1)
        keep += [conv_all, beta_all, g_all]
        base = up(s_t)
        keep.append(base)
        # reference verify: T launches on the candidate row blocks, state chained (forward_verify's loop)
        cur = base if bw == Bmax else ttnn.slice(base, (0, 0, 0, 0), (bw, Nv, Dk, Dv))
        if cur is not base:
            keep.append(cur)
        ref_o, hists = [], []
        for j in range(T):
            lo, hi = j * bw, (j + 1) * bw
            blk = [ttnn.slice(t, (0, lo, 0), (1, hi, w)) for t, w in ((conv_all, C), (beta_all, Nv), (g_all, Nv))]
            keep += blk
            o, h = seq_op(blk[0], blk[1], blk[2], cur)
            keep += [o, h]
            ref_o.append([x.reshape(bw, Nv, Dv) for x in chips(o)])
            hists.append(h)
            cur = h
        o_all, cin, _ = verify(conv_all, beta_all, g_all, base, Nk, Nv, Dk, Dv, scale, T, bw, fid=fid, t0=t0,
                               t0_mf=t0_mf_served, row0=row0, row0_mf=row0_mf_served, **fz, **sz, **iz,
                               **dict(qz, vsplit_mf=vsplit_mf_served))
        keep += [o_all, cin]
        got_o = chips(o_all)
        ok = all(_same(got_o[c].reshape(T, bw, Nv, Dv)[j], ref_o[j][c]) for j in range(T) for c in range(len(got_o)))
        res["verify_o"] = "exact" if ok else "differs"
        if t0:
            # rows < bw: the sequential hist_0 (per chip); rows >= bw: the base, untouched
            h0 = chips(hists[0])
            exp = []
            for c in range(len(h0)):
                e = s_t.clone()
                e[:bw] = h0[c].reshape(bw, Nv, Dk, Dv)
                exp.append(e)
            res["verify_state_t0"] = "exact" if all(_same(x, e) for x, e in zip(chips(base), exp)) else "differs"
        else:
            res["verify_state_untouched"] = "exact" if all(_same(x, s_t) for x in chips(base)) else "differs"
        pages = selftest_pages(T, bw, Bmax, seed)

        def one(page, mf, cin_, base_):
            """commit_rec of the reference page vs commit() of `page`; True when every row is bitwise equal."""
            rpage = t0_reference_page(page, T, bw) if t0 else page
            ridx = up(torch.tensor(rpage, dtype=torch.int32).reshape(1, -1), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
            idx = up(torch.tensor(page, dtype=torch.int32).reshape(1, -1), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
            ref_st = up(s_t)
            got_st = ttnn.clone(base_, memory_config=DR) if t0 else up(s_t)  # t0: the post-verify state
            keep.extend([ridx, idx, ref_st, got_st])
            _ss.commit_rec(ref_st, hists, ridx, bw, Nv, Dk, Dv)
            commit(got_st, cin_, idx, Nk, Nv, Dk, Dv, scale, T, bw, mf=mf, fid=fid, t0=t0, row0=row0,
                   row0_mf=row0_mf_served, **fz, **iz)
            return all(_same(x, y) for x, y in zip(chips(ref_st), chips(got_st)))

        for k, page in enumerate(pages):
            res[f"commit_page{k}"] = "exact" if one(page, mf_served, cin, base) else "differs"
        res["mustfail_last_row"] = "exact" if one(pages[0], True, cin, base) else "differs"
        if t0:
            # explicit SO_T0 must-fail: a fresh base, the verify writes the last row after token 1
            base2 = up(s_t)
            keep.append(base2)
            o2, cin2, _ = verify(conv_all, beta_all, g_all, base2, Nk, Nv, Dk, Dv, scale, T, bw, fid=fid, t0=True,
                                 t0_mf=True, row0=row0, row0_mf=row0_mf_served, **fz, **sz, **iz, **qz)
            keep += [o2, cin2]
            res["mustfail_t0_verify"] = "exact" if one(pages[0], False, cin2, base2) else "differs"
        if row0:
            # explicit SFPU_ROW0 must-fail (the decay skips exp): o of a fresh verify must differ from the sequential
            base3 = up(s_t)
            keep.append(base3)
            o3, cin3, _ = verify(conv_all, beta_all, g_all, base3, Nk, Nv, Dk, Dv, scale, T, bw, fid=fid, t0=t0,
                                 t0_mf=False, row0=True, row0_mf=True, **fz, **sz, **iz, **qz)
            keep += [o3, cin3]
            got3 = chips(o3)
            ok3 = all(_same(got3[c].reshape(T, bw, Nv, Dv)[j], ref_o[j][c]) for j in range(T) for c in range(len(got3)))
            res["mustfail_sfpu_row0"] = "exact" if ok3 else "differs"
        if fuse and "delta" in fuse:
            # explicit SO_FUSE must-fail (the v - kn@h subtraction skipped): o of a fresh verify must differ
            base4 = up(s_t)
            keep.append(base4)
            o4, cin4, _ = verify(conv_all, beta_all, g_all, base4, Nk, Nv, Dk, Dv, scale, T, bw, fid=fid, t0=t0,
                                 t0_mf=False, row0=row0, row0_mf=False, fuse=fuse, fuse_mf=True, **sz, **iz, **qz)
            keep += [o4, cin4]
            got4 = chips(o4)
            ok4 = all(_same(got4[c].reshape(T, bw, Nv, Dv)[j], ref_o[j][c]) for j in range(T) for c in range(len(got4)))
            res["mustfail_so_fuse"] = "exact" if ok4 else "differs"
        if split:
            # explicit SO_SPLITREAD must-fail (every instance reads its neighbour's state): o must differ
            base5 = up(s_t)
            keep.append(base5)
            o5, cin5, _ = verify(conv_all, beta_all, g_all, base5, Nk, Nv, Dk, Dv, scale, T, bw, fid=fid, t0=t0,
                                 t0_mf=False, row0=row0, row0_mf=False, **fz, split=True, split_mf=True, **iz, **qz)
            keep += [o5, cin5]
            got5 = chips(o5)
            ok5 = all(_same(got5[c].reshape(T, bw, Nv, Dv)[j], ref_o[j][c]) for j in range(T) for c in range(len(got5)))
            res["mustfail_so_splitread"] = "exact" if ok5 else "differs"
        if iosrc:
            # explicit SO_IOSRC must-fail (the decay rotates the state's tiles): o of a fresh verify must differ
            base6 = up(s_t)
            keep.append(base6)
            o6, cin6, _ = verify(conv_all, beta_all, g_all, base6, Nk, Nv, Dk, Dv, scale, T, bw, fid=fid, t0=t0,
                                 t0_mf=False, row0=row0, row0_mf=False, **fz, split=split, split_mf=split_mf_served,
                                 iosrc=True, iosrc_mf=True, **qz)
            keep += [o6, cin6]
            got6 = chips(o6)
            ok6 = all(_same(got6[c].reshape(T, bw, Nv, Dv)[j], ref_o[j][c]) for j in range(T) for c in range(len(got6)))
            res["mustfail_iosrc"] = "exact" if ok6 else "differs"
        if qkh:
            # explicit SO_QKH must-fail (the last row's beta on the next token's row): o of a fresh verify must differ
            base7 = up(s_t)
            keep.append(base7)
            o7, cin7, _ = verify(conv_all, beta_all, g_all, base7, Nk, Nv, Dk, Dv, scale, T, bw, fid=fid, t0=t0,
                                 t0_mf=False, row0=row0, row0_mf=False, **fz, split=split, split_mf=split_mf_served,
                                 iosrc=iosrc, iosrc_mf=False, qkh=True, qkh_mf=True, vsplit=vcap,
                                 vsplit_mf=False)
            keep += [o7, cin7]
            got7 = chips(o7)
            ok7 = all(_same(got7[c].reshape(T, bw, Nv, Dv)[j], ref_o[j][c]) for j in range(T) for c in range(len(got7)))
            res["mustfail_so_qkh"] = "exact" if ok7 else "differs"
        if vP > 1:
            # SO_VSPLIT: the split verify (served, above: o_all/cin/base) vs the P=1 verify on a fresh base, bitwise
            base8 = up(s_t)
            keep.append(base8)
            o8, cin8, _ = verify(conv_all, beta_all, g_all, base8, Nk, Nv, Dk, Dv, scale, T, bw, fid=fid, t0=t0,
                                 t0_mf=t0_mf_served, row0=row0, row0_mf=row0_mf_served, **fz, **sz, **iz, qkh=qkh,
                                 qkh_mf=qkh_mf_served, vsplit=0, vsplit_mf=False)
            keep += [o8, cin8]
            same = all(_same(a, b) for a, b in zip(chips(o_all), chips(o8)))
            same = same and all(_same(a, b) for a, b in zip(chips(cin), chips(cin8)))
            same = same and all(_same(a, b) for a, b in zip(chips(base), chips(base8)))
            res["verify_vsplit_vs_p1"] = "exact" if same else "differs"
            base9 = up(s_t)
            keep.append(base9)
            o9, cin9, _ = verify(conv_all, beta_all, g_all, base9, Nk, Nv, Dk, Dv, scale, T, bw, fid=fid, t0=t0,
                                 t0_mf=False, row0=row0, row0_mf=False, **fz, split=split, split_mf=False, iosrc=iosrc,
                                 iosrc_mf=False, qkh=qkh, qkh_mf=False, vsplit=vcap, vsplit_mf=True)
            keep += [o9, cin9]
            got9 = chips(o9)
            ok9 = all(_same(got9[c].reshape(T, bw, Nv, Dv)[j], ref_o[j][c]) for j in range(T) for c in range(len(got9)))
            res["mustfail_so_vsplit"] = "exact" if ok9 else "differs"
    finally:
        for t in keep:
            ttnn.deallocate(t)
    return res


MUSTFAIL_KEYS = ("mustfail_last_row", "mustfail_t0_verify", "mustfail_sfpu_row0", "mustfail_so_fuse",
                 "mustfail_so_splitread", "mustfail_iosrc", "mustfail_so_qkh", "mustfail_so_vsplit")


def selftest_ok(res):
    """Every served check exact, every must-fail rung present in res differs (mustfail_last_row always; the SO_T0
    rung when the self-test ran with t0; the SFPU_ROW0 rung, required whenever QWEN36_MTP_GDN_SO_SFPU_ROW0=1)."""
    if not res or res.get("mustfail_last_row") != "differs":
        return False
    if ("verify_state_t0" in res) != ("mustfail_t0_verify" in res):
        return False
    if row0_enabled() and "mustfail_sfpu_row0" not in res:
        return False
    if "delta" in fuse_parts() and "mustfail_so_fuse" not in res:
        return False
    if split_enabled() and "mustfail_so_splitread" not in res:
        return False
    if iosrc_enabled() and "mustfail_iosrc" not in res:
        return False
    if qkh_enabled() and "mustfail_so_qkh" not in res:
        return False
    if ("verify_vsplit_vs_p1" in res) != ("mustfail_so_vsplit" in res):
        return False
    return all((v == "differs") if k in MUSTFAIL_KEYS else (v == "exact") for k, v in res.items())
