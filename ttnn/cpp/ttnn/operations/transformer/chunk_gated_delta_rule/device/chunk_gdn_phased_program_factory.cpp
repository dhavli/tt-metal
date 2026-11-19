#include "tt-metalium/tt_stack_public_defaults.hpp"
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Program factories for the phase-split chunk_gdn ops.
//   PREP: the state-independent per-chunk math. Every (head, chunk) pair is an independent
//         work-item (no cross-chunk dependency), so we fan the full BH*NC set of work-items
//         out across the ENTIRE compute grid — mirroring FLA's WY-prep Triton kernels, whose
//         launch grid spans (chunk-tile index, batch*head). This is the perf payoff of the
//         phase split: prep runs chunk-parallel across dozens of cores instead of 1 core/head.
//   SCAN: one Tensix core per head walks chunks sequentially carrying state S [K,V]
//         (inherently sequential — the recurrence forbids chunk-parallelism here).

#include "chunk_gdn_phased.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::prim {

// CB index plan — kept in sync with the prep/scan compute + dataflow kernels.
// Uniquely named (pcb) so it does not ODR-clash with the sibling factory's pcb:: under unity builds.
namespace pcb {
constexpr uint32_t q = tt::CBIndex::c_0;
constexpr uint32_t k = tt::CBIndex::c_1;
constexpr uint32_t v = tt::CBIndex::c_2;
constexpr uint32_t g = tt::CBIndex::c_3;
constexpr uint32_t beta = tt::CBIndex::c_4;
constexpr uint32_t eye = tt::CBIndex::c_5;
constexpr uint32_t tril = tt::CBIndex::c_6;
constexpr uint32_t ones = tt::CBIndex::c_7;
constexpr uint32_t S = tt::CBIndex::c_8;
constexpr uint32_t decay = tt::CBIndex::c_9;
constexpr uint32_t decay_exp = tt::CBIndex::c_10;
constexpr uint32_t decayfac = tt::CBIndex::c_11;  // prep; reused as cb_dl in scan
constexpr uint32_t lmask = tt::CBIndex::c_12;
constexpr uint32_t Tinv = tt::CBIndex::c_13;
constexpr uint32_t vbeta = tt::CBIndex::c_14;
constexpr uint32_t kbeta = tt::CBIndex::c_15;
constexpr uint32_t out = tt::CBIndex::c_16;
constexpr uint32_t u = tt::CBIndex::c_17;
constexpr uint32_t w = tt::CBIndex::c_18;
constexpr uint32_t qdecay = tt::CBIndex::c_19;
constexpr uint32_t intra = tt::CBIndex::c_20;
constexpr uint32_t s2 = tt::CBIndex::c_21;
constexpr uint32_t vnew = tt::CBIndex::c_22;
constexpr uint32_t ointer = tt::CBIndex::c_23;
constexpr uint32_t kdec_t = tt::CBIndex::c_24;
constexpr uint32_t supd = tt::CBIndex::c_25;
constexpr uint32_t stmp = tt::CBIndex::c_26;
constexpr uint32_t final_s = tt::CBIndex::c_27;
constexpr uint32_t scr1 = tt::CBIndex::c_28;
constexpr uint32_t scr2 = tt::CBIndex::c_29;
constexpr uint32_t scr3 = tt::CBIndex::c_30;
constexpr uint32_t s3 = tt::CBIndex::c_31;
constexpr uint32_t dl = decayfac;  // scan reads dl into this slot
}  // namespace pcb

namespace {

ComputeConfigDescriptor compute_cfg() {
    return ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false};
}

// Chunk-parallel work distribution for PREP: split `total` independent (head, chunk) work-items
// as evenly as possible across the compute grid. Work-item wi in [0,total) maps directly to the
// flat DRAM tile index used by every prep tensor ([BH, NC, ...] => wi = h*NC + c), so a core that
// is assigned a contiguous range [wi_start, wi_start+wi_count) reads/writes exactly those tiles —
// the DRAM result is byte-identical no matter how items are partitioned. The first `rem` cores get
// one extra item (ceil), the rest get floor; every used core gets >= 1 (since P <= total).
struct PrepWorkDist {
    std::vector<CoreCoord> cores;  // used cores, in assignment order
    std::vector<uint32_t> wi_start;
    std::vector<uint32_t> wi_count;
    CoreRangeSet core_set;  // all used cores (kernel placement + CB alloc)
};

// `core_cap` bounds how many cores we may use (defaults to the whole grid). Perf A/B only:
// setting core_cap=BH reproduces the old "1 core/head, NC chunks serial" layout exactly (each
// head's NC work-items are contiguous, wi in [h*NC, h*NC+NC)), using the identical kernels.
PrepWorkDist distribute_prep(CoreCoord grid, uint32_t total, uint32_t core_cap) {
    const uint32_t max_cores = std::min<uint32_t>(grid.x * grid.y, core_cap);
    const uint32_t P = std::min(total, max_cores);
    TT_FATAL(P > 0, "prep work distribution needs >= 1 work-item (total={})", total);
    const uint32_t base = total / P;
    const uint32_t rem = total % P;  // first `rem` cores get base+1

    PrepWorkDist d;
    d.cores.reserve(P);
    d.wi_start.reserve(P);
    d.wi_count.reserve(P);
    std::set<CoreRange> crs;
    uint32_t off = 0;
    for (uint32_t i = 0; i < P; i++) {
        const CoreCoord core{i % grid.x, i / grid.x};  // row-major over the grid
        const uint32_t cnt = base + (i < rem ? 1u : 0u);
        d.cores.push_back(core);
        d.wi_start.push_back(off);
        d.wi_count.push_back(cnt);
        crs.insert(CoreRange{core, core});
        off += cnt;
    }
    d.core_set = CoreRangeSet{crs}.merge_ranges();  // cfx 0030
    return d;
}

// Value-parallel work distribution for SCAN. The chunk recurrence is sequential in TIME, but it
// factorizes EXACTLY over the value dimension: each V-block S[:, vb] evolves independently (every
// scan op — kd@S, T_inv@diff, q_decay@S, intra@v_new, k_dec_t@v_new, S*dl — is column-wise in V,
// needing only the full-K shared per-chunk tensors + that block's own V-slice). This mirrors FLA's
// fwd_h/fwd_o launch grid over (i_v, i_bh) with a sequential loop over time. K is NOT split (v_new
// and o reduce over all K). We pick the finest V-blocking that fits the grid: the largest NV | Vt
// with BH*NV <= cores. Each core runs one (head, v-block) => BH*NV independent scans vs BH today.
struct ScanWorkDist {
    std::vector<CoreCoord> cores;
    std::vector<uint32_t> head;  // head index per core
    std::vector<uint32_t> vblk;  // v-block index per core
    uint32_t Vtl = 1;            // per-core v-block width (tiles) = Vt / NV
    uint32_t NV = 1;             // v-blocks per head
    CoreRangeSet core_set;
};

ScanWorkDist distribute_scan(CoreCoord grid, uint32_t BH, uint32_t Vt) {
    const uint32_t ncores = grid.x * grid.y;
    TT_FATAL(BH <= ncores, "num_heads {} exceeds compute cores {}", BH, ncores);
    // QWEN_GDN_SCAN_SERIAL=1 forces NV=1 (full V on 1 core/head, the old layout) for perf A/B only.
    const char* serial_env = tt_stack_public::getenv("QWEN_GDN_SCAN_SERIAL");
    const bool force_serial = serial_env && serial_env[0] == '1';
    uint32_t NV = 1;
    for (uint32_t cand = force_serial ? 1u : Vt; cand >= 1; cand--) {  // cand==1 always satisfies
        if (Vt % cand == 0 && BH * cand <= ncores) {
            NV = cand;
            break;
        }
    }
    ScanWorkDist d;
    d.NV = NV;
    d.Vtl = Vt / NV;
    const uint32_t total = BH * NV;
    d.cores.reserve(total);
    d.head.reserve(total);
    d.vblk.reserve(total);
    std::set<CoreRange> crs;
    for (uint32_t i = 0; i < total; i++) {
        const CoreCoord core{i % grid.x, i / grid.x};  // row-major over the grid
        d.cores.push_back(core);
        d.head.push_back(i / NV);  // heads' v-blocks grouped contiguously
        d.vblk.push_back(i % NV);
        crs.insert(CoreRange{core, core});
    }
    d.core_set = CoreRangeSet{crs}.merge_ranges();  // cfx 0030
    return d;
}

// cfx (tt-stack 0028): placement for the scan's shared-tile multicast (upstream 5f02a6718f4).
// Upstream forces each head's NV v-block cores into one 1xNV ROW and lowers NV until BH such rows
// fit: on an 11x10 grid at BH=24 that is NV=2 (48 cores, half the V parallelism). Here NV is the
// DEFAULT schedule's (distribute_scan: largest NV | Vt with BH*NV <= cores, so every core keeps the
// default's per-core V-slice and arithmetic) and only the placement changes: each head's NV cores
// form a w x h rectangle (w*h == NV) -- a 1xNV row when BH rows fit (upstream's layout), else the
// most square shape that does (2x2 for NV=4: 5x5 = 25 blocks >= 24 heads on 11x10). v-block vb sits
// at (x0 + vb % w, y0 + vb / w), so v-block 0 (the sender) is the rectangle's top-left core.
// Returns false (the caller FATALs: a requested multicast never falls back silently) when NV < 2 or
// no rectangle shape fits.
bool distribute_scan_mcast(CoreCoord grid, uint32_t BH, uint32_t Vt, ScanWorkDist& d, uint32_t& rw, uint32_t& rh) {
    d = distribute_scan(grid, BH, Vt);  // the default NV / Vtl (and its env handling)
    const uint32_t NV = d.NV;
    if (NV < 2) {
        return false;
    }
    std::vector<std::pair<uint32_t, uint32_t>> shapes;  // (w, h): the 1xNV row first, then most square
    shapes.emplace_back(NV, 1u);
    std::vector<std::pair<uint32_t, uint32_t>> rest;
    for (uint32_t h = 2; h <= NV; ++h) {
        if (NV % h == 0) {
            rest.emplace_back(NV / h, h);
        }
    }
    std::stable_sort(rest.begin(), rest.end(), [](const auto& a, const auto& b) {
        const int da = std::abs(static_cast<int>(a.first) - static_cast<int>(a.second));
        const int db = std::abs(static_cast<int>(b.first) - static_cast<int>(b.second));
        return (da != db) ? da < db : a.first > b.first;
    });
    shapes.insert(shapes.end(), rest.begin(), rest.end());
    for (const auto& [w, h] : shapes) {
        if (w > grid.x || h > grid.y) {
            continue;
        }
        const uint32_t per_row = grid.x / w;
        if (BH > per_row * (grid.y / h)) {
            continue;
        }
        rw = w;
        rh = h;
        std::set<CoreRange> crs;
        for (uint32_t i = 0; i < BH * NV; i++) {
            const uint32_t hd = i / NV;
            const uint32_t vb = i % NV;
            const CoreCoord core{(hd % per_row) * w + vb % w, (hd / per_row) * h + vb / w};
            d.cores[i] = core;
            d.head[i] = hd;
            d.vblk[i] = vb;
            crs.insert(CoreRange{core, core});
        }
        d.core_set = CoreRangeSet{crs}.merge_ranges();  // cfx 0030
        return true;
    }
    return false;
}

}  // namespace

// ---------------------------------------------------------------------------
// PREP
// ---------------------------------------------------------------------------
tt::tt_metal::ProgramDescriptor ChunkGdnPrepProgramFactory::create_descriptor(
    const ChunkGdnPrepParams& attrs, const ChunkGdnPrepInputs& in, std::vector<Tensor>& outputs) {
    const uint32_t BH = attrs.BH;
    const uint32_t NC = attrs.num_chunks;
    const uint32_t Ct = attrs.chunk_size / TILE_HEIGHT;
    const uint32_t Kt = attrs.key_dim / TILE_WIDTH;
    const uint32_t Vt = attrs.val_dim / TILE_WIDTH;

    const uint32_t cc = Ct * Ct, ck = Ct * Kt, cv = Ct * Vt, kv = Kt * Vt, kc = Kt * Ct;
    uint32_t scr = std::max({cc, ck, cv, kv, kc});

    const tt::DataFormat df_io = tt::DataFormat::Float16_b;  // bf16 q/k/v

    auto* device = in.q.device();
    // Fan the BH*NC independent (head, chunk) prep work-items across the whole grid.
    // QWEN_GDN_PREP_SERIAL=1 caps to BH cores (old 1-core/head layout) for perf A/B only.
    const uint32_t total_work = BH * NC;
    const char* serial_env = tt_stack_public::getenv("QWEN_GDN_PREP_SERIAL");
    const uint32_t core_cap = (serial_env && serial_env[0] == '1') ? BH : ~0u;
    auto dist = distribute_prep(device->compute_with_storage_grid_size(), total_work, core_cap);
    const CoreRangeSet& cores = dist.core_set;
    const uint32_t n_used = static_cast<uint32_t>(dist.cores.size());

    ProgramDescriptor desc;
    auto add_cb = [&](uint32_t idx, uint32_t n_tiles, uint32_t nbuf = 1, tt::DataFormat fmt = tt::DataFormat::Float32) {
        const uint32_t ts = tt::tile_size(fmt);
        desc.cbs.push_back(CBDescriptor{
            .total_size = n_tiles * nbuf * ts,
            .core_ranges = cores,
            .format_descriptors = {
                {CBFormatDescriptor{.buffer_index = static_cast<uint8_t>(idx), .data_format = fmt, .page_size = ts}}}});
    };

    // Allocate the full CB set in the SAME order/sizes as the monolithic op, so the prep phase's
    // L1 layout is byte-identical (the Horner's matmul L1 access pattern is layout-sensitive).
    add_cb(pcb::q, ck, 1, df_io);
    add_cb(pcb::k, ck, 1, df_io);
    add_cb(pcb::v, cv, 1, df_io);
    add_cb(pcb::g, Ct);
    add_cb(pcb::beta, Ct);
    add_cb(pcb::eye, cc);
    add_cb(pcb::tril, cc);
    add_cb(pcb::ones, cc);
    add_cb(pcb::S, kv, 2);
    add_cb(pcb::decay, Ct);
    add_cb(pcb::decay_exp, Ct);
    add_cb(pcb::decayfac, Ct);
    add_cb(pcb::lmask, cc);
    add_cb(pcb::Tinv, cc);
    add_cb(pcb::vbeta, cv);
    add_cb(pcb::kbeta, ck);
    add_cb(pcb::out, cv, 2, df_io);
    add_cb(pcb::u, cv);
    add_cb(pcb::w, ck);
    add_cb(pcb::qdecay, ck);
    add_cb(pcb::intra, cc);
    add_cb(pcb::s2, kv, 2);
    add_cb(pcb::vnew, cv);  // aliased as cb_dl in the prep kernel (1 tile used)
    add_cb(pcb::ointer, cv);
    add_cb(pcb::kdec_t, kc);
    add_cb(pcb::supd, kv);
    add_cb(pcb::stmp, kv);
    add_cb(pcb::final_s, kv);
    add_cb(pcb::scr1, scr);
    add_cb(pcb::scr2, scr);
    add_cb(pcb::scr3, scr);
    add_cb(pcb::s3, kv, 2);

    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/";
    const std::vector<uint32_t> ct_args = {Ct, Kt, Vt};

    std::vector<uint32_t> reader_ct = ct_args;
    TensorAccessorArgs(*in.q.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.k.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.v.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.g.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.beta.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.eye_c.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.tril_c.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.ones_c.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.masks_c.buffer()).append_to(reader_ct);
    // OPT-A: trailing compile args after all TensorAccessorArgs — 1 => read that tensor flat token-major.
    reader_ct.push_back(attrs.v_flat ? 1u : 0u);
    reader_ct.push_back(attrs.qk_flat ? 1u : 0u);

    std::vector<uint32_t> writer_ct = ct_args;
    for (auto& t : outputs) {
        TensorAccessorArgs(*t.buffer()).append_to(writer_ct);
    }

    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_chunk_gdn_prep.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.compile_time_args = reader_ct;
    reader.config = ReaderConfigDescriptor{};
    reader.runtime_args.reserve(n_used);

    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_chunk_gdn_prep.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = writer_ct;
    writer.config = WriterConfigDescriptor{};
    writer.runtime_args.reserve(n_used);

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/chunk_gdn_prep.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    // Compute gets extra args for the in-kernel q/k L2-norm (OPT-B): QK_NORM flag, and scale/eps as
    // fp32 bit patterns. QK_NORM is only enabled for Ct==1 (attrs.qk_norm gated on chunk_size==32).
    auto f32_bits = [](float f) {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        return u;
    };
    std::vector<uint32_t> compute_ct = ct_args;
    compute_ct.push_back(attrs.qk_norm ? 1u : 0u);
    compute_ct.push_back(f32_bits(attrs.scale));
    compute_ct.push_back(f32_bits(1e-6f));
    compute.compile_time_args = compute_ct;
    compute.config = compute_cfg();
    compute.runtime_args.reserve(n_used);

    auto* q_buf = in.q.buffer();
    auto* k_buf = in.k.buffer();
    auto* v_buf = in.v.buffer();
    auto* g_buf = in.g.buffer();
    auto* beta_buf = in.beta.buffer();
    auto* eye_buf = in.eye_c.buffer();
    auto* tril_buf = in.tril_c.buffer();
    auto* ones_buf = in.ones_c.buffer();
    auto* masks_buf = in.masks_c.buffer();
    auto* vb_buf = outputs[0].buffer();    // v_beta
    auto* kd_buf = outputs[1].buffer();    // kd = k_beta*decay_exp
    auto* qd_buf = outputs[2].buffer();    // q_decay
    auto* it_buf = outputs[3].buffer();    // intra
    auto* kdec_buf = outputs[4].buffer();  // k_dec_t
    auto* dl_buf = outputs[5].buffer();    // dl
    auto* ti_buf = outputs[6].buffer();    // t_inv

    // Each used core processes its contiguous slice [wi_start, wi_start+wi_count) of the BH*NC
    // work-items. wi is the flat DRAM tile index (h*NC + c), so the kernels need no h/c at all.
    for (uint32_t i = 0; i < n_used; i++) {
        const auto& core = dist.cores[i];
        const uint32_t wi_start = dist.wi_start[i];
        const uint32_t wi_count = dist.wi_count[i];
        // Trailing runtime args NC, HV, Hk are consumed by the reader's flat branches (V_FLAT/QK_FLAT).
        reader.emplace_runtime_args(
            core,
            {wi_start,
             wi_count,
             q_buf,
             k_buf,
             v_buf,
             g_buf,
             beta_buf,
             eye_buf,
             tril_buf,
             ones_buf,
             masks_buf,
             NC,
             attrs.HV,
             attrs.Hk});
        writer.emplace_runtime_args(
            core, {wi_start, wi_count, vb_buf, kd_buf, qd_buf, it_buf, kdec_buf, dl_buf, ti_buf});
        compute.emplace_runtime_args(core, {wi_count});
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

// ---------------------------------------------------------------------------
// SCAN
// ---------------------------------------------------------------------------
tt::tt_metal::ProgramDescriptor ChunkGdnScanProgramFactory::create_descriptor(
    const ChunkGdnScanParams& attrs, const ChunkGdnScanInputs& in, std::vector<Tensor>& outputs) {
    const uint32_t BH = attrs.BH;
    const uint32_t NC = attrs.num_chunks;
    const uint32_t Ct = attrs.chunk_size / TILE_HEIGHT;
    const uint32_t Kt = attrs.key_dim / TILE_WIDTH;
    const uint32_t Vt_full = attrs.val_dim / TILE_WIDTH;
    const uint32_t has_s0 = attrs.has_initial_state ? 1u : 0u;

    // o output is fp32 (matches the scan op's compute_output_specs; a bf16 o degraded full-model
    // quality and was removed). cb_out format must match, else the writer strides wrong.
    const tt::DataFormat df_io = tt::DataFormat::Float32;

    auto* device = in.v_beta.device();
    // Value-parallel fan-out: each core runs one (head, v-block) sequential scan.
    // cfx (tt-stack 0028): with use_mcast the same NV / per-core V-slice is kept and each head's NV
    // cores are re-placed on a rectangle (see distribute_scan_mcast); without it this is the image's
    // placement and program, byte for byte.
    const bool do_mcast = attrs.use_mcast;
    uint32_t mc_w = 0, mc_h = 0;
    ScanWorkDist sdist;
    if (do_mcast) {
        // TT_GDN_SCAN_MCAST_GRID=WxH (test only, multicast path only): place on at most a WxH corner of the
        // grid. ttsim's Blackhole grid is 13x10, where BH=24 fits 1x4 rows; 11x10 reproduces the cards' 2x2
        // placement there. The default phased path never reads it.
        CoreCoord mc_grid = device->compute_with_storage_grid_size();
        if (const char* e = tt_stack_public::getenv("TT_GDN_SCAN_MCAST_GRID"); e != nullptr && e[0] != '\0') {
            unsigned gx = 0, gy = 0;
            TT_FATAL(std::sscanf(e, "%ux%u", &gx, &gy) == 2 && gx > 0 && gy > 0,
                     "TT_GDN_SCAN_MCAST_GRID must be WxH, got '{}'", e);
            mc_grid = CoreCoord{std::min<std::size_t>(mc_grid.x, gx), std::min<std::size_t>(mc_grid.y, gy)};
        }
        const bool placed = distribute_scan_mcast(mc_grid, BH, Vt_full, sdist, mc_w, mc_h);
        TT_FATAL(
            placed,
            "chunk_gdn_scan: use_mcast requested but not applicable: BH={} Vt={} grid={}x{} NV={} (needs NV >= 2 and "
            "BH rectangles of NV cores on the grid)",
            BH,
            Vt_full,
            mc_grid.x,
            mc_grid.y,
            sdist.NV);
    } else {
        sdist = distribute_scan(device->compute_with_storage_grid_size(), BH, Vt_full);
    }
    const CoreRangeSet& cores = sdist.core_set;
    const uint32_t Vt = sdist.Vtl;  // per-core V-block width; CBs/compute use this
    const uint32_t n_used = static_cast<uint32_t>(sdist.cores.size());

    // V-independent tensors (kd, q_decay, intra, k_dec_t, T_inv, dl) are read in FULL; only the
    // V-dependent CBs (v_beta/state/out/scratch) shrink to the per-core V-block width Vt(=Vtl).
    const uint32_t cc = Ct * Ct, ck = Ct * Kt, cv = Ct * Vt, kv = Kt * Vt, kc = Kt * Ct;
    uint32_t scr = std::max({cc, ck, cv, kv, kc});

    ProgramDescriptor desc;
    auto add_cb = [&](uint32_t idx, uint32_t n_tiles, uint32_t nbuf = 1, tt::DataFormat fmt = tt::DataFormat::Float32) {
        const uint32_t ts = tt::tile_size(fmt);
        desc.cbs.push_back(CBDescriptor{
            .total_size = n_tiles * nbuf * ts,
            .core_ranges = cores,
            .format_descriptors = {
                {CBFormatDescriptor{.buffer_index = static_cast<uint8_t>(idx), .data_format = fmt, .page_size = ts}}}});
    };

    // Per-chunk inputs (streamed from DRAM). u-slot holds v_beta, w-slot holds kd. nbuf=1.
    add_cb(pcb::u, cv, 1);  // v_beta
    add_cb(pcb::w, ck, 1);  // kd
    add_cb(pcb::qdecay, ck, 1);
    add_cb(pcb::intra, cc, 1);
    add_cb(pcb::kdec_t, kc, 1);
    add_cb(pcb::dl, 1, 1);
    add_cb(pcb::Tinv, cc, 1);  // t_inv (WY inverse)
    // State: cb_S is reader-produced (chunk 0 only); s2/s3 are compute-only ping-pong.
    add_cb(pcb::S, kv);
    add_cb(pcb::s2, kv);
    add_cb(pcb::s3, kv);
    // Outputs.
    add_cb(pcb::out, cv, 2, df_io);
    add_cb(pcb::final_s, kv);
    // Scratch.
    add_cb(pcb::vnew, cv);
    add_cb(pcb::ointer, cv);
    add_cb(pcb::supd, kv);
    add_cb(pcb::stmp, kv);
    add_cb(pcb::scr1, scr);

    // cfx (tt-stack 0028): per-head multicast of the six shared V-independent inputs (kd, q_decay,
    // intra, k_dec_t, dl, t_inv) -- upstream 5f02a6718f4. The head's v-block-0 core reads them from
    // DRAM once per chunk and multicasts them into its NV-1 siblings' CBs, which would otherwise
    // re-read the identical DRAM pages. Receivers get the same bytes into the same CB slots and the
    // compute kernel is unchanged, so the result is bit-identical by construction.
    CoreRangeSet sender_set, receiver_set;
    std::vector<CoreCoord> head_rect_start(BH), head_rect_end(BH), head_sender(BH);
    if (do_mcast) {
        std::set<CoreRange> s_crs, r_crs;
        for (uint32_t i = 0; i < n_used; i++) {
            (sdist.vblk[i] == 0 ? s_crs : r_crs).insert(CoreRange{sdist.cores[i], sdist.cores[i]});
        }
        sender_set = CoreRangeSet{s_crs};
        receiver_set = CoreRangeSet{r_crs};
        // Handshake semaphores on ALL scan cores so each id resolves to the same L1 address on sender
        // and receivers. Ids 0 (ready) / 1 (valid) are mirrored as SEM_READY / SEM_VALID constants in
        // kernels_scan_mcast/dataflow/reader_chunk_gdn_scan_mcast.cpp -- keep in sync.
        desc.semaphores.push_back(
            SemaphoreDescriptor{.id = 0, .core_type = tt::CoreType::WORKER, .core_ranges = cores, .initial_value = 0});
        desc.semaphores.push_back(
            SemaphoreDescriptor{.id = 1, .core_type = tt::CoreType::WORKER, .core_ranges = cores, .initial_value = 0});
        // Multicast rectangle per head, in virtual worker coords (the reader runs on NOC_0: top-left ->
        // bottom-right, unswapped). The whole w x h block including the sender: a non-loopback multicast
        // never writes the sending core, and num_dests = NV-1 counts only the siblings. Tightness (as the
        // SDPA GQA packer): every worker core whose NoC coords fall inside must belong to this head.
        const CoreCoord full_grid = device->logical_grid_size();
        for (uint32_t hd = 0; hd < BH; hd++) {
            const CoreCoord tl = sdist.cores[hd * sdist.NV];
            const CoreCoord br = sdist.cores[hd * sdist.NV + sdist.NV - 1];
            const CoreCoord p0 = device->worker_core_from_logical_core(tl);
            const CoreCoord p1 = device->worker_core_from_logical_core(br);
            const uint32_t min_x = std::min<uint32_t>(p0.x, p1.x), max_x = std::max<uint32_t>(p0.x, p1.x);
            const uint32_t min_y = std::min<uint32_t>(p0.y, p1.y), max_y = std::max<uint32_t>(p0.y, p1.y);
            uint32_t inside = 0;
            for (uint32_t y = 0; y < full_grid.y; ++y) {
                for (uint32_t x = 0; x < full_grid.x; ++x) {
                    const CoreCoord p = device->worker_core_from_logical_core(CoreCoord{x, y});
                    if (p.x >= min_x && p.x <= max_x && p.y >= min_y && p.y <= max_y) {
                        ++inside;
                    }
                }
            }
            TT_FATAL(
                inside == sdist.NV,
                "chunk_gdn_scan: use_mcast head {} rectangle ({},{})-({},{}) holds {} worker cores, expected NV={}",
                hd,
                min_x,
                min_y,
                max_x,
                max_y,
                inside,
                sdist.NV);
            head_rect_start[hd] = CoreCoord{min_x, min_y};
            head_rect_end[hd] = CoreCoord{max_x, max_y};
            head_sender[hd] = p0;
        }
        log_info(
            tt::LogOp,
            "chunk_gdn_scan: shared-tile multicast ENABLED (BH={}, NV={}, Vtl={}, rect={}x{}, cores={}, NC={})",
            BH,
            sdist.NV,
            sdist.Vtl,
            mc_w,
            mc_h,
            n_used,
            NC);
    }

    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/";
    // ct arg 2 = per-core Vt(=Vtl); arg 4 = Vt_full (full V in tiles) for the readers'/writer's
    // V-slice row stride. Compute reads only args 0..2 (Ct, Kt, Vt) so the extra arg is harmless.
    const std::vector<uint32_t> ct_args = {Ct, Kt, Vt, has_s0, Vt_full};

    std::vector<uint32_t> reader_ct = ct_args;
    TensorAccessorArgs(*in.v_beta.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.kd.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.q_decay.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.intra.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.k_dec_t.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.dl.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.t_inv.buffer()).append_to(reader_ct);
    TensorAccessorArgs(in.initial_state.has_value() ? in.initial_state->buffer() : nullptr).append_to(reader_ct);

    // Mcast receivers read only their private V-sliced tensors (v_beta, s0) from DRAM.
    std::vector<uint32_t> receiver_ct;
    if (do_mcast) {
        receiver_ct = ct_args;
        TensorAccessorArgs(*in.v_beta.buffer()).append_to(receiver_ct);
        TensorAccessorArgs(in.initial_state.has_value() ? in.initial_state->buffer() : nullptr).append_to(receiver_ct);
    }

    std::vector<uint32_t> writer_ct = ct_args;
    TensorAccessorArgs(*outputs[0].buffer()).append_to(writer_ct);
    TensorAccessorArgs(*outputs[1].buffer()).append_to(writer_ct);

    // Plain reader (the image's file) or, with use_mcast, the sender variant of 5f02a6718f4's reader. The
    // multicast reader lives in its own directory so kernels/ stays the image's file set, byte for byte.
    const std::string kdir_mcast =
        "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels_scan_mcast/";
    KernelDescriptor reader;
    reader.kernel_source = do_mcast ? kdir_mcast + "dataflow/reader_chunk_gdn_scan_mcast.cpp"
                                    : kdir + "dataflow/reader_chunk_gdn_scan.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = do_mcast ? sender_set : cores;
    reader.compile_time_args = reader_ct;
    if (do_mcast) {
        reader.defines = {{"GDN_MCAST_SENDER", "1"}};
    }
    reader.config = ReaderConfigDescriptor{};
    reader.runtime_args.reserve(do_mcast ? BH : n_used);

    KernelDescriptor receiver;  // used only with use_mcast
    if (do_mcast) {
        receiver.kernel_source = kdir_mcast + "dataflow/reader_chunk_gdn_scan_mcast.cpp";
        receiver.source_type = KernelDescriptor::SourceType::FILE_PATH;
        receiver.core_ranges = receiver_set;
        receiver.compile_time_args = receiver_ct;
        receiver.defines = {{"GDN_MCAST_RECEIVER", "1"}};
        receiver.config = ReaderConfigDescriptor{};
        receiver.runtime_args.reserve(n_used - BH);
    }

    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_chunk_gdn_scan.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = writer_ct;
    writer.config = WriterConfigDescriptor{};
    writer.runtime_args.reserve(n_used);

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/chunk_gdn_scan.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = ct_args;
    compute.config = compute_cfg();
    compute.runtime_args.reserve(n_used);

    auto* vb_buf = in.v_beta.buffer();
    auto* kd_buf = in.kd.buffer();
    auto* qd_buf = in.q_decay.buffer();
    auto* it_buf = in.intra.buffer();
    auto* kdec_buf = in.k_dec_t.buffer();
    auto* dl_buf = in.dl.buffer();
    auto* ti_buf = in.t_inv.buffer();
    auto* s0_buf = in.initial_state.has_value() ? in.initial_state->buffer() : nullptr;
    auto* o_buf = outputs[0].buffer();
    auto* fs_buf = outputs[1].buffer();

    for (uint32_t i = 0; i < n_used; i++) {
        const auto& core = sdist.cores[i];
        const uint32_t h = sdist.head[i];
        const uint32_t vb = sdist.vblk[i];
        if (!do_mcast) {
            reader.emplace_runtime_args(
                core, {h, vb, NC, vb_buf, kd_buf, qd_buf, it_buf, kdec_buf, dl_buf, ti_buf, s0_buf});
        } else if (vb == 0) {
            reader.emplace_runtime_args(
                core,
                {h,
                 vb,
                 NC,
                 vb_buf,
                 kd_buf,
                 qd_buf,
                 it_buf,
                 kdec_buf,
                 dl_buf,
                 ti_buf,
                 s0_buf,
                 static_cast<uint32_t>(head_rect_start[h].x),
                 static_cast<uint32_t>(head_rect_start[h].y),
                 static_cast<uint32_t>(head_rect_end[h].x),
                 static_cast<uint32_t>(head_rect_end[h].y),
                 sdist.NV - 1});
        } else {
            receiver.emplace_runtime_args(
                core,
                {h,
                 vb,
                 NC,
                 vb_buf,
                 s0_buf,
                 static_cast<uint32_t>(head_sender[h].x),
                 static_cast<uint32_t>(head_sender[h].y)});
        }
        writer.emplace_runtime_args(core, {h, vb, NC, o_buf, fs_buf});
        compute.emplace_runtime_args(core, {NC});
    }

    desc.kernels.push_back(std::move(reader));
    if (do_mcast) {
        desc.kernels.push_back(std::move(receiver));
    }
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::prim
