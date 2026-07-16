"""
evaluate_full_comparison.py — Comprehensive denoising algorithm comparison
on blob synthetic data with per-channel background white noise.

Compares:
  Raw            — no processing (baseline)
  StatGate_k3    — Stage1 hard gate (k=3.0)
  StatGate_k1    — Stage1 hard gate (k=1.0)
  Spatial        — gaussian spatial smoothing (sigma=1.5)
  Hybrid         — StatGate + Spatial
  WienerGate     — WebRTC-style soft gain (k=1.0)
  WienerGate_k1.5 — WebRTC-style soft gain (k=1.5)

Metrics computed on SIGNAL frames only (background frames excluded).
Uses the same metric definitions as 评估指标设计文档.md.
"""
from __future__ import annotations

import os, sys, csv, time
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "..", "src"))
from alg.noise_suppressor import Analyzer, Processor

DATA_DIR = os.path.join(BASE_DIR, "data", "synthetic_blob")
ROWS, COLS = 64, 64
N_BG = 15

# =====================================================
# WienerGate implementation (inlined for self-contained script)
# =====================================================
class WienerGate:
    def __init__(self, k_sigma=1.0, over_sub=1.0, dd_alpha=0.98, min_snr=0.1):
        self.k_sigma = k_sigma
        self.over_sub = over_sub
        self.dd_alpha = dd_alpha
        self.min_snr = min_snr
        self.baseline = None
        self.noise_std = None
        self.n_channels = 0
        self.prior_snr = None

    def fit(self, bg_frames):
        N, C = bg_frames.shape
        self.n_channels = C
        self.baseline = np.median(bg_frames, axis=0)
        self.noise_std = np.std(bg_frames, axis=0, ddof=1)
        self.noise_std = np.maximum(self.noise_std, 0.5)
        self.prior_snr = np.full(C, self.min_snr, dtype=np.float64)

    def process_frame(self, frame):
        residual = np.maximum(frame - self.baseline, 0.0)
        noise_power = self.noise_std * self.noise_std
        post_snr_power = residual * residual / noise_power
        post_snr = np.maximum(post_snr_power - 1.0, 0.0)
        self.prior_snr = (
            self.dd_alpha * self.prior_snr
            + (1.0 - self.dd_alpha) * post_snr
        )
        self.prior_snr = np.maximum(self.prior_snr, self.min_snr)
        gain = self.prior_snr / (self.over_sub + self.prior_snr)
        return residual * gain

    def process_batch(self, frames):
        N = frames.shape[0]
        out = np.empty_like(frames)
        for i in range(N):
            out[i] = self.process_frame(frames[i])
        return out


# =====================================================
# Metrics (signal frames only)
# =====================================================
def m_rmse(p, g):
    return float(np.sqrt(np.mean((p - g) ** 2)))

def m_snr(p, g):
    mse = np.mean((p - g) ** 2)
    vt = np.var(g)
    if mse < 1e-12 or vt < 1e-12:
        return 0.0
    return float(10 * np.log10(vt / mse))

def m_bnsr(p, g, src):
    mask = g < 1e-6
    if mask.sum() < 1:
        return 0.0
    vi = np.var(src[mask])
    vo = np.var(p[mask])
    if vo < 1e-12 or vi < 1e-12:
        return 0.0
    return float(10 * np.log10(vi / vo))

def m_srr(p, g):
    mask = g >= 1e-6
    if mask.sum() < 1:
        return 1.0
    pm = np.abs(p[mask]).mean()
    gm = np.abs(g[mask]).mean()
    return float(pm / gm) if gm > 1e-12 else 1.0

def m_ssim(p, g):
    N = p.shape[0]
    vals = []
    for i in range(N):
        p2 = p[i].reshape(ROWS, COLS)
        g2 = g[i].reshape(ROWS, COLS)
        mx = g2.mean()
        my = p2.mean()
        vx = g2.var()
        vy = p2.var()
        cxy = np.mean((g2 - mx) * (p2 - my))
        c1, c2 = 1e-4, 1e-4
        num = (2 * mx * my + c1) * (2 * cxy + c2)
        den = (mx ** 2 + my ** 2 + c1) * (vx + vy + c2)
        vals.append(float(num / den) if den > 1e-12 else 0.0)
    return float(np.mean(vals))

def m_centroid(p, g):
    shifts = []
    for i in range(p.shape[0]):
        p2 = p[i].reshape(ROWS, COLS)
        g2 = g[i].reshape(ROWS, COLS)
        if g2.sum() < 1e-6:
            continue
        ys, xs = np.mgrid[0:ROWS, 0:COLS]
        sp = p2.sum()
        sg = g2.sum()
        if sp < 1e-6 or sg < 1e-6:
            continue
        cxp = (xs * p2).sum() / sp
        cyp = (ys * p2).sum() / sp
        cxg = (xs * g2).sum() / sg
        cyg = (ys * g2).sum() / sg
        shifts.append(np.sqrt((cxp - cxg) ** 2 + (cyp - cyg) ** 2))
    return float(np.mean(shifts)) if shifts else 0.0

def m_cs(snr, srr, ssim_v, cent):
    sn = max(0, min(1, (snr + 10) / 50))
    cn = max(0, 1 - cent / 32)
    return 0.25 * sn + 0.25 * min(srr, 1) + 0.30 * max(ssim_v, 0) + 0.20 * cn

def m_temp(p, g):
    if p.shape[0] < 2:
        return 1.0
    pd = np.abs(np.diff(p, axis=0)).mean()
    gd = np.abs(np.diff(g, axis=0)).mean()
    return float(pd / gd) if gd > 1e-12 else 1.0

def m_area(p, g, thr=0.1):
    ratios = []
    for i in range(p.shape[0]):
        pp = p[i].reshape(ROWS, COLS)
        gg = g[i].reshape(ROWS, COLS)
        if gg.max() < 1e-6:
            continue
        tp = pp.max() * thr
        tg = gg.max() * thr
        ap = (pp >= tp).sum()
        ag = (gg >= tg).sum()
        ratios.append(ap / ag if ag > 0 else 1.0)
    return float(np.mean(ratios)) if ratios else 1.0


# =====================================================
# Algorithm definitions
# =====================================================
def algo_raw(frames, model, bg=None):
    return frames.copy()

def algo_stat_k3(frames, model, bg=None):
    if model is None:
        return frames.copy()
    return Processor(model).process_batch(frames)

def algo_stat_k1(frames, model, bg=None):
    if bg is None:
        return frames.copy()
    analyzer = Analyzer(k_sigma=1.0)
    m = analyzer.analyze(bg)
    return Processor(m).process_batch(frames)

def algo_spatial(frames, model, bg=None):
    from scipy.ndimage import gaussian_filter
    out = frames.copy().reshape(frames.shape[0], ROWS, COLS)
    for i in range(out.shape[0]):
        out[i] = gaussian_filter(out[i], sigma=1.5, mode="reflect")
    return out.reshape(frames.shape)

def algo_hybrid(frames, model, bg=None):
    out1 = algo_stat_k1(frames, model, bg)
    from scipy.ndimage import gaussian_filter
    out = out1.reshape(out1.shape[0], ROWS, COLS)
    for i in range(out.shape[0]):
        out[i] = gaussian_filter(out[i], sigma=1.5, mode="reflect")
    return out.reshape(frames.shape)

def algo_wiener_k1(frames, model, bg=None):
    if bg is None:
        return frames.copy()
    wg = WienerGate(k_sigma=1.0)
    wg.fit(bg)
    return wg.process_batch(frames)

def algo_wiener_k15(frames, model, bg=None):
    if bg is None:
        return frames.copy()
    wg = WienerGate(k_sigma=1.5)
    wg.fit(bg)
    return wg.process_batch(frames)


ALGOS = [
    ("raw",            "Raw (no processing)",            algo_raw, False),
    ("StatGate_k3",    "StatGate k=3.0 (Stage1)",        algo_stat_k3, False),
    ("StatGate_k1",    "StatGate k=1.0",                  algo_stat_k1, True),
    ("Spatial",        "Spatial Gaussian sigma=1.5",      algo_spatial, False),
    ("Hybrid",         "Hybrid Stat+Space",               algo_hybrid, True),
    ("WienerGate_k1",  "WienerGate k=1.0 (WebRTC)",       algo_wiener_k1, True),
    ("WienerGate_k1.5","WienerGate k=1.5 (WebRTC)",       algo_wiener_k15, True),
]


def evaluate():
    csv_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".csv") and not f.startswith("._")
    ])
    if not csv_files:
        print("No CSV files found.")
        return

    print("=" * 120)
    print("  Full Algorithm Comparison — Signal + Additive White Noise Model (WebRTC NS)")
    print("  Metrics computed on SIGNAL frames only (excludes background frames)")
    print("=" * 120)

    all_summary = []

    for csv_name in csv_files:
        csv_path = os.path.join(DATA_DIR, csv_name)
        label = csv_name.replace(".csv", "")

        noisy_list = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            in_data = False
            for row in reader:
                if not row:
                    continue
                if row[0] == "##数据开始":
                    in_data = True
                    continue
                if in_data and len(row) > 2:
                    try:
                        noisy_list.append([float(v) for v in row[1:]])
                    except ValueError:
                        continue
        noisy = np.array(noisy_list, dtype=np.float64)

        gt_path = csv_path.replace(".csv", "_gt.npz")
        gt_all = np.load(gt_path)["ground_truth"]

        bg = noisy[:N_BG]
        sig_noisy = noisy[N_BG:]
        sig_gt = gt_all[N_BG:]

        try:
            analyzer = Analyzer(k_sigma=3.0)
            model = analyzer.analyze(bg)
        except Exception:
            model = None

        results = []
        for algo_key, algo_label, func, needs_bg in ALGOS:
            t0 = time.perf_counter()
            out_sig = func(sig_noisy, model, bg if needs_bg else None)
            elapsed = time.perf_counter() - t0

            rmse = m_rmse(out_sig, sig_gt)
            snr = m_snr(out_sig, sig_gt)
            bnsr = m_bnsr(out_sig, sig_gt, sig_noisy)
            srr = m_srr(out_sig, sig_gt)
            ssim = m_ssim(out_sig, sig_gt)
            cent = m_centroid(out_sig, sig_gt)
            cs = m_cs(snr, srr, ssim, cent)
            temp = m_temp(out_sig, sig_gt)
            area = m_area(out_sig, sig_gt)
            ms = elapsed / sig_noisy.shape[0] * 1000

            results.append((algo_key, rmse, snr, bnsr, srr, ssim, cent, cs, ms))

        print(f"\n{'─' * 120}")
        print(f"  {label}")
        bl_mean = bg.mean()
        ns_mean = np.std(bg, axis=0).mean()
        print(f"  bg_frames={N_BG}, sig_frames={sig_noisy.shape[0]}, "
              f"bg_baseline_mean={bl_mean:.1f}, noise_std_mean={ns_mean:.1f}")
        print(f"{'─' * 120}")
        print(f"{'Algo':<18s} {'RMSE':>8s} {'SNR':>8s} {'BNSR':>8s} {'SRR':>7s} "
              f"{'SSIM':>7s} {'Centr':>7s} {'CS':>7s} {'ms/fr':>7s}")
        print("-" * 95)
        for a, r, s, bn, sr, ss, ct, cs_v, ms in results:
            print(f"{a:<18s} {r:>8.2f} {s:>8.2f} {bn:>8.2f} {sr:>7.4f} "
                  f"{ss:>7.4f} {ct:>7.2f} {cs_v:>7.4f} {ms:>7.4f}")

        best = max(results, key=lambda x: x[7])
        print(f"  => Best: {best[0]} (CS={best[7]:.4f}, SNR={best[2]:.1f}dB, SRR={best[4]:.4f})")
        all_summary.append((label, best))

    print(f"\n{'=' * 120}")
    print("  SUMMARY — Best algorithm per dataset")
    print(f"{'=' * 120}")
    for label, best in all_summary:
        print(f"  {label:<30s} → {best[0]:<18s} CS={best[7]:.4f}, SNR={best[2]:.1f}dB, SRR={best[4]:.4f}")


if __name__ == "__main__":
    evaluate()
