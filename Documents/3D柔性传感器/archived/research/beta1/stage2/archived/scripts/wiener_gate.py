"""
wiener_gate.py — WebRTC-inspired per-channel Wiener-style soft-gain noise suppressor.

Based on WebRTC NoiseSuppressor architecture (see teaching doc), adapted from
spectral domain to per-channel spatial domain for tactile sensor arrays.

Core model:  x = baseline + signal + noise  (additive white noise)

Key WebRTC concepts adapted:
  [1] Analyze → Process two-stage: learn noise model from bg frames,
      then apply per-frame suppression.
  [2] Post-SNR: γ_post = max(residual² / σ² - 1, 0) matching |X|/N - 1
  [3] Decision-Directed Prior SNR: ξ = α*ξ_prev + (1-α)*γ_post
  [4] Wiener gain: G = ξ / (α_over + ξ)
  [5] Output: y = residual * G

Critical difference from StatGate (Stage1):
  StatGate:  if residual >= k*σ → keep; else → 0           (HARD gate)
  WienerGate: output = residual * G, G ∈ [0,1] continuous   (SOFT gate)

This preserves weak signals that StatGate clips, gives smooth
transition from noise suppression to signal passthrough.
"""
from __future__ import annotations

import numpy as np


class WienerGate:
    def __init__(self, k_sigma=1.0, over_sub=1.0, dd_alpha=0.98, min_snr=0.1):
        """
        Parameters
        ----------
        k_sigma : float
            Noise floor multiplier: noise_floor = baseline + k_sigma * noise_std.
            Default 1.0 (aggressive, good for well-modeled white noise).
            Higher values are more conservative (harder to gate).
        over_sub : float
            Over-subtraction factor for Wiener gain denominator.
            Default 1.0. Use 1.25 for more aggressive suppression.
        dd_alpha : float
            Decision-directed smoothing factor. 0.98 = heavily depend on
            prior frame → smooth, no jitter. 0.5 = responsive to changes.
        min_snr : float
            Minimum prior SNR floor to prevent complete signal loss.
        """
        self.k_sigma = k_sigma
        self.over_sub = over_sub
        self.dd_alpha = dd_alpha
        self.min_snr = min_snr

        self.baseline = None
        self.noise_std = None
        self.n_channels = 0
        self.prior_snr = None

    def fit(self, background_frames):
        """Learn per-channel noise model from pure-background frames."""
        N, C = background_frames.shape
        self.n_channels = C
        self.baseline = np.median(background_frames, axis=0)
        self.noise_std = np.std(background_frames, axis=0, ddof=1)
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
        output = residual * gain
        return output.astype(np.float64)

    def process_batch(self, frames):
        N = frames.shape[0]
        out = np.empty_like(frames, dtype=np.float64)
        for i in range(N):
            out[i] = self.process_frame(frames[i])
        return out


def evaluate_all():
    import os
    import numpy as np

    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_blob")
    ROWS, COLS = 64, 64

    csv_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".csv") and not f.startswith("._")
    ])
    if not csv_files:
        print(f"No CSV files in {DATA_DIR}.")
        return

    def names():
        return [
            ("raw", "Raw (no processing)"),
            ("Wiener_k1.0", "WienerGate k=1.0"),
            ("Wiener_k1.5", "WienerGate k=1.5"),
            ("Wiener_k2.0", "WienerGate k=2.0"),
            ("Wiener_k1.0_os1.25", "WienerGate k=1.0 os=1.25"),
        ]

    def metric_rmse(p, g):
        return float(np.sqrt(np.mean((p - g) ** 2)))

    def metric_snr(p, g):
        mse = np.mean((p - g) ** 2)
        vt = np.var(g)
        if mse < 1e-12 or vt < 1e-12:
            return 0.0
        return float(10 * np.log10(vt / mse))

    def metric_bnsr(p, g, src):
        mask = g < 1e-6
        if mask.sum() < 1:
            return 0.0
        vi = np.var(src[mask] - g[mask])
        vo = np.var(p[mask] - g[mask])
        if vo < 1e-12 or vi < 1e-12:
            return 0.0
        return float(10 * np.log10(vi / vo))

    def metric_srr(p, g):
        mask = g >= 1e-6
        if mask.sum() < 1:
            return 1.0
        pm = np.abs(p[mask]).mean()
        gm = np.abs(g[mask]).mean()
        return float(pm / gm) if gm > 1e-12 else 1.0

    def metric_ssim(p, g):
        N = p.shape[0]
        vals = []
        for i in range(N):
            p2 = p[i].reshape(ROWS, COLS)
            g2 = g[i].reshape(ROWS, COLS)
            mx, my = g2.mean(), p2.mean()
            vx, vy = g2.var(), p2.var()
            cxy = np.mean((g2 - mx) * (p2 - my))
            c1, c2 = 1e-4, 1e-4
            num = (2 * mx * my + c1) * (2 * cxy + c2)
            den = (mx**2 + my**2 + c1) * (vx + vy + c2)
            vals.append(float(num / den) if den > 1e-12 else 0.0)
        return float(np.mean(vals))

    def metric_centroid(p, g):
        shifts = []
        for i in range(p.shape[0]):
            p2 = p[i].reshape(ROWS, COLS)
            g2 = g[i].reshape(ROWS, COLS)
            ys, xs = np.mgrid[0:ROWS, 0:COLS]
            sp = p2.sum()
            sg = g2.sum()
            if sp < 1e-6 or sg < 1e-6:
                shifts.append(32.0)
            else:
                cxp = (xs * p2).sum() / sp
                cyp = (ys * p2).sum() / sp
                cxg = (xs * g2).sum() / sg
                cyg = (ys * g2).sum() / sg
                shifts.append(np.sqrt((cxp - cxg)**2 + (cyp - cyg)**2))
        return float(np.mean(shifts))

    def metric_cs(snr, srr, ssim, cs):
        sn = max(0, min(1, (snr + 10) / 50))
        cn = max(0, 1 - cs / 32)
        return 0.25 * sn + 0.25 * min(srr, 1) + 0.30 * max(ssim, 0) + 0.20 * cn

    print("=" * 110)
    print("  WienerGate Evaluation — Signal + Additive White Noise Model")
    print("  (bg frames used for noise model, sig frames for evaluation)")
    print("=" * 110)

    bg_frame_count = 15

    for csv_name in csv_files:
        csv_path = os.path.join(DATA_DIR, csv_name)
        label = csv_name.replace(".csv", "")

        import csv as csvmod
        noisy_data = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csvmod.reader(f)
            in_data = False
            for row in reader:
                if not row:
                    continue
                if row[0] == "##数据开始":
                    in_data = True
                    continue
                if in_data and len(row) > 2:
                    try:
                        noisy_data.append([float(v) for v in row[1:]])
                    except ValueError:
                        continue
        noisy = np.array(noisy_data, dtype=np.float64)

        gt_path = csv_path.replace(".csv", "_gt.npz")
        gt = np.load(gt_path)["ground_truth"]

        N, C = noisy.shape
        bg = noisy[:bg_frame_count]
        sig = noisy[bg_frame_count:]
        sig_gt = gt[bg_frame_count:]

        results = []
        for algo_key, algo_label in names():
            if algo_key == "raw":
                out_sig = sig.copy()
                out_all = noisy.copy()
            else:
                if "k1.5" in algo_key and "os" not in algo_key:
                    wg = WienerGate(k_sigma=1.5)
                elif "k2.0" in algo_key and "os" not in algo_key:
                    wg = WienerGate(k_sigma=2.0)
                elif "os1.25" in algo_key:
                    wg = WienerGate(k_sigma=1.0, over_sub=1.25)
                else:
                    wg = WienerGate(k_sigma=1.0)
                wg.fit(bg)
                out_sig = wg.process_batch(sig)
                out_all = np.concatenate([bg, out_sig], axis=0)

            rmse = metric_rmse(out_all, gt)
            snr = metric_snr(out_all, gt)
            bnsr = metric_bnsr(out_all, gt, noisy)
            srr_v = metric_srr(out_all, gt)
            ssim = metric_ssim(out_all, gt)
            cent = metric_centroid(out_all, gt)
            cs = metric_cs(snr, srr_v, ssim, cent)

            results.append((algo_key, rmse, snr, bnsr, srr_v, ssim, cent, cs))

        print(f"\n{'─' * 110}")
        print(f"  {label}")
        print(f"  N={N}, bg_baseline_mean={bg.mean():.1f}, noise_std_mean={np.std(bg,axis=0).mean():.1f}")
        print(f"{'─' * 110}")
        print(f"{'Algo':<24s} {'RMSE':>8s} {'SNR':>8s} {'BNSR':>8s} {'SRR':>7s} {'SSIM':>7s} {'Centr':>7s} {'CS':>7s}")
        print("-" * 90)
        for a, r, s, bn, sr, ss, ct, cs_v in results:
            print(f"{a:<24s} {r:>8.2f} {s:>8.2f} {bn:>8.2f} {sr:>7.4f} {ss:>7.4f} {ct:>7.2f} {cs_v:>7.4f}")

        best = max(results, key=lambda x: x[7])
        print(f"  => Best: {best[0]} (CS={best[7]:.4f}, SNR={best[2]:.1f}dB, SRR={best[4]:.4f})")


if __name__ == "__main__":
    evaluate_all()
