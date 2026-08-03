import json
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import invres, tf2ss, lsim
from scipy.signal import StateSpace as ScipySS
from scipy.integrate import cumulative_trapezoid

from source.classes import vectfit as _vf
from source.classes.Plotter import Plotter

"""
# -------------------------------------------------------------------------
# Name:            Linear_si.py
# Description:     System identification scheme for WEC.
#                  Identifies two transfer functions from CFD signal data:
#                    - eta2f : wave elevation  → excitation force
#                    - f2v   : excitation force → velocity
#                  Both are identified via Vector Fitting (frequency domain).
#                  f2v can optionally use N4SID (time domain).
#                  The cascade eta2f ∘ f2v gives a full η→ẋ surrogate.
#
# Identification pipeline (per TF):
#   1. Load input/output signals from DataHandle columns
#   2. Compute ETFE (FFT ratio averaged over cases)
#   3. VF with causality sweep over order_list × tau_trials
#   4. Pick best (order, tau) by MSE — build state-space
#
# Author:          Antoine
# Date created:    07/2026
# Project:         wec_modeling_benchmark
# -------------------------------------------------------------------------
"""

log = logging.getLogger(__name__)

# ── scheme flag ───────────────────────────────────────────────────────────────

SELF_LOADING = False  # DDFeed loads DataHandle objects via load_training_data


# ── helpers ───────────────────────────────────────────────────────────────────


def _compute_fft(signal, dt):
    """One-sided complex FFT with peak normalisation."""
    N = len(signal)
    X = np.fft.rfft(signal) * 2 / N
    X[0] /= 2
    if N % 2 == 0:
        X[-1] /= 2
    w = 2 * np.pi * np.fft.rfftfreq(N, dt)
    return w, X


def _smooth(amp, window):
    if window <= 1:
        return amp
    kernel = np.ones(window) / window
    return np.convolve(amp, kernel, mode="same")


def _to_dB(x):
    return 20 * np.log10(np.maximum(x, 1e-30))


def _extract_signal(case, signal_list):
    """Sum multiple DataHandle columns into one signal."""
    t = case.dataset["t"].to_numpy()
    sig = np.zeros(len(t))
    for col in signal_list:
        if col in case.dataset.columns:
            sig += case.dataset[col].to_numpy()
        else:
            log.warning("Column '%s' not in case %s — skipping", col, case.label)
    return t, sig


def _build_ss(num, den):
    """Convert TF coefficients to real state-space matrices."""
    return [np.real(m) for m in tf2ss(num, den)]


def _cascade_ss(A_e, B_e, C_e, A_f, B_f, C_f):
    """
    Series connection: sys1 (eta2f) followed by sys2 (f2v), both strictly proper.
    State z = [x_e; x_f]:
        dz/dt = [A_e,      0 ] z + [B_e] η
                [B_f C_e,  A_f]     [0  ]
        ẋ     = [0,  C_f] z
    """
    n_e = A_e.shape[0]
    n_f = A_f.shape[0]
    A = np.block([[A_e, np.zeros((n_e, n_f))], [B_f @ C_e, A_f]])
    B = np.vstack([B_e, np.zeros((n_f, 1))])
    C = np.hstack([np.zeros((1, n_e)), C_f])
    D = np.zeros((1, 1))
    return A, B, C, D


# ── ETFE computation ──────────────────────────────────────────────────────────


def _compute_etfe(cases, in_cols, out_cols, dt, smooth_window=1):
    """
    Compute average ETFE G(jω) = Ẑ_out / Ẑ_in across all cases.
    Smoothing applied to amplitudes separately before division.
    Returns (w, G_complex, per_case_list).
    """
    G_list = []
    w_ref = None
    for case in cases:
        _, u = _extract_signal(case, in_cols)
        _, y = _extract_signal(case, out_cols)
        w, U = _compute_fft(u, dt)
        _, Y = _compute_fft(y, dt)
        amp_U = _smooth(np.abs(U), smooth_window)
        amp_Y = _smooth(np.abs(Y), smooth_window)
        G_amp = amp_Y / (amp_U + 1e-30)
        G_ph = np.angle(Y) - np.angle(U)
        G_list.append(G_amp * np.exp(1j * G_ph))
        w_ref = w
    G_avg = np.mean(np.array(G_list), axis=0)
    return w_ref, G_avg, G_list


# ── VF identification with causality sweep ────────────────────────────────────


def _vf_causality_sweep(G_etfe, w, order_list, tau_trials):
    """
    For each (order, tau): shift G by e^{-jwτ}, fit VF, compute error.
    Returns results dict keyed by order.
    """
    results = {}
    s = 1j * w
    for ord_i in order_list:
        n_poles = int(ord_i / 2)
        if n_poles < 1:
            continue
        errors_i, fits_i = [], []
        for tau in tau_trials:
            G_shifted = G_etfe * np.exp(-1j * w * tau)
            try:
                p_i, r_i, d_i, _ = _vf.vectfit_auto(
                    G_shifted, s, n_poles=n_poles, n_iter=50
                )
                G_fit_i = _vf.model(s, p_i, r_i, d_i, h=0) * np.exp(1j * w * tau)
                err_i = float(np.mean(np.abs(G_fit_i - G_etfe) ** 2))
            except Exception as e:
                log.debug("VF order=%d tau=%.2f: %s", ord_i, tau, e)
                err_i = np.inf
                p_i, r_i, d_i = np.array([]), np.array([]), 0.0
            errors_i.append(err_i)
            fits_i.append((p_i, r_i, d_i, tau))
        best_i = int(np.argmin(errors_i))
        results[ord_i] = dict(
            errors=errors_i, fits=fits_i, best_idx=best_i, best_tau=fits_i[best_i][3]
        )
        log.info(
            "  order=%2d  best_tau=%.2fs  err=%.4e",
            ord_i,
            fits_i[best_i][3],
            errors_i[best_i],
        )
    return results


def _pick_best_and_build_ss(vf_results):
    """Pick global best (order, tau), convert poles/residues to SS."""
    best_order = min(
        vf_results, key=lambda o: vf_results[o]["errors"][vf_results[o]["best_idx"]]
    )
    r = vf_results[best_order]
    p, res, d, tau = (*r["fits"][r["best_idx"]][:3], r["best_tau"])
    num, den = invres(res, p, [d], tol=1e-8, rtype="avg")
    A, B, C, D = _build_ss(num, den)
    D = np.zeros_like(D)  # strictly proper
    return best_order, tau, A, B, C, D, p, res, d


# ── N4SID identification (f2v, time domain) ───────────────────────────────────


def _n4sid_identification(cases, in_cols, out_cols, order_list, dt, num_block_rows=40):
    """
    N4SID from concatenated time-domain I/O.
    Returns results dict keyed by order (compatible with VF result format).
    """
    try:
        import pandas as pd
        from nfoursid.nfoursid import NFourSID
        import control
        from scipy.linalg import logm
    except ImportError as e:
        log.error("N4SID deps missing: %s  (pip install nfoursid control)", e)
        return {}

    u_all, y_all = [], []
    for case in cases:
        _, u = _extract_signal(case, in_cols)
        _, y = _extract_signal(case, out_cols)
        u_all.append(u)
        y_all.append(y)
    u_data = np.concatenate(u_all)
    y_data = np.concatenate(y_all)

    df = pd.DataFrame({"y": y_data, "u": u_data})
    results = {}

    for ord_i in order_list:
        try:
            n4 = NFourSID(
                df,
                output_columns=["y"],
                input_columns=["u"],
                num_block_rows=num_block_rows,
            )
            n4.subspace_identification()
            ss_n4, _ = n4.system_identification(rank=ord_i)

            A_d = np.real(ss_n4.a)
            B_d = np.real(ss_n4.b)
            n = A_d.shape[0]
            A_c = np.real(logm(A_d) / dt)
            B_c = np.real(np.linalg.solve(A_d - np.eye(n), A_c @ B_d))
            C_c = np.real(ss_n4.c)
            D_c = np.zeros((1, 1))

            # placeholder error
            results[ord_i] = dict(
                errors=[0.0],
                fits=[(None, None, 0.0, 0.0)],
                best_idx=0,
                best_tau=0.0,
                _ss=(A_c, B_c, C_c, D_c),
            )
            log.info("  N4SID order=%d OK", ord_i)
        except Exception as e:
            log.warning("  N4SID order=%d failed: %s", ord_i, e)
    return results


# ── prep ─────────────────────────────────────────────────────────────────────


def prep(feed):
    """DataHandle objects already loaded by DDFeed — nothing extra needed."""
    pass


# ── run ───────────────────────────────────────────────────────────────────────


def run(opt, feed, config):
    sp = config.scheme_params
    e_c = sp.get("eta2f", {})
    f_c = sp.get("f2v", {})
    dt = config.solver.get("dt", 0.05)

    log.info("Linear_si: %d training case(s)", len(feed.data))

    # ── eta2f ETFE ────────────────────────────────────────────────────────────
    log.info("eta2f ETFE...")
    w_e, G_e, G_e_list = _compute_etfe(
        feed.data,
        e_c.get("input_signals", ["eta"]),
        e_c.get("output_signals", ["fe_lin"]),
        dt,
        smooth_window=e_c.get("smooth_window", 1),
    )
    msk_e = (w_e >= e_c.get("w_rel_start", 0.3)) & (w_e <= e_c.get("w_rel_end", 4.0))
    w_e_rel = w_e[msk_e]
    G_e_rel = G_e[msk_e]

    tau_trials_e = np.linspace(
        e_c.get("tau_min", 0.0), e_c.get("tau_max", 20.0), max(1, e_c.get("n_tau", 40))
    )
    order_list_e = e_c.get("order_list", [2, 4, 6, 8, 10])
    log.info(
        "eta2f VF: orders=%s  tau=[%.1f,%.1f] n=%d",
        order_list_e,
        tau_trials_e[0],
        tau_trials_e[-1],
        len(tau_trials_e),
    )
    eta2f_vf = _vf_causality_sweep(G_e_rel, w_e_rel, order_list_e, tau_trials_e)
    best_ord_e, tau_e, A_e, B_e, C_e, D_e, p_e, r_e, d_e = _pick_best_and_build_ss(
        eta2f_vf
    )
    log.info("eta2f best: order=%d  tau=%.3fs", best_ord_e, tau_e)

    # ── f2v ETFE / identification ─────────────────────────────────────────────
    log.info("f2v ETFE...")
    w_f, G_f, G_f_list = _compute_etfe(
        feed.data,
        f_c.get("input_signals", ["fe_lin"]),
        f_c.get("output_signals", ["xdot"]),
        dt,
        smooth_window=f_c.get("smooth_window", 20),
    )
    msk_f = (w_f >= f_c.get("w_rel_start", 0.8)) & (w_f <= f_c.get("w_rel_end", 2.5))
    w_f_rel = w_f[msk_f]
    G_f_rel = G_f[msk_f]

    order_list_f = f_c.get("order_list", [2, 4, 6, 8, 10])
    id_method = f_c.get("identification", "frequency")
    tau_trials_f = np.linspace(
        f_c.get("tau_min", 0.0), f_c.get("tau_max", 0.0), max(1, f_c.get("n_tau", 1))
    )

    if id_method == "time":
        log.info("f2v N4SID (time domain)")
        f2v_raw = _n4sid_identification(
            feed.data,
            f_c.get("input_signals", ["fe_lin"]),
            f_c.get("output_signals", ["xdot"]),
            order_list_f,
            dt,
            num_block_rows=f_c.get("num_block_rows", 40),
        )
        # pick best order by last available (no error comparison for N4SID)
        best_ord_f = order_list_f[-1] if f2v_raw else order_list_f[0]
        if best_ord_f in f2v_raw and "_ss" in f2v_raw[best_ord_f]:
            A_f, B_f, C_f, D_f = f2v_raw[best_ord_f]["_ss"]
            tau_f = 0.0
            p_f, r_f, d_f = None, None, 0.0
        else:
            A_f = B_f = C_f = D_f = np.zeros((1, 1))
            tau_f = 0.0
            p_f = r_f = None
            d_f = 0.0
        f2v_vf = f2v_raw
    else:
        log.info(
            "f2v VF: orders=%s  tau=[%.1f,%.1f] n=%d",
            order_list_f,
            tau_trials_f[0],
            tau_trials_f[-1],
            len(tau_trials_f),
        )
        f2v_vf = _vf_causality_sweep(G_f_rel, w_f_rel, order_list_f, tau_trials_f)
        best_ord_f, tau_f, A_f, B_f, C_f, D_f, p_f, r_f, d_f = _pick_best_and_build_ss(
            f2v_vf
        )
    log.info("f2v best: order=%d  tau=%.3fs  method=%s", best_ord_f, tau_f, id_method)

    # ── cascade ───────────────────────────────────────────────────────────────
    A_c, B_c, C_c, D_c = _cascade_ss(A_e, B_e, C_e, A_f, B_f, C_f)

    # ── build result ──────────────────────────────────────────────────────────
    result = {
        "eta2f": {
            "best_order": int(best_ord_e),
            "best_tau": float(tau_e),
            "A": A_e.tolist(),
            "B": B_e.tolist(),
            "C": C_e.tolist(),
            "D": D_e.tolist(),
        },
        "f2v": {
            "best_order": int(best_ord_f),
            "best_tau": float(tau_f),
            "method": id_method,
            "A": A_f.tolist(),
            "B": B_f.tolist(),
            "C": C_f.tolist(),
            "D": D_f.tolist(),
        },
        "cascade": {
            "tau": float(tau_e),
            "A": A_c.tolist(),
            "B": B_c.tolist(),
            "C": C_c.tolist(),
            "D": D_c.tolist(),
        },
    }

    # grids_by_method carries numpy data for plots (not serialized)
    grids_by_method = {
        "eta2f": {
            "w": w_e,
            "G": G_e,
            "w_rel": w_e_rel,
            "G_rel": G_e_rel,
            "vf_results": eta2f_vf,
            "tau_trials": tau_trials_e,
            "order_list": order_list_e,
            "best_order": best_ord_e,
            "best_tau": tau_e,
            "A": A_e,
            "B": B_e,
            "C": C_e,
            "p": p_e,
            "r": r_e,
            "d": d_e,
            "w_min": e_c.get("w_min", 0.3),
            "w_max": e_c.get("w_max", 5.0),
            "w_rel_start": e_c.get("w_rel_start", 0.3),
            "w_rel_end": e_c.get("w_rel_end", 4.0),
        },
        "f2v": {
            "w": w_f,
            "G": G_f,
            "w_rel": w_f_rel,
            "G_rel": G_f_rel,
            "vf_results": f2v_vf,
            "tau_trials": tau_trials_f,
            "order_list": order_list_f,
            "best_order": best_ord_f,
            "best_tau": tau_f,
            "A": A_f,
            "B": B_f,
            "C": C_f,
            "p": p_f,
            "r": r_f,
            "d": d_f,
            "method": id_method,
            "w_min": f_c.get("w_min", 0.05),
            "w_max": f_c.get("w_max", 7.0),
            "w_rel_start": f_c.get("w_rel_start", 0.8),
            "w_rel_end": f_c.get("w_rel_end", 2.5),
        },
        "cascade": {
            "A": A_c,
            "B": B_c,
            "C": C_c,
            "D": D_c,
            "tau": tau_e,
        },
    }
    simruns_by_method = {}
    return result, grids_by_method, simruns_by_method


# ── save ─────────────────────────────────────────────────────────────────────


def save(opt, result, output_name):
    """Save model.json with all three SS models (eta2f, f2v, cascade)."""
    from source.config import MODEL_DIR

    out_dir = MODEL_DIR / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = {
        "type": "linear_si",
        "name": output_name,
        "eta2f": result["eta2f"],
        "f2v": result["f2v"],
        "cascade": result["cascade"],
    }
    path = out_dir / "model.json"
    with open(path, "w") as f:
        json.dump(model, f, indent=2)
    log.info("model.json saved: %s", path)


def save_result(result, id_dir):
    pass  # Optimizer.save_result handles result.json


def log_result(result):
    log.info("Linear SI results:")
    log.info(
        "  eta2f: order=%d  tau=%.3fs",
        result["eta2f"]["best_order"],
        result["eta2f"]["best_tau"],
    )
    log.info(
        "  f2v:   order=%d  tau=%.3fs  method=%s",
        result["f2v"]["best_order"],
        result["f2v"]["best_tau"],
        result["f2v"]["method"],
    )
    log.info(
        "  cascade: A size=%d×%d",
        len(result["cascade"]["A"]),
        len(result["cascade"]["A"][0]),
    )


def save_data(result, simruns_by_method, grids_by_method, feed, data_dir):
    """Save ETFE data as CSV for each TF."""
    import pandas as pd

    for tf_name in ("eta2f", "f2v"):
        g = grids_by_method.get(tf_name, {})
        w_rel = g.get("w_rel")
        G_rel = g.get("G_rel")
        if w_rel is None:
            continue
        pd.DataFrame(
            {
                "w_rad_s": w_rel,
                "G_amp_dB": _to_dB(np.abs(G_rel)),
                "G_phase_deg": np.degrees(np.unwrap(np.angle(G_rel))),
            }
        ).to_csv(data_dir / f"etfe_{tf_name}.csv", index=False)
    log.info("ETFE CSVs saved to %s", data_dir)


# ── plots ─────────────────────────────────────────────────────────────────────


def _plot_error_vs_delay(g, title_prefix, save_path):
    vf_results = g["vf_results"]
    tau_trials = g["tau_trials"]
    order_list = g["order_list"]
    best_order = g["best_order"]
    best_tau = g["best_tau"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(order_list)))

    fig, ax = plt.subplots(figsize=(10, 5))
    for col, ord_i in zip(colors, order_list):
        if ord_i not in vf_results:
            continue
        r = vf_results[ord_i]
        ax.semilogy(
            tau_trials,
            r["errors"],
            "o-",
            ms=4,
            color=col,
            lw=1.5 if ord_i == best_order else 0.8,
            label=f"order {ord_i}" + (" ← BEST" if ord_i == best_order else ""),
        )
        ax.axvline(r["best_tau"], color=col, ls="--", lw=0.8, alpha=0.5)
    ax.set(
        xlabel="τ [s]",
        ylabel="MSE",
        title=f"{title_prefix} — error vs delay  |  best: order={best_order}  τ={best_tau:.2f}s",
    )
    ax.legend(fontsize=7)
    ax.grid(True, which="both")
    plt.tight_layout()
    Plotter._save(
        fig, save_path, None, f"error_vs_delay_{title_prefix.lower().replace(' ', '_')}"
    )


def _plot_bode(g, hs, title_prefix, save_path):
    """Bode plot: ETFE, BEM (optional), all VF fits, best highlighted."""
    vf_results = g["vf_results"]
    order_list = g["order_list"]
    best_order = g["best_order"]
    best_tau = g["best_tau"]
    w_rel = g["w_rel"]
    G_rel = g["G_rel"]
    w_min = g["w_min"]
    w_max = g["w_max"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(order_list)))

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ETFE
    ax[0].plot(w_rel, _to_dB(np.abs(G_rel)), color="red", lw=2, label="ETFE", zorder=8)
    ax[1].plot(
        w_rel, np.degrees(np.unwrap(np.angle(G_rel))), color="red", lw=2, zorder=8
    )

    # BEM reference (eta2f only, from HydroSphere)
    if hs is not None and title_prefix.lower().startswith("eta2f"):
        ax[0].plot(hs.w, _to_dB(hs.Fe_mod), color="black", lw=2, label="BEM", zorder=10)
        ax[1].plot(
            hs.w, np.degrees(np.unwrap(hs.Fe_ang)), color="black", lw=2, zorder=10
        )

    # VF fits
    for col, ord_i in zip(colors, order_list):
        if ord_i not in vf_results:
            continue
        r = vf_results[ord_i]
        p_b, res_b, d_b, tau_b = (*r["fits"][r["best_idx"]][:3], r["best_tau"])
        if p_b is None:
            continue
        s_plot = 1j * w_rel
        G_fit = _vf.model(s_plot, p_b, res_b, d_b, h=0) * np.exp(1j * w_rel * tau_b)
        is_best = ord_i == best_order
        lbl = f"VF order {ord_i}  τ={tau_b:.1f}s" + ("  ← BEST" if is_best else "")
        lw = 2.0 if is_best else 0.9
        ax[0].plot(w_rel, _to_dB(np.abs(G_fit)), color=col, lw=lw, ls="--", label=lbl)
        ax[1].plot(
            w_rel, np.degrees(np.unwrap(np.angle(G_fit))), color=col, lw=lw, ls="--"
        )

    for a in ax:
        a.axvline(g["w_rel_start"], color="grey", ls=":", lw=0.8)
        a.axvline(g["w_rel_end"], color="grey", ls=":", lw=0.8)
        a.set_xscale("log")
        a.set_xlim(w_min, w_max)
        a.grid(True, which="both")
    ax[0].set(
        ylabel="|G| [dB]",
        title=f"{title_prefix}  |  BEST: order={best_order}  τ={best_tau:.2f}s",
    )
    ax[0].legend(fontsize=7, ncol=2)
    ax[1].set(ylabel="Phase [°]", xlabel="ω [rad/s]")
    plt.tight_layout()
    Plotter._save(
        fig, save_path, None, f"bode_{title_prefix.lower().replace(' ', '_')}"
    )


def _plot_pzmap(g, title_prefix, save_path):
    """Pole-zero map for the best identified model."""
    p_b = g.get("p")
    res_b = g.get("r")
    d_b = g.get("d", 0.0)
    if p_b is None:
        return
    try:
        num, den = invres(res_b, p_b, [d_b], tol=1e-8, rtype="avg")
        zeros = np.roots(num)
        all_stable = np.all(p_b.real < 0)
        min_phase = np.all(zeros.real < 0)
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.scatter(
        p_b.real,
        p_b.imag,
        marker="x",
        s=120,
        color="C1",
        zorder=5,
        label=f"poles ({len(p_b)})  {'stable ✓' if all_stable else 'UNSTABLE ✗'}",
    )
    ax.scatter(
        zeros.real,
        zeros.imag,
        marker="o",
        s=80,
        facecolors="none",
        edgecolors="C0",
        zorder=5,
        label=f"zeros ({len(zeros)})  {'min phase ✓' if min_phase else 'non-min phase ✗'}",
    )
    ax.set(
        xlabel="Re",
        ylabel="Im",
        title=f"Pole-zero map — {title_prefix}  order={g['best_order']}  τ={g['best_tau']:.2f}s",
    )
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    Plotter._save(
        fig, save_path, None, f"pzmap_{title_prefix.lower().replace(' ', '_')}"
    )


# ── PLOT_DISPATCH ─────────────────────────────────────────────────────────────


def plot_error_vs_delay(data, simruns, result, grids, save_path):
    g_e = grids.get("eta2f", {})
    g_f = grids.get("f2v", {})
    if g_e.get("vf_results"):
        _plot_error_vs_delay(g_e, "eta2f", save_path)
    if g_f.get("vf_results") and g_f.get("method", "frequency") == "frequency":
        _plot_error_vs_delay(g_f, "f2v", save_path)


def plot_bode(data, simruns, result, grids, save_path):
    # retrieve HydroSphere from the Optimizer via opt if available
    hs = None
    _plot_bode(grids.get("eta2f", {}), hs, "eta2f", save_path)
    _plot_bode(grids.get("f2v", {}), hs, "f2v", save_path)


def plot_bode_with_hs(hs):
    """Return a plot_bode function that has HydroSphere for BEM overlay."""

    def _fn(data, simruns, result, grids, save_path):
        _plot_bode(grids.get("eta2f", {}), hs, "eta2f", save_path)
        _plot_bode(grids.get("f2v", {}), hs, "f2v", save_path)

    return _fn


def plot_pzmap(data, simruns, result, grids, save_path):
    _plot_pzmap(grids.get("eta2f", {}), "eta2f", save_path)
    _plot_pzmap(grids.get("f2v", {}), "f2v", save_path)


def plot_etfe(data, simruns, result, grids, save_path):
    """Four-panel ETFE plot for eta2f and f2v."""
    for tf_name, label in [("eta2f", "η → Fe"), ("f2v", "Fe → ẋ")]:
        g = grids.get(tf_name, {})
        w = g.get("w")
        G = g.get("G")
        w_r = g.get("w_rel")
        G_r = g.get("G_rel")
        if w is None:
            continue
        w_min = g["w_min"]
        w_max = g["w_max"]
        msk_w = (w >= w_min) & (w <= w_max)

        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.suptitle(f"ETFE — {label}")
        ax[0].plot(
            w[msk_w],
            _to_dB(np.abs(G[msk_w])),
            lw=0.8,
            alpha=0.5,
            color="C0",
            label="broad",
        )
        ax[0].plot(w_r, _to_dB(np.abs(G_r)), lw=1.5, color="C0", label="relevant")
        ax[0].set(ylabel="|G| [dB]")
        ax[0].legend(fontsize=7)
        ax[0].grid(True, which="both")
        ax[1].plot(
            w[msk_w],
            np.degrees(np.unwrap(np.angle(G[msk_w]))),
            lw=0.8,
            alpha=0.5,
            color="C1",
        )
        ax[1].plot(w_r, np.degrees(np.unwrap(np.angle(G_r))), lw=1.5, color="C1")
        ax[1].set(ylabel="Phase [°]", xlabel="ω [rad/s]")
        ax[1].grid(True, which="both")
        for a in ax:
            a.axvline(g["w_rel_start"], color="red", ls="--", lw=0.8)
            a.axvline(g["w_rel_end"], color="red", ls="--", lw=0.8)
            a.set_xscale("log")
            a.set_xlim(w_min, w_max)
        plt.tight_layout()
        Plotter._save(fig, save_path, None, f"etfe_{tf_name}")


PLOT_DISPATCH = {
    "etfe": plot_etfe,
    "error_vs_delay": plot_error_vs_delay,
    "bode": plot_bode,
    "pzmap": plot_pzmap,
}
