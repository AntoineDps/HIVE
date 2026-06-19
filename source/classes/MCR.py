import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable
import inspect
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from joblib import Parallel, delayed

from source.package_1.classes.Wave import Wave
from source.package_1.classes.HydroSphere import HydroSphere
from source.package_1.classes.IntegratorPhyLinear1DOF import IntegratorPhyLinear1DOF
from source.package_1.classes.IntegratorNLinear1DOF import IntegratorNLinear1DOF


# TODO : X0 np array iterable problem
# TODO : improve progression bar unaccurate for parrallelisation
# TODO : add a method to do MCR without parrallelisation
# TODO : implement for nonlinear wec


class MCR:
    @staticmethod
    def extract_params(all_params, sweep_keys, combo):
        """Merge fixed params with one sweep combination"""
        p = all_params.copy()
        p.update(dict(zip(sweep_keys, combo)))
        return p

    @staticmethod
    def get_attr_by_path(obj_map, path):
        """
        path = 'wec.mean_power'
        """
        obj_name, attr = path.split(".", 1)
        return getattr(obj_map[obj_name], attr)

    @staticmethod
    def is_sweepable(x):
        return isinstance(x, Iterable) and not isinstance(x, (str, bytes))

    def __init__(self, wave_param, body_param, wec_param, store=None):
        """
        param_grid : dict
            Keys are in the form "object.attribute" (e.g., "wave.T").
            Values are lists of values to sweep.
        store : list of str
            List of attributes to store in log (e.g., ["wave.J", "body.ma"])
        """
        self.wave_param = wave_param
        self.body_param = body_param
        self.wec_param = wec_param

        self.log = []
        self.store = store if store is not None else []

    def run(self):
        def split_params(params):  #! speciale X0
            sweep, fixed = {}, {}
            for k, v in params.items():
                if self.is_sweepable(v) and k != "X0":
                    sweep[k] = list(v)
                else:
                    fixed[k] = v
            return sweep, fixed

        def _ncomb(vals):
            return np.prod([len(v) for v in vals]) if vals else 1

        # Split parameters in sweep and fixed
        wave_sweep, wave_fixed = split_params(self.wave_param)
        body_sweep, body_fixed = split_params(self.body_param)
        wec_sweep, wec_fixed = split_params(self.wec_param)

        # extract sweep key and values
        wave_keys, wave_vals = zip(*wave_sweep.items()) if wave_sweep else ([], [])
        body_keys, body_vals = zip(*body_sweep.items()) if body_sweep else ([], [])
        wec_keys, wec_vals = zip(*wec_sweep.items()) if wec_sweep else ([], [])

        # count
        n_wave = _ncomb(wave_vals)
        n_body = _ncomb(body_vals)
        n_wec = _ncomb(wec_vals)
        n_total = n_wave * n_body * n_wec

        # progression initiate
        pbar = tqdm(total=n_total, desc="MCR run")

        # --- Helper function for parallel execution ---
        def process_wec(cv, wave, body, wv, bv):
            wec_param_all = self.extract_params(wec_fixed, wec_keys, cv)

            # remove uncorrect param
            wec_param = {
                k: v for k, v in wec_param_all.items() if k not in {"integrator", "dyn"}
            }

            if wec_param_all["integrator"] == "linear":
                wec = IntegratorPhyLinear1DOF(
                    m=body.mass,
                    ma_inf=body.ma_inf,
                    rad_ss=body.rad_ss,
                    S=body.stiffness,
                    **wec_param,
                )

                wec.build_fe(wave.eta, body.t_irf, body.fe_irf)
                F = wec.fe_hist

            elif wec_param_all["integrator"] == "nlinear":
                wec = IntegratorNLinear1DOF(
                    m=body.mass,  # ? dt
                    r=body.r,
                    ma_inf=body.ma_inf,
                    rad_ss=body.rad_ss,
                    **wec_param,
                )

                wec.build_fe(wave.eta, body.t_irf, body.fe_irf)
                F = wec.fhd_hist

            else:
                raise KeyError("Integrator type must be 'linear' or 'nlinear'")

            # Forces system
            dyn_fun = getattr(wec, wec_param_all["dyn"])

            unstable = False
            for f in F:
                X, t_i = wec.RK4_step(dyn_fun, f)

                # check instability
                if abs(X[0] - wec.eta_for_hs[-1]) > 1.5 * body.r:
                    unstable = True
                    break

            if wec_param_all["integrator"] == "linear":
                if unstable:
                    # wec.post_process()
                    wec.mean_power = np.Nan
                else:
                    wec.post_process()
            elif wec_param_all["integrator"] == "nlinear":
                if unstable:
                    # wec.post_process(True)
                    wec.mean_power = -np.inf
                else:
                    wec.post_process(True)

            # Build log entry
            entry = {}
            for k, v in zip(wave_keys, wv):
                entry[k] = v
            for k, v in zip(body_keys, bv):
                entry[k] = v
            for k, v in zip(wec_keys, cv):
                entry[k] = v

            obj_map = {"wave": wave, "body": body, "wec": wec}
            for s in self.store:
                entry[s.split(".", 1)[1]] = self.get_attr_by_path(obj_map, s)

            return entry

        # --- Main Execution Loops ---
        for wv in itertools.product(*wave_vals) if wave_vals else [()]:
            wave_param = self.extract_params(wave_fixed, wave_keys, wv)
            wave = Wave(**wave_param)

            for bv in itertools.product(*body_vals) if body_vals else [()]:
                body_param = self.extract_params(body_fixed, body_keys, bv)
                body = HydroSphere(**body_param)

                # Parallelize the 'cv' loop
                wec_combinations = (
                    list(itertools.product(*wec_vals)) if wec_vals else [()]
                )

                results = Parallel(n_jobs=-1)(  # n_jobs=-1 uses all available cores
                    delayed(process_wec)(cv, wave, body, wv, bv)
                    for cv in wec_combinations
                )

                # Collect results and update progress
                self.log.extend(results)
                pbar.update(len(results))

        pbar.close()

    def save_log_csv(self, path):
        if not self.log:
            raise RuntimeError("Log is empty. Run the sweep first.")

        df = pd.DataFrame(self.log)
        df.to_csv(path, index=False)

    def PI_map(self, output="mean_power", save_path="PI_map"):
        """
        Extract the best [B_pto, S_pto] for each (T, H) from the sweep log.

        Parameters
        ----------
        output : str
            The log key to optimize (e.g., "mean_power")
        save_csv_path : str or None
            If given, saves the resulting table as a CSV with (B,S) tuples in cells

        Returns
        -------
        B_map, S_map : pd.DataFrame
            DataFrames with T as index, H as columns, cells = best B_pto / S_pto
        """

        if not self.log:
            raise RuntimeError("Log is empty. Run the sweep first.")

        df = pd.DataFrame(self.log)
        cols = df.columns
        cols_PI = {"T", "H", "B_pto", "S_pto", "mean_power"}

        # check required columns
        if not cols_PI.issubset(cols):
            raise KeyError(
                "Log must contain 'T', 'H', 'B_pto', 'S_pto' and 'mean_power' columns."
            )

        # define grid
        T = sorted(df["T"].unique())
        H = sorted(df["H"].unique())

        B_pto_map = pd.DataFrame(index=H, columns=T)
        S_pto_map = pd.DataFrame(index=H, columns=T)

        rows_PI = []

        # fill grid
        for t in T:  #! choose B and S matrix or both combined
            for h in H:
                df_sub = df.copy()
                df_sub = df_sub[df_sub["T"] == t]
                df_sub = df_sub[df_sub["H"] == h]

                idx_max = df_sub[output].idxmax()
                best_row = df_sub.loc[idx_max]

                B_pto_map.loc[h, t] = best_row["B_pto"]
                S_pto_map.loc[h, t] = best_row["S_pto"]

                rows_PI.append(
                    {
                        "T": t,
                        "H": h,
                        "B_pto": best_row["B_pto"],
                        "S_pto": best_row["S_pto"],
                    }
                )

        # save CSV if requested
        B_pto_map.to_csv(save_path + "_Kp.csv")
        S_pto_map.to_csv(save_path + "_Ki.csv")

        # scatter plot
        df_plot = pd.DataFrame(rows_PI)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Kp subplot
        sc1 = axes[0].scatter(
            df_plot["T"],
            df_plot["H"],
            c=df_plot["B_pto"],
            cmap="viridis",
            s=100,
        )

        axes[0].set_xlabel("T")
        axes[0].set_ylabel("H")
        axes[0].set_title("Optimal Kp")

        for _, row in df_plot.iterrows():
            axes[0].text(
                row["T"],
                row["H"],
                f"{row['B_pto']:.2g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        cbar1 = fig.colorbar(sc1, ax=axes[0])
        cbar1.set_label("B_pto")

        # Ki subplot
        sc2 = axes[1].scatter(
            df_plot["T"],
            df_plot["H"],
            c=df_plot["S_pto"],
            cmap="viridis",
            s=100,
        )

        axes[1].set_xlabel("T")
        axes[1].set_ylabel("H")
        axes[1].set_title("Optimal Ki")

        for _, row in df_plot.iterrows():
            axes[1].text(
                row["T"],
                row["H"],
                f"{row['S_pto']:.2g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        cbar2 = fig.colorbar(sc2, ax=axes[1])
        cbar2.set_label("S_pto")

        # finalize
        plt.tight_layout()
        plt.savefig(save_path + ".png", dpi=300)
        plt.show()
        plt.close(fig)

        # print
        print("Kp matrix:")
        print(B_pto_map)
        print("Ki matrix:")
        print(S_pto_map)

    def plot_log(self, x, z, y=None, mode="max"):
        """
        Plot logged results.

        Parameters
        ----------
        x : str
            Log key for X axis
        y : str or None
            Log key for Y axis (None → 2D plot)
        z : str
            Log key for Z axis (output variable)
        mode : str
            'max' or 'min' for extreme value
        """

        if not self.log:
            raise RuntimeError("Log is empty. Run the sweep first.")

        df = pd.DataFrame(self.log)

        if z not in df:
            raise KeyError(f"{z} not found in log")

        # find extreme
        if mode == "max":
            idx = df[z].idxmax()
            extreme_label = "Max"
        elif mode == "min":
            idx = df[z].idxmin()
            extreme_label = "Min"
        else:
            raise ValueError("mode must be 'max' or 'min'")

        extreme_value = df.loc[idx, z]

        # 2D plot
        if y is None:
            fig, ax = plt.subplots(figsize=(7, 5))

            sc = ax.scatter(
                df[x],
                df[z],
                c=df[z],
                cmap="viridis",
            )

            ax.set_xlabel(x)
            ax.set_ylabel(z)
            ax.set_title(f"{extreme_label} {z} = {extreme_value:.3g}")

            plt.colorbar(sc, ax=ax, label=z)
            plt.tight_layout()
            plt.show()
            return

        # 3D panel plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        sc = ax.scatter(
            df[x],
            df[y],
            df[z],
            c=df[z],
            cmap="viridis",
            depthshade=True,
        )

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)

        ax.set_title(
            f"{extreme_label} {z} = {extreme_value:.3g}\n"
            f"{x}={df.loc[idx, x]}, {y}={df.loc[idx, y]}"
        )

        fig.colorbar(sc, ax=ax, shrink=0.6, label=z)
        plt.tight_layout()
        plt.show()
