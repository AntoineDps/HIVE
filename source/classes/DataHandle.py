import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from source.function.load_hydro_json import load_hydro_json
from source.function.sea_state_utils import eta2all, CWR
from source.function.fe_conv import fe_conv
from source.function.resample import resample
from source.function.save_pathing import save_pathing
from source.classes.Plotter import Plotter

#! check fr
#! check fe
# TODO : pbl spectra do not match

# %% CONSTRUCTOR


class DataHandle:
    def __init__(
        self,
        case_path: Path,
        wave_type,
        T_case,
        H_case,
        damping=0,
        stiffness=0,
        cut_time: tuple = None,
        duration=None,
        data_name={
            "time_s_": "t",
            "center_z_m_": "x",
            "fvel_z_m_s_": "xdot",
            "Elevation_0_m_": "eta",
            "ForceFluid_z_N_": "fhyd_cfd",
        },
        g=9.81,
        rho=1025,
    ):
        self.T_case = T_case
        self.H_case = H_case
        self.wave_type = wave_type
        self.damping = damping
        self.stiffness = stiffness
        self.cut_time = cut_time
        self.duration = duration
        self.data = data_name
        self.rho = rho
        self.g = g

        # build case file
        self.param2case(T_case, H_case)

        # load files
        self.load_cfd_data(case_path, data_name)

        # dt
        self.dt = self.dataset["t"][1] - self.dataset["t"][0]

    # %% METHODS

    def param2case(self, T, H):  #: format depend on sim
        """
        case name input -> raw data file name (CFD folder name, unchanged).
        Also sets save_name: the compact name used for processed output files.
        """
        parts = [
            "out",
            "Case",
            # wave_code,
            str(T).replace(".", ""),
            str(H).replace(".", ""),
        ]

        if self.damping is not None:
            parts.append(str(self.damping))

        if self.stiffness is not None:
            parts.append(str(self.stiffness))

        self.case = "_".join(parts)

        # compact save name: Te5p0_Hs2p0_d120000_k-280000
        def _f(v: float) -> str:
            return f"{float(v):.1f}".replace(".", "p")

        save_parts = [f"Hs{_f(H)}", f"Te{_f(T)}"]
        if self.damping is not None:
            save_parts.append(f"d{int(self.damping)}")
        if self.stiffness is not None:
            save_parts.append(f"k{int(self.stiffness)}")
        self.save_name = "_".join(save_parts)

    def load_cfd_data(self, case_path, data_name):
        """
        load data: from cfd and process or from processed file.
        Looks for both new format (save_name) and old format (case).
        """

        # try new compact save_name format first
        if ((case_path / f"{self.save_name}_data").with_suffix(".csv")).is_file():
            self.dataset = pd.read_csv(
                (case_path / f"{self.save_name}_data").with_suffix(".csv")
            )
            self.wave_params = pd.read_csv(
                (case_path / f"{self.save_name}_wave").with_suffix(".csv")
            )
        # look for existing specific name file (legacy)
        elif (case_path.with_suffix(".csv")).is_file():
            case_path = case_path.parent / case_path.name.rsplit("_", 1)[0]
            self.dataset = pd.read_csv(
                (case_path.parent / (case_path.name + "_data")).with_suffix(".csv")
            )
            self.wave_params = pd.read_csv(
                (case_path.parent / (case_path.name + "_wave")).with_suffix(".csv")
            )
        # look for existing old-format name file
        elif ((case_path / f"{self.case}_data").with_suffix(".csv")).is_file():
            self.dataset = pd.read_csv(
                (case_path / f"{self.case}_data").with_suffix(".csv")
            )
            self.wave_params = pd.read_csv(
                (case_path / f"{self.case}_wave").with_suffix(".csv")
            )

            # process from scratch
        else:
            full_path = case_path / f"{self.case}"
            files = [
                "displacement.csv",
                "wg0.csv",
                "force.csv",
            ]

            df_all = pd.concat(
                [pd.read_csv(f"{full_path}/{f}", sep=";") for f in files], axis=1
            )

            # extract
            cols_to_keep = df_all.columns.intersection(data_name.keys())
            df_all = df_all.loc[:, ~df_all.columns.duplicated()]
            df_all = df_all[cols_to_keep]

            # rename
            df_all = df_all.rename(columns=data_name)

            # cut the dataframe if NaNs exist
            nan_row_idx = df_all.isna().any(axis=1).idxmax()
            if df_all.isna().any().any():
                df_all = df_all.loc[: nan_row_idx - 1]

            # cut transient
            if self.cut_time is not None:
                t_end = df_all["t"].iloc[-1] - self.cut_time[1]
                df_all = df_all[
                    (df_all["t"] >= self.cut_time[0]) & (df_all["t"] <= t_end)
                ].copy()

            # cut duration
            if self.duration is None:
                self.duration = df_all["t"].iloc[-1] - df_all["t"].iloc[0]
            else:
                t_end = df_all["t"].iloc[0] + self.duration
                df_all = df_all[(df_all["t"] <= t_end)]

            # log
            df_all.reset_index(drop=True, inplace=True)
            self.dataset = df_all

    def processData(
        self,
        r=None,
        hydro_sphere=None,
        full_hyd_json_path=None,
        freq="f",
        S_type="JONSWAP",
        gamma=3.3,
    ):
        """
        force convention:
            CFD:
            fhyd_cfd: total nonlinear fluid force from CFD [loaded]

            NL:
            fhs_nl_eta: nonlinear hydrostatic force considering eta [computed]
            fb_nl_eta: nonlinear hydrostatic force considering eta and including gravity [computed]
            fhd_nl_eta: nonlinear hydrodynamic force including radiation force, fhyd_cfd - fb_nl_eta [deducted]
            fhd_nl_eta_r: nonlinear hydrodynamic force not including radiation force, fhyd_cfd - fb_nl_eta - fr [deducted]

            fhs_nl: nonlinear hydrostatic force not considering eta [computed]
            fb_nl: nonlinear hydrostatic force not considering eta and including gravity [computed]
            fhd_nl: nonlinear hydrodynamic force not considering eta and including radiation force, fhyd_cfd - fb_nl [deducted]
            fhd_nl_r: nonlinear hydrodynamic force not considering eta and including radiation force, fhyd_cfd - fb_nl - fr [deducted]

            LIN:
            fhyd_lin: total linear fluid force from BEM [computed]
            fhd_lin_comp: equivalent to fe + fr, linear excitation force [computed]
            fe_lin : fe from linear convolution [computed]
            fr_lin : fr from radiation state-space [computed]
            fhs_lin: linear hydrostatic force [computed]
            fb_lin: linear hydrostatic force including gravity [computed]

            fhd_lin_r: nonlinear hydrodynamic force not including radiation force, fhyd_cfd - fb_lin -fr [deducted]
            fhd_lin: nonlinear hydrodynamic force not including radiation force, fhyd_cfd - fb_lin [deducted]
        """

        self.S_type = S_type
        self.gamma = gamma

        # wave parameters
        self.wave_params = eta2all(
            self.dataset["t"],
            self.dataset["eta"],
            wave_type=self.wave_type,
            freq=freq,
            S_type=self.S_type,
            gamma=self.gamma,
        )

        # fpto
        self.dataset["fd_pto"] = -self.damping * self.dataset["xdot"]
        self.dataset["fs_pto"] = -self.stiffness * self.dataset["x"]
        self.dataset["fpto"] = self.dataset["fd_pto"] + self.dataset["fs_pto"]

        # power
        self.dataset["Pd"] = -self.dataset["fd_pto"] * self.dataset["xdot"]
        self.dataset["Ps"] = -self.dataset["fs_pto"] * self.dataset["xdot"]
        self.dataset["Pabs"] = self.dataset["Pd"] + self.dataset["Ps"]

        self.dataset["Pabs_mean"] = np.mean(self.dataset["Pabs"])
        self.dataset["Pd_mean"] = np.mean(self.dataset["Pd"])
        self.dataset["Ps_mean"] = np.mean(self.dataset["Ps"])

        self.dataset["Eabs_accum"] = np.cumsum(self.dataset["Pabs"]) * self.dt
        self.dataset["Ed_accum"] = np.cumsum(self.dataset["Pd"]) * self.dt
        self.dataset["Es_accum"] = np.cumsum(self.dataset["Ps"]) * self.dt

        if r is not None:
            self.dataset["CWRabs"] = CWR(
                self.dataset["Pabs_mean"], self.wave_params["J"], r
            )

        # others
        self.dataset["etadot"] = np.gradient(self.dataset["eta"].to_numpy(), self.dt)
        self.dataset["xddot"] = np.gradient(self.dataset["xdot"].to_numpy(), self.dt)

        if r is not None:
            m = self.rho * 2 / 3 * r**3
            self.dataset["fg"] = -m * self.g

        self.dataset["fi"] = m * self.dataset["xddot"]

        # base hydro
        if r is not None:
            # fhs_nl
            h = r - self.dataset["x"]
            h = np.clip(h, 0, 2 * r)
            self.dataset["Sw"] = 2 * np.pi * r * h
            self.dataset["Vs"] = (np.pi * h**2 * (3 * r - h)) / 3
            self.dataset["fhs_nl"] = self.dataset["Vs"] * self.rho * self.g
            self.dataset["fb_nl"] = self.dataset["fhs_nl"] + self.dataset["fg"]

            # fhs_nl_eta
            h_eta = r + (self.dataset["eta"] - self.dataset["x"])
            h_eta = np.clip(h_eta, 0, 2 * r)
            self.dataset["Sw_eta"] = 2 * np.pi * r * h_eta
            self.dataset["Vs_eta"] = (np.pi * h_eta**2 * (3 * r - h_eta)) / 3
            self.dataset["fhs_nl_eta"] = self.dataset["Vs_eta"] * self.rho * self.g
            self.dataset["fb_nl_eta"] = self.dataset["fhs_nl_eta"] + self.dataset["fg"]

            # fb_lin
            self.dataset["fb_lin"] = (
                -(self.dataset["x"] * np.pi * r**2) * self.rho * self.g
            )
            self.dataset["fhs_lin"] = self.dataset["fb_lin"] - self.dataset["fg"]

        if hydro_sphere is not None or full_hyd_json_path is not None:
            if hydro_sphere is not None:
                fe_irf = hydro_sphere.fe_irf
                t_irf = hydro_sphere.t_irf
                rad_ss = hydro_sphere.rad_ss
                ma_inf = hydro_sphere.ma_inf
            else:
                data = load_hydro_json(full_hyd_json_path)
                fe_irf = data["fe_irf"]
                t_irf = data["t_irf"]
                rad_ss = data["rad_ss"]
                ma_inf = data["ma_inf"]

            # fe
            self.dataset["fe_lin"] = fe_conv(
                self.dataset["eta"], fe_irf, self.dataset["t"], t_irf, resample="irf"
            )

            # fr

            Ar = rad_ss["Ar"]
            Br = rad_ss["Br"]
            Cr = rad_ss["Cr"]
            Dr = rad_ss["Dr"]

            xdot = self.dataset["xdot"]

            n_steps = len(xdot)
            n_states = Ar.shape[0]

            z = np.zeros((n_steps, n_states))
            fr = np.zeros(n_steps)

            for k in range(1, n_steps):
                # Forward Euler (or better integrator)
                z[k] = z[k - 1] + self.dt * (Ar @ z[k - 1] + Br * xdot[k - 1])

                fr[k] = Cr @ z[k] + Dr * xdot[k]

            self.dataset["fr"] = -(fr + ma_inf * self.dataset["xddot"])

            # fhd_lin, fhyd_lin
            self.dataset["fhd_lin_comp"] = self.dataset["fe_lin"] + self.dataset["fr"]
            self.dataset["fhyd_lin"] = (
                self.dataset["fhd_lin_comp"] + self.dataset["fhs_lin"]
            )

        # deduced hydro
        if r is not None:
            self.dataset["fhd_lin"] = self.dataset["fhyd_cfd"] - self.dataset["fb_lin"]
            self.dataset["fhd_nl"] = self.dataset["fhyd_cfd"] - self.dataset["fb_nl"]
            self.dataset["fhd_nl_eta"] = (
                self.dataset["fhyd_cfd"] - self.dataset["fb_nl_eta"]
            )

            if hydro_sphere is not None or full_hyd_json_path is not None:  #!
                self.dataset["fhd_lin_r"] = (
                    self.dataset["fhyd_cfd"]
                    - self.dataset["fb_lin"]
                    - self.dataset["fr"]
                )
                self.dataset["fhd_nl_r"] = (
                    self.dataset["fhyd_cfd"]
                    - self.dataset["fb_nl"]
                    - self.dataset["fr"]
                )
                self.dataset["fhd_nl_eta_r"] = (
                    self.dataset["fhyd_cfd"]
                    - self.dataset["fb_nl_eta"]
                    - self.dataset["fr"]
                )

        # active draft
        if r is not None:
            self.dataset["d"] = r + (self.dataset["eta"] - self.dataset["x"])
            self.dataset["d"] = np.clip(self.dataset["d"], 0, 2 * r)

    def resample(self, new_dt, time_col="t", kind="linear"):
        if new_dt == self.dt:
            print("Warning: new_dt is the same as current dt. No resampling performed.")
            return
        if new_dt <= 0:
            raise ValueError("new_dt must be a positive value.")

        t = self.dataset[time_col].values
        new_data = {}
        t_new = None
        for col in self.dataset.columns:
            if col == time_col:
                continue
            t_new, x_new = resample(
                t, self.dataset[col].values, new_dt=new_dt, kind=kind
            )
            new_data[col] = x_new
        new_data[time_col] = t_new

        self.dataset = pd.DataFrame(new_data)
        self.dt = new_dt

    @staticmethod
    def cut_time(
        dataset,
        t_start,
        t_end,
        time_col="t",
        inclusive="both",
    ):
        if t_start >= t_end:
            raise ValueError("t_start must be smaller than t_end")

        def _cut_df(df: pd.DataFrame) -> pd.DataFrame:
            if time_col not in df.columns:
                raise KeyError(f"'{time_col}' not found in DataFrame")

            mask = df[time_col].between(t_start, t_end, inclusive=inclusive)
            return df.loc[mask].reset_index(drop=True)

        # Case 1: single DataFrame
        if isinstance(dataset, pd.DataFrame):
            return _cut_df(dataset)

        # Case 2: dict of DataFrames
        if isinstance(dataset, dict):
            out = {}
            for k, v in dataset.items():
                if not isinstance(v, pd.DataFrame):
                    raise TypeError(
                        f"Value for key '{k}' is not a DataFrame (got {type(v)})"
                    )
                out[k] = _cut_df(v)
            return out

        # Otherwise
        raise TypeError("dataset must be a pandas DataFrame or a dict of DataFrames")

    def save(self, path, name=None):

        # name — use compact save_name by default
        if name is None:
            name = self.save_name

        # make path
        full_path_data = save_pathing(path, name, "data", ".csv")
        full_path_wave = save_pathing(path, name, "wave", ".csv")

        # save to csv
        self.dataset.to_csv(full_path_data, index=False, encoding="utf-8")
        self.wave_params.to_csv(full_path_wave, index=False, encoding="utf-8")

    @property
    def label(self):
        def _f(v: float) -> str:
            return f"{float(v):.1f}".replace(".", "p")

        parts = [
            f"Hs{_f(self.wave_params.Hs[0])}",
            f"Te{_f(self.wave_params.Te[0])}",
        ]
        if self.damping is not None:
            parts.append(f"d{int(self.damping)}")
        if self.stiffness is not None:
            parts.append(f"k{int(self.stiffness)}")
        return "_".join(parts)

    @classmethod
    def load_case_in(cls, case_dir: Path, json_path: Path = None) -> list:
        """
        Reconstruct the DataHandle objects for every case saved into case_dir,
        using the exact T_case/H_case/damping/stiffness/wave_type/cut_time/duration
        recorded in json_path (the run_process_raw.py-style config copied alongside
        the data). Each case is built through the normal constructor, so
        load_cfd_data picks up the already-saved CSVs directly -- no raw reprocessing.
        """
        if json_path is None:
            json_path = case_dir / f"{case_dir.name}.json"

        if not json_path.is_file():
            raise FileNotFoundError(
                f"No config copy found at {json_path}. from_processed_folder needs "
                "the run_process_raw.py config that produced this folder to "
                "reconstruct each case's exact parameters."
            )

        with open(json_path, "r") as f:
            raw = json.load(f)

        wave_type = raw.get("wave_type")
        cut_time = tuple(raw["cut_time"]) if raw.get("cut_time") is not None else None
        duration = raw.get("duration")

        case_lookup = {}
        for c in raw["cases"]:
            # old CFD folder name (for finding raw data)
            case_str = "_".join(
                [
                    "out",
                    "Case",
                    str(c["T"]).replace(".", ""),
                    str(c["H"]).replace(".", ""),
                    str(c["damping"]),
                    str(c["stiffness"]),
                ]
            )

            # new compact save name
            def _f(v):
                return f"{float(v):.1f}".replace(".", "p")

            save_name = "_".join(
                [
                    f"Hs{_f(c['H'])}",
                    f"Te{_f(c['T'])}",
                    f"d{int(c['damping'])}",
                    f"k{int(c['stiffness'])}",
                ]
            )
            case_lookup[case_str] = c  # old format key
            case_lookup[save_name] = c  # new format key

        instances = []
        for data_path in sorted(case_dir.glob("*_data.csv")):
            case = data_path.name[: -len("_data.csv")]
            if case not in case_lookup:
                raise ValueError(
                    f"Found '{data_path.name}' with no matching entry in {json_path}"
                )

            c = case_lookup[case]
            obj = cls(
                case_path=case_dir,
                wave_type=wave_type,
                T_case=c["T"],
                H_case=c["H"],
                damping=c["damping"],
                stiffness=c["stiffness"],
                cut_time=cut_time,
                duration=duration,
            )
            instances.append(obj)

        return instances

    # %% PLOTS

    # sea state
    @staticmethod
    def plot_seastate(objs, save_path=None, name=None):
        DataHandle.plot_spectrum(objs, save_path=save_path, name=name)
        DataHandle.plot_line(objs, y="eta", save_path=save_path, name=name)
        DataHandle.plot_grid_ss(objs, save_path=save_path, name=name)

    @staticmethod
    def plot_spectrum(objs, save_path=None, name=None):

        if not isinstance(objs, (list, tuple)):
            objs = [objs]

        if "w" in objs[0].wave_params.columns:
            x_col, xlabel = "w", "omega [rad/s]"
        elif "f" in objs[0].wave_params.columns:
            x_col, xlabel = "f", "f [Hz]"
        else:
            raise KeyError("wave_params must contain either 'w' or 'f'")

        series = []
        for obj in objs:
            x = obj.wave_params[x_col]
            series.append((x, obj.wave_params.S_fft, f"S_fft - {obj.label}"))
            series.append((x, obj.wave_params.S_fft_filt, f"S_fft_filt - {obj.label}"))
            series.append((x, obj.wave_params.S_welch, f"S_welch - {obj.label}"))
            series.append(
                (
                    x,
                    obj.wave_params.S_th,
                    f"theoretical ({obj.wave_params.S_type[0]}, "
                    f"gamma={obj.wave_params.gamma[0]}) - {obj.label}",
                )
            )

        Plotter.plot_lines(
            series,
            xlabel=xlabel,
            ylabel="m^2.Hz",
            title="Spectrum realization",
            save_path=save_path,
            name=name,
            suffix="spectrum",
        )

    @staticmethod
    def plot_grid_ss(objs, save_path=None, name=None):

        # TODO : modeling domain and steepness oriented grid plot // make advanced plot a reusable function

        if not isinstance(objs, (list, tuple)):
            objs = [objs]

        Te = np.array([obj.wave_params.Te[0] for obj in objs])
        Hs = np.array([obj.wave_params.Hs[0] for obj in objs])
        J = np.array([obj.wave_params.J[0] for obj in objs])
        eps = np.array([obj.wave_params.eps[0] for obj in objs])

        Plotter.plot_grid(
            Te,
            Hs,
            s=J,
            c=eps,
            xlabel="Te [s]",
            ylabel="Hs [m]",
            clabel="Steepness ε",
            title="Sea state grid",
            save_path=save_path,
            name=name,
            suffix="ss_grid",
        )

    # power
    @staticmethod
    def plot_power(objs, save_path=None, name=None):
        DataHandle.plot_line(objs, y="power", save_path=save_path, name=name)
        DataHandle.plot_grid_power(
            objs, metric="Pabs_mean", save_path=save_path, name=name
        )
        DataHandle.plot_grid_power(
            objs, metric="CWRabs", save_path=save_path, name=name
        )
        DataHandle.plot_accumulated_energy(objs, save_path=save_path, name=name)

    @staticmethod
    def plot_grid_power(objs, metric="Pabs_mean", save_path=None, name=None):
        """
        Scatter cases on the (Te, Hs) grid (marker size ~ energy flux J),
        colored by any scalar-per-case dataset column, e.g. "Pabs_mean",
        "Puse_mean", "CWRabs", "CWRuse".
        """
        if not isinstance(objs, (list, tuple)):
            objs = [objs]

        if metric not in objs[0].dataset.columns:
            raise KeyError(f"'{metric}' not found in dataset")

        Te = np.array([obj.wave_params.Te[0] for obj in objs])
        Hs = np.array([obj.wave_params.Hs[0] for obj in objs])
        J = np.array([obj.wave_params.J[0] for obj in objs])
        values = np.array([obj.dataset[metric][0] for obj in objs])

        Plotter.plot_grid(
            Te,
            Hs,
            s=J,
            c=values,
            xlabel="Te [s]",
            ylabel="Hs [m]",
            clabel=metric,
            title=f"{metric} grid",
            save_path=save_path,
            name=name,
            suffix=f"grid_{metric}",
        )

    @staticmethod
    def plot_accumulated_energy(objs, variable="Eabs_accum", save_path=None, name=None):
        if not isinstance(objs, (list, tuple)):
            objs = [objs]

        if variable not in objs[0].dataset.columns:
            raise KeyError(f"'{variable}' not found in dataset")

        series = [(obj.dataset["t"], obj.dataset[variable], obj.label) for obj in objs]

        Plotter.plot_lines(
            series,
            xlabel="t [s]",
            ylabel="Energy [J]",
            title="Accumulated energy",
            save_path=save_path,
            name=name,
            suffix="energy_accum",
        )

    # dynamics
    @staticmethod
    def plot_dynamics(objs, save_path=None, name=None):
        # x vs xdot
        DataHandle.plot_line(
            objs,
            x="x",
            y="xdot",
            xlabel="x [m]",
            ylabel="xdot [m/s]",
            title="Phase plot (x vs xdot)",
            save_path=save_path,
            name=name,
            suffix="phase",
        )
        # x, eta, xdot vs time (the default plot_line shortcut)
        DataHandle.plot_line(objs, save_path=save_path, name=name)

    # variable: takes a column name or list of column names instead of true/false
    @staticmethod
    def plot_variable(objs, variable, save_path=None, name=None):
        """Plot one or more dataset columns by name via plot_line."""
        variables = variable if isinstance(variable, list) else [variable]
        for var in variables:
            DataHandle.plot_line(
                objs,
                y=var,
                ylabel=var,
                save_path=save_path,
                name=name,
                suffix=var,
            )

    # hydro
    @staticmethod
    def plot_hydro(objs, save_path=None, name=None):
        # draft variation
        DataHandle.plot_distribution(
            objs,
            variable="d",
            xlabel="Draft [m]",
            title="Draft variation",
            save_path=save_path,
            name=name,
            suffix="draft",
        )
        # linear force (added mass/radiation, hydrostatic, fe)
        DataHandle.plot_line(objs, y="lin_force", save_path=save_path, name=name)
        # CFD force (fluid force incl mass, hydrodyn force incl mass, nl hydrostatic)
        DataHandle.plot_line(objs, y="cfd_force", save_path=save_path, name=name)
        # lin hs vs nl hs (incl mass)
        DataHandle.plot_line(objs, y="hs_lin_vs_nl", save_path=save_path, name=name)
        # fe + fr vs nlfe
        DataHandle.plot_line(objs, y="fe_lin_vs_nl", save_path=save_path, name=name)
        # total fluid force CFD (incl mass) vs BEM
        DataHandle.plot_line(objs, y="force_cfd_vs_bem", save_path=save_path, name=name)

    def distri_data(
        self,
        variables=None,
        normalize=False,
        bins=50,
        density=True,
        save_path=None,
        name=None,
    ):
        if variables is None:
            variables = ["x", "xdot", "eta"]
        elif isinstance(variables, str):
            variables = [variables]

        series = []
        for var in variables:
            if var in self.dataset.keys():
                data = self.dataset[var]

                if normalize:
                    max_val = np.max(np.abs(data))
                    if max_val > 0:
                        data = data / max_val

                series.append((data, var))
            else:
                print(f"Warning: '{var}' not recognized")

        Plotter.plot_histogram(
            series,
            xlabel="Normalized value" if normalize else "Value",
            ylabel="Probability density" if density else "Count",
            title="Distribution of variables",
            bins=bins,
            density=density,
            save_path=save_path,
            name=name,
            suffix="distri",
        )
        plt.show()

    @staticmethod
    def scatter3d(
        objs,
        x="x",
        y="xdot",
        z="eta",
        normalize=False,
        xlabel=None,
        ylabel=None,
        zlabel=None,
        title=None,
        save_path=None,
        name=None,
        suffix=None,
    ):
        """
        DataHandle's default 3D combo (x, xdot, eta) over Plotter.scatter3d.
        Pass any other column names for a different 3D view; one point
        series per case.
        """
        Plotter.scatter3d(
            objs,
            x=x,
            y=y,
            z=z,
            normalize=normalize,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            title=title if title is not None else f"{z} vs {y} vs {x}",
            save_path=save_path,
            name=name,
            suffix=suffix if suffix is not None else "scatter3d",
        )

    @staticmethod
    def plot_distribution(
        objs,
        variable,
        bins=50,
        density=True,
        normalize=False,
        xlabel=None,
        ylabel=None,
        title=None,
        save_path=None,
        name=None,
        suffix=None,
    ):
        """
        Compare one variable's distribution across one or more cases, one
        histogram per case. Different shape from distri_data (which
        compares multiple variables within a single case).
        """
        if not isinstance(objs, (list, tuple)):
            objs = [objs]

        series = []
        for obj in objs:
            if variable in obj.dataset.columns:
                data = obj.dataset[variable]

                if normalize:
                    max_val = np.max(np.abs(data))
                    if max_val > 0:
                        data = data / max_val

                series.append((data, obj.label))
            else:
                print(f"Warning: '{variable}' not recognized")

        Plotter.plot_histogram(
            series,
            xlabel=xlabel
            if xlabel is not None
            else ("Normalized value" if normalize else variable),
            ylabel=ylabel
            if ylabel is not None
            else ("Probability density" if density else "Count"),
            title=title if title is not None else f"Distribution of {variable}",
            bins=bins,
            density=density,
            save_path=save_path,
            name=name,
            suffix=suffix if suffix is not None else f"distri_{variable}",
        )

    @staticmethod
    def plot_line(
        objs,
        x="t",
        y=None,
        normalize=False,
        xlabel=None,
        ylabel=None,
        title=None,
        save_path=None,
        name=None,
        suffix=None,
    ):
        """
        DataHandle's shortcut layer over Plotter.plot_line: knows what "pto",
        "power", "eta", and the default dynamics group mean for this data
        type -- which columns, which axis labels, which title. Pass any
        other column name (or list of column names) as `y` for fully
        spontaneous plotting (e.g. x="x", y="xdot" for a phase plot); in
        that case Plotter.plot_line derives labels from whatever was
        actually asked for, since there's no predefined meaning to draw on.

        The actual data extraction and drawing lives in Plotter.plot_line,
        which only assumes objects expose `.dataset` (column access) and
        `.label` -- so it works just as well for any other data type built
        the same way, not just this one.
        """
        SHORTCUTS = {
            "pto": (["fpto", "fs_pto", "fd_pto"], "Force [N]", "PTO forces", "pto"),
            "power": (
                ["Pabs", "Pd", "Ps", "Pabs_mean", "Pd_mean", "Ps_mean"],
                "Power [W]",
                "Power time series",
                "power",
            ),
            "eta": (["eta"], "eta [m]", "Free surface elevation", "eta"),
            # linear force components: radiation (incl. added mass), hydrostatic, excitation
            "lin_force": (
                ["fr", "fb_lin", "fe_lin"],
                "Force [N]",
                "Linear force components",
                "lin_force",
            ),
            # CFD-side force breakdown: raw loaded force, deduced hydrodynamic
            # (still includes radiation/added-mass), deduced nonlinear hydrostatic
            "cfd_force": (
                ["fhyd_cfd", "fhd_nl_eta", "fb_nl_eta"],
                "Force [N]",
                "CFD force components",
                "cfd_force",
            ),
            # linear vs nonlinear hydrostatic, both including the gravity/mass term
            "hs_lin_vs_nl": (
                ["fb_lin", "fb_nl_eta"],
                "Force [N]",
                "Linear vs nonlinear hydrostatic force (incl. gravity)",
                "hs_lin_vs_nl",
            ),
            # linear excitation+radiation combo (fe+fr) vs its CFD-deduced
            # nonlinear equivalent (CFD force minus nonlinear hydrostatic)
            "fe_lin_vs_nl": (
                ["fhd_lin_comp", "fhd_nl_eta"],
                "Force [N]",
                "Linear (fe + fr) vs nonlinear hydrodynamic force",
                "fe_lin_vs_nl",
            ),
            # total fluid force as measured by CFD vs as computed by BEM
            "force_cfd_vs_bem": (
                ["fhyd_cfd", "fhyd_lin"],
                "Force [N]",
                "Total fluid force: CFD vs BEM",
                "force_cfd_vs_bem",
            ),
        }

        if y is None:
            variables, default_ylabel, default_title, default_suffix = (
                ["x", "xdot", "eta"],
                "Value",
                "Dynamics",
                "dynamics",
            )
        elif isinstance(y, str) and y in SHORTCUTS:
            variables, default_ylabel, default_title, default_suffix = SHORTCUTS[y]
        elif isinstance(y, str):
            variables, default_ylabel, default_title, default_suffix = (
                [y],
                y,
                f"{y} vs {x}",
                y,
            )
        else:
            variables, default_ylabel, default_title, default_suffix = (
                list(y),
                None,
                None,
                "line",
            )

        default_xlabel = "t [s]" if x == "t" else x

        Plotter.plot_line(
            objs,
            x=x,
            y=variables,
            normalize=normalize,
            xlabel=xlabel if xlabel is not None else default_xlabel,
            ylabel=ylabel
            if ylabel is not None
            else ("Normalized value" if normalize else default_ylabel),
            title=title if title is not None else default_title,
            save_path=save_path,
            name=name,
            suffix=suffix if suffix is not None else default_suffix,
        )
