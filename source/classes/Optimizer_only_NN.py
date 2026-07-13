import numpy as np
import torch
import os
import json
import mlflow
import optuna
from collections import defaultdict
import re
import matplotlib.pyplot as plt
import joblib

from source.package_1.classes.DataHandle import DataHandle
from source.package_1.classes.DataBasedFeed import DataBasedFeed
from source.package_1.classes.Estimator import Estimator
from source.package_1.classes.IntegratorNLinear1DOF import IntegratorNLinear1DOF

# TODO : implement X_data and y_data as search parameters
# TODO : implement norm_bond search parameters
# TODO : implement optional Mlflow logging
# TODO : save the normalizer as part of model
# TODO : robust to no eval waves
# TODO : generalize objective with flexible over estimator type
# TODO : clean up objective with helper function
# TODO : make name2params static method

# TODO : improve the figure and time series logging for best model artifact log


class Optimizer:
    def __init__(
        self,
        search_space: dict,
        params_all: dict,
        r=None,
        m=None,
        log_experiment: str = "test",
    ):
        self.search_space = search_space
        self.params_all = params_all
        self.log_experiment = log_experiment
        self.r = r
        self.m = m

        # load data
        self.dataset = DataBasedFeed.dataHandle2dict(
            self.params_all["train_waves"] + self.params_all["eval_waves"]
        )

    def _sample_params(self, trial):
        params = self.params_all.copy()

        for name, cfg in self.search_space.items():
            if cfg["type"] == "int":
                params[name] = trial.suggest_int(name, cfg["low"], cfg["high"])
            elif cfg["type"] == "float":
                params[name] = trial.suggest_float(
                    name, cfg["low"], cfg["high"], log=cfg.get("log", False)
                )
            elif cfg["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, cfg["choices"])

        return params

    # ! debug
    def objective(self, trial):
        param = self._sample_params(trial)

        # one run per Optuna trial
        with mlflow.start_run(nested=False):
            mlflow.log_params(param)

            metrics_hist = defaultdict(list)
            obj_best = float("inf")
            figures = {}

            #! N stochastic
            for s in range(self.params_all["n_stochastic"]):
                with mlflow.start_run(nested=True, run_name=f"stochastic_{s}"):
                    # mlflow.enable_system_metrics_logging() #! need gpu
                    mlflow.log_params(param)

                    # prep data
                    dataset_train = {
                        k: self.dataset[k] for k in self.params_all["train_waves"]
                    }
                    feed = DataBasedFeed(
                        dataset=dataset_train,
                        params=param,
                    )

                    # train estimator
                    estimator = Estimator(
                        model_type="dnn",
                        params=param,
                        train_data=[feed.train_X, feed.train_y],
                        val_data=[feed.val_X, feed.val_y],
                    )
                    estimator.build()

                    # test loss
                    estimator.evaluate(
                        feed.test_X,
                        feed.test_y,
                        log="test",
                    )

                    # log
                    metrics_i = {
                        "train loss": estimator.log["best train loss"],
                        "val loss": estimator.log["best val loss"],
                        "test loss": estimator.log["test loss"],
                        "final epoch": estimator.log["final epoch"],
                    }

                    # loop wave train
                    train_wave_estimator_metric = []
                    train_wave_dyn_metric = []
                    feed.prep_eval(dataset_train)
                    norm_y = feed.get_normalizer(self.params_all["y_data"][0])
                    t_sim = feed.t_eval
                    dt = round(t_sim[1] - t_sim[0], 4)
                    n_steps = len(t_sim)
                    for wave in param["train_waves"]:
                        df_wave = dataset_train[wave]
                        wave_param = self.name2params(wave)

                        # estimator metric full wave train
                        y = estimator.evaluate(
                            feed.eval_data_per_wave[wave]["X"],
                            feed.eval_data_per_wave[wave]["y"],
                            log=wave + "_estimator",
                            metric_fn=self.params_all["metric"],
                            norm=feed.normalizer[self.params_all["y_data"][0]],
                            # plot=True,
                        )
                        metrics_i[wave + "_estimator"] = estimator.log[
                            wave + "_estimator_" + self.params_all["metric"]
                        ]
                        train_wave_estimator_metric.append(
                            metrics_i[wave + "_estimator"]
                        )

                        # dynamic metric full wave train
                        mask = df_wave["t"].isin(t_sim)
                        feed.init_simulation(df_wave, t_sim, device="cpu")
                        eta = df_wave.loc[mask, "eta"].reset_index(drop=True)
                        X0 = feed.x
                        x_true = df_wave.loc[mask, "x"].to_numpy()
                        # norm_y = feed.get_normalizer(self.params_all["y_data"][0])
                        X_nn = feed.X_nn

                        # define integrator
                        wec_nn = IntegratorNLinear1DOF(
                            dt,
                            self.m,
                            t0=t_sim[0],
                            X0=X0,
                            eta=eta,
                            r=self.r,
                            B_pto=wave_param["B"],
                            S_pto=wave_param["S"],
                            eta_nl_hs_bool=True,
                        )

                        # run integrator
                        estimator.model.eval()
                        model = estimator.model
                        model.eval()
                        with torch.inference_mode():
                            for i, t in enumerate(t_sim):
                                # predict NN
                                y_nn = estimator.predict_norm(model, X_nn, norm_y)

                                # step simuation
                                state_nn, t_i = wec_nn.RK4_step(
                                    wec_nn.dyn_nonlinearHs, y_nn, log="fhd"
                                )

                                # update input vector except for last step
                                if i < n_steps - 1:
                                    X_nn = feed.step(state_nn)

                            # with torch.inference_mode():
                            #     for i in range(n_steps):
                            #         y_nn = estimator.predict_norm(model, X_nn, norm_y)
                            #         state_nn, t_i = wec_nn.RK4_step(
                            #             wec_nn.dyn_nonlinearHs, y_nn, log="fhd"
                            #         )
                            #         if i < n_steps - 1:
                            #             X_nn = feed.step(state_nn)

                            wec_nn.post_process(True)
                        # fit
                        metrics_i[wave + "_dyn"] = wec_nn.fit(x_true, "x", t=t_sim)
                        train_wave_dyn_metric.append(metrics_i[wave + "_dyn"])

                        # check
                        plt.figure(figsize=(10, 6))
                        plt.plot(t_sim, x_true, label="cfd - ref")
                        plt.plot(t_sim, wec_nn.X_hist[:, 0], label="nn", linestyle="--")
                        plt.grid(True)
                        plt.title(
                            f"Response T={wave_param['T']}s H={wave_param['H']}m / Fitting NN = {metrics_i[wave + '_dyn']}"
                        )
                        plt.legend()

                    metrics_i["known_mean_estimator"] = np.mean(
                        train_wave_estimator_metric
                    )
                    metrics_i["known_mean_dyn"] = np.mean(train_wave_dyn_metric)

                    # loop wave eval
                    if not self.params_all["eval_waves"]:
                        metrics_i["unknown_mean_estimator"] = metrics_i[
                            "known_mean_estimator"
                        ]
                        metrics_i["unknown_mean_dyn"] = metrics_i["known_mean_dyn"]
                    else:
                        dataset_eval = {
                            k: self.dataset[k] for k in self.params_all["eval_waves"]
                        }
                        eval_wave_estimator_metric = []
                        eval_wave_dyn_metric = []
                        feed.prep_eval(dataset_eval)
                        t_dyn = feed.t_eval
                        dt = round(t_dyn[1] - t_dyn[0], 4)
                        n_steps = len(t_dyn)
                        for wave in feed.eval_data_per_wave.keys():
                            df_wave = dataset_eval[wave]
                            wave_param = self.name2params(wave)

                            # estimator metric full wave eval
                            y = estimator.evaluate(
                                feed.eval_data_per_wave[wave]["X"],
                                feed.eval_data_per_wave[wave]["y"],
                                log=wave + "_estimator",
                                metric_fn=self.params_all["metric"],
                                norm=feed.normalizer[self.params_all["y_data"][0]],
                                # plot=True,
                            )
                            metrics_i[wave + "_estimator"] = estimator.log[
                                wave + "_estimator_" + self.params_all["metric"]
                            ]
                            eval_wave_estimator_metric.append(
                                metrics_i[wave + "_estimator"]
                            )
                            # dynamic metric full wave eval
                            mask = df_wave["t"].isin(t_dyn)
                            feed.init_simulation(df_wave, t_dyn, device="cpu")
                            eta = df_wave.loc[mask, "eta"].reset_index(drop=True)
                            X0 = feed.x
                            x_true = df_wave.loc[mask, "x"].to_numpy()
                            norm_y = feed.get_normalizer(self.params_all["y_data"][0])
                            X_nn = feed.X_nn

                            # define integrator
                            wec_nn = IntegratorNLinear1DOF(
                                dt,
                                self.m,
                                t0=t_dyn[0],
                                X0=X0,
                                eta=eta,
                                r=self.r,
                                B_pto=wave_param["B"],
                                S_pto=wave_param["S"],
                                eta_nl_hs_bool=True,
                            )
                            # run integrator
                            estimator.model.eval()
                            with torch.inference_mode():
                                for i in range(n_steps):
                                    y_nn = estimator.predict_norm(model, X_nn, norm_y)
                                    state_nn, t_i = wec_nn.RK4_step(
                                        wec_nn.dyn_nonlinearHs, y_nn, log="fhd"
                                    )
                                    if i < n_steps - 1:
                                        X_nn = feed.step(state_nn)
                                wec_nn.post_process(True)
                            # fit
                            metrics_i[wave + "_dyn"] = wec_nn.fit(x_true, "x", t=t_dyn)
                            eval_wave_dyn_metric.append(metrics_i[wave + "_dyn"])

                        metrics_i["unknown_mean_estimator"] = np.mean(
                            eval_wave_estimator_metric
                        )
                        metrics_i["unknown_mean_dyn"] = np.mean(eval_wave_dyn_metric)

                    # keep model if best
                    obj = metrics_i[self.params_all["objective"]]
                    if obj < obj_best:
                        obj_best = obj

                        model_best = estimator.model
                        # estimator_best = y
                        estimator_true = feed.eval_data_per_wave[wave]["y"]
                        dyn_best = wec_nn.X_hist[:, 0]
                        dyn_true = x_true
                        # fig, ax = plt.subplots()
                        # ax.plot(t_dyn, estimator_true, label="true")
                        # ax.plot(t_dyn, estimator_best, label="estimator")
                        # ax.set_xlabel(r"$t\;[\mathrm{s}]$")
                        # ax.set_ylabel(r"$F_{\mathrm{NL}}\;[\mathrm{N}]$")
                        # ax.set_title("Estimator prediction")
                        # figures[f"{wave}_estimator"] = fig

                        # fig, ax = plt.subplots()
                        # ax.plot(t_dyn, dyn_true, label="true")
                        # ax.plot(t_dyn, dyn_best, label="estimator")
                        # ax.set_xlabel(r"$t\;[\mathrm{s}]$")
                        # ax.set_ylabel(r"$\xi\;[\mathrm{m}]$")
                        # ax.set_title("Dynamic response")
                        # figures[f"{wave}_dyn"] = fig

                    # append for global metric
                    for k, v in metrics_i.items():
                        metrics_hist[k].append(v)

                    # log mlflow
                    mlflow.log_metrics(metrics_i)

                    plt.show()

            # save model
            mlflow.pytorch.log_model(
                model_best,
                name="model",
                pip_requirements=[
                    "torch==2.9.1+cu126",
                    "numpy",
                    "tqdm",
                ],  # ? to remove reuqired environment warning message
            )

            # save normalizer
            joblib.dump(feed.normalizer, "normalizer.pkl")
            mlflow.log_artifact("normalizer.pkl")

            # return averaged metric
            metric_glob = {k: np.mean(v) for k, v in metrics_hist.items()}

            series_best = {
                "t": t_sim.tolist(),
                "eta": eta.tolist(),
                # "est_pred": estimator_best.tolist(),
                "est_true": estimator_true.tolist(),
                "dyn_pred": dyn_best.tolist(),
                "dyn_true": dyn_true.tolist(),
            }
            metrics_glob_list = {
                k: {
                    "mean": float(np.mean(v)),
                    "list": v.copy(),
                }  # copy to avoid accidental mutation
                for k, v in metrics_hist.items()
            }
            artifacts = series_best | metrics_glob_list
            artifacts_safe = self.make_json_safe(artifacts)
            with open("artifacts.json", "w") as f:
                json.dump(artifacts_safe, f, indent=4)

            with open("param.json", "w") as f:
                json.dump(param, f, indent=4)

            for name, fig in figures.items():
                mlflow.log_figure(fig, f"figures/{name}.png")
                plt.close(fig)
            mlflow.log_artifact("artifacts.json")
            mlflow.log_artifact("param.json")

            mlflow.log_metrics(metric_glob)
            obj_global = metric_glob[self.params_all["objective"]]

            os.remove("artifacts.json")
            os.remove("param.json")
            os.remove("normalizer.pkl")

        return obj_global

    def run(self):
        mlflow.set_experiment(self.log_experiment)

        study = optuna.create_study(
            direction="minimize", storage="sqlite:///optuna_study.db"
        )
        study.optimize(
            self.objective,
            n_trials=self.params_all["n_trials"],
            show_progress_bar=True,
        )

        return study

    def name2params(self, s: str) -> dict:
        pattern = (
            r"T(?P<T>[-+]?\d*\.?\d+)_"
            r"H(?P<H>[-+]?\d*\.?\d+)_"
            r"B(?P<B>-?\d+)_"
            r"S(?P<S>-?\d+)"
        )

        match = re.search(pattern, s)
        if not match:
            raise ValueError(f"String does not match expected format: {s}")

        return {
            "T": float(match.group("T")),
            "H": float(match.group("H")),
            "B": int(match.group("B")),
            "S": int(match.group("S")),
        }

    def make_json_safe(self, obj):
        if isinstance(obj, dict):
            return {k: self.make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_json_safe(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
