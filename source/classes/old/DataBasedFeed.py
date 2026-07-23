import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import pickle
from pathlib import Path

# TODO : implement kfolds?
# TODO : make method static
# TODO : randomizing the split


class DataBasedFeed:
    @staticmethod
    def dataHandle2dict(names, path="data/handled_data/"):
        directory = Path(path)
        datasets = {}

        for name in names:
            file_path = directory / f"{name}.pkl"

            with open(file_path, "rb") as f:
                handle = pickle.load(f)

            datasets[name] = handle.dataset

        return datasets

    def __init__(self, dataset: dict, params: dict):
        self.dataset = dataset if isinstance(dataset, dict) else {"wave_0": dataset}
        self.params = params
        self.params["X_data"] = self._as_list(self.params["X_data"])
        self.params["y_data"] = self._as_list(self.params["y_data"])
        self.normalizer = {}
        self.data_per_wave = {}

        # all features
        all_features = self.params["X_data"] + self.params["y_data"]

        # get global parameters
        p = {
            "train": self.params.setdefault("train_p", 0),
            "val": self.params.setdefault("val_p", 0),
            "test": self.params.setdefault("test_p", 0),
        }
        norm_from = self.params.setdefault("norm_from", "train")
        self.params.setdefault("norm_bond", [-1, 1])

        self.pmax = max(self.params.setdefault(f"{col}_p", 0) for col in all_features)

        self.fmax = max(self.params.setdefault(f"{col}_f", 0) for col in all_features)

        # initialize windows for all waves -> dict{feature: []}
        data_for_norm = {k: [] for k in all_features}

        for wave_id, wave_df in self.dataset.items():
            wave_X = {"train": [], "val": [], "test": []}
            wave_y = {"train": [], "val": [], "test": []}

            for k in all_features:
                # extract value
                v = wave_df[k]

                # get taylored parameters
                pi = self.params.get(f"{k}_p")
                fi = self.params.get(f"{k}_f")

                # window
                dataW = self.window(v.to_numpy(), fi, pi, self.fmax, self.pmax)

                # split
                dataW_train, dataW_val, dataW_test = self.split(
                    dataW, p["train"], p["val"], p["test"]
                )

                # get and store norm data for later normalization
                norm_data = self._select_norm_data(
                    dataW_train, dataW_val, dataW_test, norm_from
                )
                data_for_norm[k].append(norm_data)

                # store X, y temporary for this wave
                if k in self.params["X_data"]:
                    wave_X["train"].append(dataW_train)
                    wave_X["val"].append(dataW_val)
                    wave_X["test"].append(dataW_test)
                else:
                    wave_y["train"].append(dataW_train)
                    wave_y["val"].append(dataW_val)
                    wave_y["test"].append(dataW_test)

            # store the data per wave : [X][train]["eta" as i]
            self.data_per_wave[wave_id] = {"X": wave_X, "y": wave_y}

        # identify normalizer
        for k in all_features:
            _, self.normalizer[k] = self.normalize(
                np.concatenate(data_for_norm[k], axis=0),
                fit=True,
                min_max=self.params["norm_bond"],
            )

        # normalize per wave data
        for wave_id, wave_df in self.data_per_wave.items():
            # loop split set
            for split in ["train", "val", "test"]:
                # loop i,k for X
                for i, k in enumerate(self.params["X_data"]):
                    # normalize
                    arr = wave_df["X"][split][i]
                    self.data_per_wave[wave_id]["X"][split][i] = self._ensure_2d(
                        self.normalize(arr, norm=self.normalizer[k])
                    )

                # loop i,k for y
                for i, k in enumerate(self.params["y_data"]):
                    # normalize
                    arr = wave_df["y"][split][i]
                    self.data_per_wave[wave_id]["y"][split][i] = self._ensure_2d(
                        self.normalize(arr, norm=self.normalizer[k])
                    )

                # concatenate
                self.data_per_wave[wave_id]["X"][split] = np.concatenate(
                    self.data_per_wave[wave_id]["X"][split][:], axis=1
                )
                self.data_per_wave[wave_id]["y"][split] = np.concatenate(
                    self.data_per_wave[wave_id]["y"][split][:], axis=1
                )

        # wave concatenation
        self.train_X = np.concatenate(
            [self.data_per_wave[w]["X"]["train"] for w in self.data_per_wave.keys()],
            axis=0,
        )
        self.val_X = np.concatenate(
            [self.data_per_wave[w]["X"]["val"] for w in self.data_per_wave.keys()],
            axis=0,
        )
        self.test_X = np.concatenate(
            [self.data_per_wave[w]["X"]["test"] for w in self.data_per_wave.keys()],
            axis=0,
        )

        self.train_y = np.concatenate(
            [self.data_per_wave[w]["y"]["train"] for w in self.data_per_wave.keys()],
            axis=0,
        )
        self.val_y = np.concatenate(
            [self.data_per_wave[w]["y"]["val"] for w in self.data_per_wave.keys()],
            axis=0,
        )
        self.test_y = np.concatenate(
            [self.data_per_wave[w]["y"]["test"] for w in self.data_per_wave.keys()],
            axis=0,
        )

        # check that all time vectors are equal
        first_df = next(iter(self.dataset.values()))["t"]
        first_t = first_df.to_numpy()
        for i, df in enumerate(list(self.dataset.values())[1:], start=1):
            if not np.allclose(first_t, df["t"].to_numpy(), atol=1e-8):
                raise ValueError(
                    f"Time vector of dataset {i} does not match the first dataset."
                )

        # time vector
        t = first_t
        timeW = self.window(t, 0, 0, self.fmax, self.pmax)
        self.t_split = self.split(timeW, p["train"], p["val"], p["test"])

    def prep_eval(self, dataset: dict):
        dataset = dataset if isinstance(dataset, dict) else {"wave_0": dataset}

        self.eval_data_per_wave = {}

        all_features = self.params["X_data"] + self.params["y_data"]

        for wave_id, wave_df in dataset.items():
            wave_X = []
            wave_y = []

            for k in all_features:
                # extract raw signal
                v = wave_df[k]

                # window parameters
                pi = self.params.get(f"{k}_p", 0)
                fi = self.params.get(f"{k}_f", 0)

                # windowing
                dataW = self.window(
                    v.to_numpy(),
                    fi,
                    pi,
                    self.fmax,
                    self.pmax,
                )

                # normalize (use fitted normalizer)
                dataW = self.normalize(dataW, norm=self.normalizer[k])
                dataW = self._ensure_2d(dataW)

                # store
                if k in self.params["X_data"]:
                    wave_X.append(dataW)
                else:
                    wave_y.append(dataW)

            # concatenate features per wave
            X_wave = np.concatenate(wave_X, axis=1) if wave_X else None
            y_wave = np.concatenate(wave_y, axis=1) if wave_y else None

            self.eval_data_per_wave[wave_id] = {
                "X": X_wave,
                "y": y_wave,
            }

        # check time
        first_t = next(iter(dataset.values()))["t"].to_numpy()
        for i, df in enumerate(list(dataset.values())[1:], start=1):
            if not np.allclose(first_t, df["t"].to_numpy(), atol=1e-8):
                raise ValueError(
                    f"Time vector of dataset {i} does not match the first dataset."
                )

        t = first_t
        self.t_eval = self.window(t, 0, 0, self.fmax, self.pmax)

    def init_simulation(self, dataset, t_sim, device="cpu"):
        self.i = 0
        self.t_sim = t_sim

        # initial state
        self.x = np.array(
            [
                dataset[dataset["t"] == t_sim[0]]["x"],
                dataset[dataset["t"] == t_sim[0]]["xdot"],
            ]
        ).ravel()

        # signal provider for window update
        self.signal_providers = {
            "eta": lambda: self.eta_sim[self.i + self.params["eta_f"]],
            "x": lambda: self.x[0],
            "xdot": lambda: self.x[1],
        }

        # determine wave as simulation reference
        id_start = dataset.index[np.isclose(dataset["t"], t_sim[0], atol=1e-4)][0]
        id_end = (
            dataset.index[np.isclose(dataset["t"], t_sim[-1], atol=1e-4)][0]
            + self.params["eta_f"]
        )
        self.eta_sim = dataset["eta"].iloc[id_start : id_end + 1].reset_index(drop=True)

        self.device = device

        # initialize raw windows
        self.dicW = self.init_window(dataset, t_sim[0])

        # compute offsets
        self.offsets = {}
        start = 0
        for k, v in self.dicW.items():
            size = v.size
            self.offsets[k] = (start, start + size)
            start += size

        # allocate NN input tensor once
        self.X_nn = torch.empty((1, start), dtype=torch.float32, device=device)

        # fill initial normalized values
        for k, v in self.dicW.items():
            i0, i1 = self.offsets[k]
            v_norm = self.normalizer[k].transform(v.reshape(-1, 1)).ravel()  #!
            self.X_nn[0, i0:i1] = torch.from_numpy(v_norm)

    def step(self, x):
        self.x = x
        self.i += 1

        # update raw windows
        updates = {k: self.signal_providers[k]() for k in self.params["X_data"]}
        self.dicW = self.update_window(self.dicW, updates)

        # update NN input tensor in-place
        for k, v in self.dicW.items():
            i0, i1 = self.offsets[k]
            v_norm = self.normalizer[k].transform(v.reshape(-1, 1)).ravel()
            self.X_nn[0, i0:i1] = torch.from_numpy(v_norm).to(self.device)

        return self.X_nn

    def window(
        self, vector: np.ndarray, future_values, past_values, futur_max, past_max
    ):
        windows = np.array(
            [
                vector[i - past_values : i + future_values + 1]
                for i in range(past_max, len(vector) - futur_max)
            ]
        )

        windows = windows.reshape(-1) if windows.shape[1] == 1 else windows
        return windows

    def split(self, dataW, train_p, val_p, test_p):
        total_p = train_p + val_p + test_p

        if not np.isclose(total_p, 1.0, atol=1e-2):
            raise ValueError("Sum of percentage is different than 1")

        n_train = int(dataW.shape[0] * train_p)
        n_val = int(dataW.shape[0] * val_p)

        dataW_train = dataW[:n_train]
        dataW_val = dataW[n_train : n_train + n_val]
        if test_p != 0:
            dataW_test = dataW[n_train + n_val :]
        else:
            dataW_test = np.array([])

        dataW_train = None if dataW_train.shape[0] == 0 else dataW_train
        dataW_val = None if dataW_val.shape[0] == 0 else dataW_val
        dataW_test = None if dataW_test.shape[0] == 0 else dataW_test

        return dataW_train, dataW_val, dataW_test

    def normalize(self, data: np.array, fit=False, norm=None, min_max=[-1, 1]):
        """
        Normalize the data
        """
        # robust None
        if data is None:
            return (None, norm) if fit else None
        # create normalizer
        if norm is None:
            norm = MinMaxScaler(feature_range=(min_max[0], min_max[1]))

        # save the initial shape of the array
        shape = data.shape

        # flatten
        vector = data.reshape(-1, 1)

        # normalize
        if fit is True:
            data_norm = norm.fit_transform(vector)
            return data_norm.reshape(shape), norm
        else:
            data_norm = norm.transform(vector)
            return data_norm.reshape(shape)

    def init_window(self, dataset, time):
        initial_window = {}

        idx = dataset.index[dataset["t"] == time][0]
        for k in self.params["X_data"]:
            initial_window[k] = (
                dataset[k]
                .iloc[idx - self.params[f"{k}_p"] : idx + self.params[f"{k}_f"] + 1]
                .to_numpy(copy=True)
            )

        return initial_window

    def update_window(self, window, updates):
        for k, next_val in updates.items():
            w = window[k]
            w[:-1] = w[1:]
            w[-1] = next_val
            window[k] = w

        return window

    def _as_list(self, x):
        if x is None:
            return []
        return x if isinstance(x, (list, tuple)) else [x]

    def _select_norm_data(self, train, val, test, source):
        if source == "train":
            return train
        elif source == "val":
            return val
        elif source == "test":
            return test
        else:
            raise ValueError(f"Unknown normalization source: {source}")

    def _ensure_2d(self, arr):
        if arr is None:
            return None
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr

    def _concat_safe(self, arrays, axis=1):
        arrays = [a for a in arrays if a is not None]

        if len(arrays) == 0:
            return None

        return np.concatenate(arrays, axis=axis)

    def get_normalizer(self, keys):
        if isinstance(keys, (list, tuple)):
            return [self.normalizer[k] for k in keys]
        else:
            return self.normalizer[keys]

    def plotSample(self, k):
        pass
