import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from source.classes.DNN import DNN
from source.function.metricError import metricError
from source.function.generate_dnn_structure import generate_dnn_structure

# TODO : pbl with num_worker > 0
# TODO : introduce no validation
# TODO : GPU


class Estimator:
    @staticmethod
    def predict_norm(model, X_norm, norm):
        # predict
        y_pred = model(X_norm)
        y_pred_np = y_pred.cpu().numpy()

        # unormalize
        y_pred_np = norm.inverse_transform(y_pred_np)[0, 0]

        return y_pred_np

    def __init__(
        self,
        model_type,
        params,
        train_data,
        val_data,
        t_split=None,
        wave=None,
        verbose=0,
    ):
        self.model_type = model_type
        self.train_X = torch.as_tensor(train_data[0], dtype=torch.float32)
        self.train_y = torch.as_tensor(train_data[1], dtype=torch.float32)
        self.val_X = torch.as_tensor(val_data[0], dtype=torch.float32)
        self.val_y = torch.as_tensor(val_data[1], dtype=torch.float32)
        self.params = params
        self.wave = wave
        self.t_split = t_split
        self.verbose = verbose

    def build(self):
        model_types = {"dnn": self.build_dnn}

        if self.model_type in model_types:
            self.model = model_types[self.model_type]()  # call the model type method
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def build_dnn(self):
        def run_epoch(epoch_idx):
            ep_tloss = 0.0
            ep_avtloss = 0.0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch_idx + 1}/{self.params.setdefault('epochs', 25)} [train]",
                leave=False,
            )

            for x, y in pbar:
                optimizer.zero_grad()
                y_pred = self.model(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()

                # Gather data and report
                ep_tloss += loss.item() * x.size(0)  #!

                # update progress bar
                pbar.set_postfix(batch_loss=loss.item())

            ep_avtloss = ep_tloss / len(train_loader.dataset)

            return ep_avtloss

        # add implicite parameters
        self.params["input_size"] = self.train_X.shape[1]
        self.params["output_size"] = self.train_y.shape[1]
        self.params["hidden_layers"] = generate_dnn_structure(
            self.params.setdefault("layer", 3),
            self.params.setdefault("mid_neurons", 10),
            self.params.setdefault("min_neurons", 5),
        )

        # build dnn
        self.params.setdefault("dropout", 0.0)
        self.params.setdefault("activation", "relu")
        self.model = DNN(self.params)

        # log
        self.log = {
            "val loss": [],
            "train loss": [],
        }

        # convert data #!shuffle, num_worker,
        train_set = TensorDataset(self.train_X, self.train_y)
        val_set = TensorDataset(self.val_X, self.val_y)
        train_loader = DataLoader(
            train_set,
            batch_size=self.params.setdefault("batch_size", 128),
            shuffle=True,
            # num_workers=0,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.params.setdefault("batch_size", 128),
            shuffle=False,
            # num_workers=0,
        )

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.params.setdefault("lr", 1e-3)
        )

        best_tloss = float("inf")
        best_vloss = float("inf")
        patience_counter = 0
        epoch_counter = 0

        # training
        for epoch in range(self.params.setdefault("epochs", 25)):
            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            ep_avtloss = run_epoch(epoch)
            self.log["train loss"].append(ep_avtloss)

            # We don't need gradients on to do reporting
            self.model.eval()
            ep_vloss = 0

            # validation
            with torch.no_grad():
                for x, y in val_loader:
                    y_pred = self.model(x)
                    loss = loss_fn(y_pred, y)
                    ep_vloss += loss.item() * x.size(0)
                ep_avvloss = ep_vloss / len(val_loader.dataset)
                self.log["val loss"].append(ep_avvloss)

            if self.verbose > 2:
                print(
                    f"Epoch {epoch + 1}/{self.params.setdefault('epochs', 25)} | "
                    f"train_loss={ep_avtloss:.6f} | "
                    f"val_loss={ep_avvloss:.6f}"
                )

            # early stopping
            if ep_avvloss < best_vloss:
                best_vloss = ep_avvloss
                best_tloss = ep_avtloss
                best_model_state = self.model.state_dict()
                patience_counter = 0

                if self.verbose > 1:
                    print(
                        f"\n✔ New best model at epoch {epoch + 1} | "
                        f"train_loss={best_tloss:.6f} | "
                        f"val_loss={best_vloss:.6f} |\n"
                    )
            else:
                patience_counter += 1
                if patience_counter >= self.params.setdefault("patience", 5):
                    if self.verbose > 1:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            epoch_counter += 1

        self.log["best val loss"] = best_vloss
        self.log["best train loss"] = best_tloss
        self.log["final epoch"] = epoch + 1

        self.model.load_state_dict(best_model_state)
        if self.verbose > 0:
            print(
                f"Training finished at epoch {epoch + 1} | "
                f"val_loss={best_vloss:.6f} | "
                f"train_loss={best_tloss:.6f}"
            )

        return self.model

    def evaluate(
        self,
        X,
        y,
        metric_fn=None,
        plot=False,
        log=None,
        time=None,
        norm=None,
        verbose=False,
    ):
        self.model.eval()

        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)

        loss_fn = nn.MSELoss()

        with torch.no_grad():
            y_pred = self.model(X)
            loss = loss_fn(y_pred, y).item()

        y_np = y.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        # normalize
        if norm is not None:
            y_np = norm.inverse_transform(y_np)
            y_pred_np = norm.inverse_transform(y_pred_np)

        # Extra metric
        if metric_fn is not None:
            metric_value = metricError(y_pred_np, y_np, error_type=metric_fn)

        # log
        if log is not None:
            self.log[log + " loss"] = loss
            if metric_fn is not None:
                self.log[log + "_" + metric_fn] = metric_value

        # print
        if verbose:
            if metric_fn is not None:
                print(f"loss = {loss:.6f} | {metric_fn} = {metric_value:.6f}")
            else:
                print(f"loss = {loss:.6f}")

        # plot
        if plot is True:
            plt.figure(figsize=(6, 4))
            if time is not None:
                plt.plot(time, y_np, label="true")
                plt.plot(time, y_pred_np, label="predicted")
                plt.xlabel("time")
            else:
                plt.plot(y_np, label="true")
                plt.plot(y_pred_np, label="predicted")
                plt.xlabel("-")
            plt.ylabel("output")

            if metric_fn is not None:
                plt.title(f"loss = {loss:.6f} | {metric_fn} = {metric_value:.6f}")
            else:
                plt.title(f"loss = {loss:.6f}")

            plt.legend()
            plt.tight_layout()
            plt.show()

        return y_pred_np

    def plot_training(self):
        if "train loss" not in self.log or "val loss" not in self.log:
            raise ValueError("No training history found")

        plt.figure(figsize=(6, 4))
        plt.plot(self.log["train loss"], label="Train loss")
        plt.plot(self.log["val loss"], label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")  # optional but often useful
        plt.legend()
        plt.grid(True)
        plt.title("Training history")
        plt.tight_layout()
        plt.show()

    def save_model(self, path):
        """
        Save model weights and metadata.
        """
        path = Path(path + ".pt")
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_type": self.model_type,
            "model_state_dict": self.model.state_dict(),
            "params": self.params,
            "log": self.log,
        }

        torch.save(checkpoint, path)

    def save_estimator(self, path):
        path = Path(path + ".pkl")
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)
