import pickle

import matplotlib.pyplot as plt
import numpy as np

from source.function.save_pathing import save_pathing

"""
# -------------------------------------------------------------------------
# Name:            Plotter.py
# Description:     systematic plotting tools for timeseries and scatter data, with save/load
#
# Author:          Antoine
# Collaborator:    Bona
# Date created:    06/2026
# Project:         surrogate_hydro
# -------------------------------------------------------------------------
"""

# TODO : implement parralllele corrdinate plot
# TODO : implement phase comparison plot


class Plotter:
    @staticmethod
    def _save(fig, save_path, name, suffix):
        """Save a PNG (for quick viewing) and a pickle (to reopen interactively later)."""
        png_path = save_pathing(save_path, name, suffix, ".png")
        fig.savefig(png_path, dpi=300)

        pkl_path = save_pathing(save_path, name, suffix, ".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(fig, f)

    @staticmethod
    def load_figure(pkl_path):
        """Reload a figure previously saved via _save, ready to plt.show() again."""
        with open(pkl_path, "rb") as f:
            fig = pickle.load(f)
        return fig

    @staticmethod
    def plot_lines(
        series: list,
        xlabel="",
        ylabel="",
        title=None,
        save_path=None,
        name=None,
        suffix="series",
    ):
        """
        Draw one line per (x, y, label) tuple in series. DataHandle decides what
        the data and labels mean; this just draws them.
        """
        plt.figure(figsize=(10, 6))

        for x, y, label in series:
            plt.plot(x, y, label=label)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()

        if save_path:
            Plotter._save(plt.gcf(), save_path, name, suffix)

    @staticmethod
    def plot_line(
        objs,
        x,
        y,
        normalize=False,
        xlabel=None,
        ylabel=None,
        title=None,
        save_path=None,
        name=None,
        suffix="line",
    ):
        """
        Generic line plotting tool: given a list of objects that each expose
        a tabular `.dataset` (column access by name) and a `.label` string,
        plot column `y` (or each column in a list `y`) against column `x`,
        one line per (object, variable) pair. This makes no assumption about
        what the data represents -- any class following the dataset/label
        convention can use it, not just one particular data type.
        """
        if not isinstance(objs, (list, tuple)):
            objs = [objs]

        variables = [y] if isinstance(y, str) else list(y)
        multi = len(variables) > 1

        series = []
        for obj in objs:
            if x not in obj.dataset.columns:
                print(f"Warning: '{x}' not recognized")
                continue
            x_data = obj.dataset[x]

            for var in variables:
                if var in obj.dataset.columns:
                    data = obj.dataset[var]

                    if normalize:
                        max_val = np.max(np.abs(data))
                        if max_val > 0:
                            data = data / max_val

                    label = f"{var} - {obj.label}" if multi else obj.label
                    series.append((x_data, data, label))
                else:
                    print(f"Warning: '{var}' not recognized")

        Plotter.plot_lines(
            series,
            xlabel=xlabel if xlabel is not None else x,
            ylabel=ylabel
            if ylabel is not None
            else ("Normalized value" if normalize else "Value"),
            title=title,
            save_path=save_path,
            name=name,
            suffix=suffix,
        )

    @staticmethod
    def plot_grid(
        x,
        y,
        s=None,
        c=None,
        base_marker_size=200,
        xlabel="",
        ylabel="",
        clabel="",
        title=None,
        save_path=None,
        name=None,
        suffix="grid",
    ):
        # marker — cap size to something readable
        if s is not None:
            size = np.clip(base_marker_size * (s / np.max(s)), 30, 200)
        else:
            size = 60

        plt.figure()

        sc = plt.scatter(
            x,
            y,
            s=size,
            c=c,
        )

        # colorbar (only meaningful if a color dimension was actually given)
        if c is not None:
            cbar = plt.colorbar(sc)
            cbar.set_label(clabel)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid()

        if save_path:
            Plotter._save(plt.gcf(), save_path, name, suffix)

    @staticmethod
    def plot_histogram(
        series: list,
        xlabel="",
        ylabel="",
        title=None,
        bins=50,
        density=True,
        alpha=0.5,
        save_path=None,
        name=None,
        suffix="hist",
    ):
        """
        Draw one histogram per (data, label) tuple in series. DataHandle
        decides what the data and labels mean; this just draws them.
        """
        plt.figure(figsize=(10, 6))

        for data, label in series:
            plt.hist(data, bins=bins, density=density, alpha=alpha, label=label)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()

        if save_path:
            Plotter._save(plt.gcf(), save_path, name, suffix)

    @staticmethod
    def plot_scatter3d(
        series: list,
        xlabel="",
        ylabel="",
        zlabel="",
        title=None,
        save_path=None,
        name=None,
        suffix="scatter3d",
    ):
        """
        Draw one 3D scatter series per (x, y, z, label) tuple in series.
        DataHandle decides what the data and labels mean; this just draws them.
        """
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

        for x, y, z, label in series:
            ax.scatter(x, y, z, label=label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        ax.legend()

        if save_path:
            Plotter._save(fig, save_path, name, suffix)

    @staticmethod
    def scatter3d(
        objs,
        x,
        y,
        z,
        normalize=False,
        xlabel=None,
        ylabel=None,
        zlabel=None,
        title=None,
        save_path=None,
        name=None,
        suffix="scatter3d",
    ):
        """
        Generic 3D scatter tool: given a list of objects that each expose a
        tabular `.dataset` (column access by name) and a `.label` string,
        scatter columns `x`, `y`, `z` against each other, one series per
        object. Same dataset/label convention as Plotter.plot_line -- not
        tied to any particular data type.
        """
        if not isinstance(objs, (list, tuple)):
            objs = [objs]

        def _norm(data):
            if not normalize:
                return data
            max_val = np.max(np.abs(data))
            return data / max_val if max_val > 0 else data

        series = []
        for obj in objs:
            cols = obj.dataset.columns
            if x in cols and y in cols and z in cols:
                series.append(
                    (
                        _norm(obj.dataset[x]),
                        _norm(obj.dataset[y]),
                        _norm(obj.dataset[z]),
                        obj.label,
                    )
                )
            else:
                print(f"Warning: one of '{x}', '{y}', '{z}' not recognized")

        Plotter.plot_scatter3d(
            series,
            xlabel=xlabel if xlabel is not None else x,
            ylabel=ylabel if ylabel is not None else y,
            zlabel=zlabel if zlabel is not None else z,
            title=title,
            save_path=save_path,
            name=name,
            suffix=suffix,
        )

    # %% EXTENDED METHODS FOR METRIC PLOTS

    @staticmethod
    def plot_multi_scatter(
        per_model: dict,
        vmin: float,
        vmax: float,
        xlabel="Te [s]",
        ylabel="Hs [m]",
        clabel="",
        title=None,
        save_path=None,
        name=None,
        suffix="multi_scatter",
    ):
        """
        Multiple scatter subplots in one figure with a shared colorbar.
        Designed for Te×Hs metric grids across models.

        Parameters
        ----------
        per_model : {model_label: {"x": arr, "y": arr, "s": arr, "c": arr}}
            x, y   -- scatter coordinates (Te, Hs)
            s      -- marker sizes (proportional to wave energy J)
            c      -- color values (metric)
        vmin, vmax : unified color scale across all subplots
        """
        models = list(per_model.keys())
        n = len(models)
        if n == 0:
            return

        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        sc = None
        for col, label in enumerate(models):
            ax = axes[0, col]
            d = per_model[label]
            sc = ax.scatter(
                d["x"],
                d["y"],
                s=60,
                c=d["c"],
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(label[:35], fontsize=8)
            ax.grid(True)

        if sc is not None:
            fig.colorbar(sc, ax=axes[0, :].tolist(), label=clabel)

        if title:
            fig.suptitle(title)
        plt.tight_layout()
        if save_path:
            Plotter._save(fig, save_path, name, suffix)
        # do NOT close — let plt.show() in run_plots display it

    @staticmethod
    def plot_grouped_bars(
        group_labels: list,
        series: list,
        ylabel="",
        title=None,
        normalize=False,
        save_path=None,
        name=None,
        suffix="grouped_bars",
    ):
        """
        Grouped bar chart. One group of bars per entry in group_labels.
        Multiple series shown side-by-side with different colors.

        Parameters
        ----------
        group_labels : list of str
        series       : list of (label, values) tuples
        normalize    : if True, each series is divided by its max absolute
                       value so different units are comparable on one axis.
        """
        n_groups = len(group_labels)
        n_series = len(series)
        width = 0.8 / max(n_series, 1)
        x = np.arange(n_groups)
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_series, 1)))

        fig, ax = plt.subplots(figsize=(max(6, 2 * n_groups), 5))
        for i, (label, values) in enumerate(series):
            vals = np.array(values, dtype=float)
            if normalize:
                vmax = np.nanmax(np.abs(vals))
                if vmax > 0:
                    vals = vals / vmax
                lbl = f"{label} (norm)"
            else:
                lbl = label
            ax.bar(
                x + i * width, vals, width=width, color=colors[i], label=lbl, alpha=0.85
            )

        ax.set_xticks(x + width * (n_series - 1) / 2)
        ax.set_xticklabels([l[:20] for l in group_labels], rotation=30, ha="right")
        ax.set_ylabel("normalised value" if normalize else ylabel)
        if title:
            ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y")
        plt.tight_layout()

        if save_path:
            Plotter._save(fig, save_path, name, suffix)
        plt.close(fig)

    @staticmethod
    def plot_subplots_bars(
        subplot_titles: list,
        group_labels: list,
        subplot_series: list,
        ncols: int = 4,
        ylabel="",
        suptitle=None,
        normalize=False,
        save_path=None,
        name=None,
        suffix="subplots_bars",
    ):
        """
        Grid of grouped-bar subplots. One subplot per case/group.
        Maximum 4 columns; rows expand as needed.
        Figure height scales with number of rows.

        Parameters
        ----------
        normalize : if True each series normalised by its max abs value.
        """
        n = len(subplot_titles)
        ncols = min(n, max(1, min(ncols, 4)))  # hard cap at 4
        nrows = (n + ncols - 1) // ncols
        n_ser = max(len(s) for s in subplot_series) if subplot_series else 1
        width = 0.8 / max(n_ser, 1)
        x = np.arange(len(group_labels))
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_ser, 1)))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 4 * nrows),
            squeeze=False,
        )

        legend_labels = []
        for idx, (suptit, series) in enumerate(zip(subplot_titles, subplot_series)):
            ax = axes[idx // ncols][idx % ncols]
            for i, (label, values) in enumerate(series):
                vals = np.array(values, dtype=float)
                if normalize:
                    vmax = np.nanmax(np.abs(vals))
                    if vmax > 0:
                        vals = vals / vmax
                    lbl = f"{label} (norm)"
                else:
                    lbl = label
                ax.bar(
                    x + i * width,
                    vals,
                    width=width,
                    color=colors[i],
                    label=lbl if idx == 0 else "_",
                    alpha=0.85,
                )
                if idx == 0:
                    legend_labels.append(lbl)
            ax.set_xticks(x + width * (len(series) - 1) / 2)
            ax.set_xticklabels([l[:15] for l in group_labels], rotation=30, ha="right")
            ax.set_title(suptit[:40], fontsize=8)
            ax.set_ylabel("norm." if normalize else ylabel)
            ax.grid(axis="y")

        if legend_labels:
            axes[0, 0].legend(fontsize=7)

        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        if suptitle:
            fig.suptitle(suptitle)
        plt.tight_layout()
        if save_path:
            Plotter._save(fig, save_path, name, suffix)
        plt.close(fig)
