import pickle

import matplotlib.pyplot as plt
import numpy as np

from source.function.save_pathing import save_pathing

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
        # marker
        if s is not None:
            size = base_marker_size * (s / np.max(s))
        else:
            size = None

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
