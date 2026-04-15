"""ipywidgets UI for adjusting weights/biases of a `MultiLayerNetwork` and watching the
exact polyhedral decomposition update live.
"""

from __future__ import annotations

from typing import Sequence

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from matplotlib.patches import Polygon as MplPolygon

from .decomposition import build_dual_graph, compute_decomposition
from .network import MultiLayerNetwork

_PALETTE = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F8C471", "#82E0AA", "#F1948A", "#AED6F1", "#A9DFBF",
    "#F9E79F", "#D7BDE2", "#A3E4D7", "#F5B7B1", "#FAD7A0",
]


def _region_color(pattern: tuple[int, ...]) -> str:
    """Stable color: hash the binary pattern so region colors persist as sliders move (as long
    as the pattern itself exists)."""
    if not pattern:
        return _PALETTE[0]
    idx = 0
    for bit in pattern:
        idx = (idx * 2 + int(bit)) % (1 << 30)
    return _PALETTE[idx % len(_PALETTE)]


class PolyhedraWidget:
    """Build a network with the given architecture, render one slider per weight/bias, and a
    matplotlib panel showing the *exact* polyhedral decomposition that updates on every slider
    change.

    The architecture must have input size 2 (the decomposition is a partition of the 2-D input
    plane).

    Example::

        from src import PolyhedraWidget
        w = PolyhedraWidget(architecture=[2, 4, 1])
        w.display()
    """

    def __init__(
        self,
        architecture: Sequence[int] = (2, 4, 1),
        view_half: float = 3.0,
        slider_range: tuple[float, float] = (-3.0, 3.0),
        slider_step: float = 0.1,
        seed: int | None = 0,
        show_dual_graph: bool = True,
        show_neuron_lines: bool = True,
    ):
        if architecture[0] != 2:
            raise ValueError("PolyhedraWidget requires a 2-input network for the 2-D decomposition")
        self.architecture = list(architecture)
        self.view_half = view_half
        self.slider_range = slider_range
        self.slider_step = slider_step
        self.show_dual_graph = show_dual_graph
        self.show_neuron_lines = show_neuron_lines

        rng = np.random.default_rng(seed) if seed is not None else None
        self.net = MultiLayerNetwork(self.architecture, rng=rng)

        self._weight_sliders: list[list[list[widgets.FloatSlider]]] = []
        self._bias_sliders: list[list[widgets.FloatSlider]] = []
        self._suppress_callbacks = False

        self._plot_output = widgets.Output()
        self._info = widgets.HTML()

        self._build_ui()
        self._render_plot()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        slider_panel = self._build_slider_panel()
        controls = self._build_action_buttons()
        right = widgets.VBox([self._plot_output, self._info])
        self._root = widgets.HBox(
            [widgets.VBox([controls, slider_panel],
                          layout=widgets.Layout(width="380px", overflow="auto", max_height="720px")),
             right]
        )

    def _build_slider_panel(self) -> widgets.Widget:
        layer_blocks = []
        for l in range(self.net.layers - 1):
            layer_name = "Output" if l == self.net.layers - 2 else f"Hidden {l + 1}"
            header = widgets.HTML(
                f"<b>{layer_name} Layer</b> "
                f"<span style='color:#888'>(W<sup>{l+1}</sup>: "
                f"{self.architecture[l+1]}×{self.architecture[l]}, "
                f"b<sup>{l+1}</sup>: {self.architecture[l+1]})</span>"
            )

            w_rows: list[list[widgets.FloatSlider]] = []
            w_widgets = []
            for i in range(self.architecture[l + 1]):
                row = []
                for j in range(self.architecture[l]):
                    s = widgets.FloatSlider(
                        value=float(self.net.weights[l][i, j]),
                        min=self.slider_range[0],
                        max=self.slider_range[1],
                        step=self.slider_step,
                        description=f"W{l+1}[{i+1},{j+1}]",
                        continuous_update=True,
                        readout_format=".2f",
                        style={"description_width": "70px"},
                        layout=widgets.Layout(width="320px"),
                    )
                    s.observe(self._on_weight_change(l, i, j), names="value")
                    row.append(s)
                    w_widgets.append(s)
                w_rows.append(row)
            self._weight_sliders.append(w_rows)

            b_widgets = []
            b_row: list[widgets.FloatSlider] = []
            for i in range(self.architecture[l + 1]):
                s = widgets.FloatSlider(
                    value=float(self.net.biases[l][i, 0]),
                    min=self.slider_range[0],
                    max=self.slider_range[1],
                    step=self.slider_step,
                    description=f"b{l+1}[{i+1}]",
                    continuous_update=True,
                    readout_format=".2f",
                    style={"description_width": "70px"},
                    layout=widgets.Layout(width="320px"),
                )
                s.observe(self._on_bias_change(l, i), names="value")
                b_row.append(s)
                b_widgets.append(s)
            self._bias_sliders.append(b_row)

            layer_blocks.append(widgets.VBox([
                header,
                widgets.HTML("<i>Weights</i>"),
                widgets.VBox(w_widgets),
                widgets.HTML("<i>Biases</i>"),
                widgets.VBox(b_widgets),
                widgets.HTML("<hr>"),
            ]))

        return widgets.VBox(layer_blocks)

    def _build_action_buttons(self) -> widgets.Widget:
        randomize_btn = widgets.Button(description="Randomize", icon="random")
        zero_btn = widgets.Button(description="Zero", icon="circle")
        view_slider = widgets.FloatSlider(
            value=self.view_half, min=0.5, max=10.0, step=0.5,
            description="View ±", continuous_update=False,
            style={"description_width": "80px"},
            layout=widgets.Layout(width="320px"),
        )
        lines_toggle = widgets.Checkbox(
            value=self.show_neuron_lines, description="Draw neuron boundaries",
            indent=False,
        )
        dual_toggle = widgets.Checkbox(
            value=self.show_dual_graph, description="Show dual graph", indent=False,
        )

        randomize_btn.on_click(lambda _: self.randomize())
        zero_btn.on_click(lambda _: self.zero())

        def _on_view(change):
            self.view_half = float(change["new"])
            self._render_plot()

        def _on_lines(change):
            self.show_neuron_lines = bool(change["new"])
            self._render_plot()

        def _on_dual(change):
            self.show_dual_graph = bool(change["new"])
            self._render_plot()

        view_slider.observe(_on_view, names="value")
        lines_toggle.observe(_on_lines, names="value")
        dual_toggle.observe(_on_dual, names="value")

        return widgets.VBox([
            widgets.HBox([randomize_btn, zero_btn]),
            view_slider,
            lines_toggle,
            dual_toggle,
        ])

    # --------------------------------------------------------- callbacks

    def _on_weight_change(self, l: int, i: int, j: int):
        def _cb(change):
            if self._suppress_callbacks:
                return
            self.net.weights[l][i, j] = float(change["new"])
            self._render_plot()
        return _cb

    def _on_bias_change(self, l: int, i: int):
        def _cb(change):
            if self._suppress_callbacks:
                return
            self.net.biases[l][i, 0] = float(change["new"])
            self._render_plot()
        return _cb

    # --------------------------------------------------- network actions

    def randomize(self) -> None:
        rng = np.random.default_rng()
        for l in range(self.net.layers - 1):
            self.net.weights[l] = (rng.random(self.net.weights[l].shape) - 0.5) * 2.0
            self.net.biases[l] = (rng.random(self.net.biases[l].shape) - 0.5) * 2.0
        self._sync_sliders_from_net()
        self._render_plot()

    def zero(self) -> None:
        for l in range(self.net.layers - 1):
            self.net.weights[l][...] = 0.0
            self.net.biases[l][...] = 0.0
        self._sync_sliders_from_net()
        self._render_plot()

    def _sync_sliders_from_net(self) -> None:
        self._suppress_callbacks = True
        try:
            for l in range(self.net.layers - 1):
                for i in range(self.architecture[l + 1]):
                    for j in range(self.architecture[l]):
                        self._weight_sliders[l][i][j].value = float(self.net.weights[l][i, j])
                    self._bias_sliders[l][i].value = float(self.net.biases[l][i, 0])
        finally:
            self._suppress_callbacks = False

    # ---------------------------------------------------------- plotting

    def _render_plot(self) -> None:
        v = self.view_half
        x_min, x_max, y_min, y_max = -v, v, -v, v
        decomp = compute_decomposition(self.net, x_min, x_max, y_min, y_max)
        n_regions = len(decomp.regions)

        with self._plot_output:
            self._plot_output.clear_output(wait=True)
            ncols = 2 if self.show_dual_graph else 1
            fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
            if ncols == 1:
                axes = [axes]
            ax_decomp, *rest = axes

            # Fill every polygon.
            for region in decomp.regions:
                color = _region_color(region.pattern)
                patch = MplPolygon(
                    region.polygon, closed=True,
                    facecolor=color, edgecolor="none", alpha=0.85,
                )
                ax_decomp.add_patch(patch)

            # Draw neuron boundary segments on top.
            if self.show_neuron_lines:
                for segs in decomp.neuron_segments:
                    for p, q in segs:
                        ax_decomp.plot(
                            [p[0], q[0]], [p[1], q[1]],
                            color="black", lw=1.0, solid_capstyle="round",
                        )

            ax_decomp.set_xlim(x_min, x_max)
            ax_decomp.set_ylim(y_min, y_max)
            ax_decomp.set_aspect("equal")
            ax_decomp.set_title(f"Polyhedral decomposition  ({n_regions} regions)")
            ax_decomp.set_xlabel(r"$x_1$")
            ax_decomp.set_ylabel(r"$x_2$")
            ax_decomp.axhline(0, color="#333", lw=0.4, zorder=0.5)
            ax_decomp.axvline(0, color="#333", lw=0.4, zorder=0.5)

            if self.show_dual_graph:
                ax_dual = rest[0]
                centroids, edges = build_dual_graph(decomp)
                for i, j in edges:
                    ax_dual.plot(
                        [centroids[i, 0], centroids[j, 0]],
                        [centroids[i, 1], centroids[j, 1]],
                        color="#888", lw=1.0, zorder=1,
                    )
                for idx, region in enumerate(decomp.regions):
                    ax_dual.scatter(
                        centroids[idx, 0], centroids[idx, 1],
                        s=120, c=_region_color(region.pattern),
                        edgecolors="black", linewidths=1.0, zorder=2,
                    )
                ax_dual.set_xlim(x_min, x_max)
                ax_dual.set_ylim(y_min, y_max)
                ax_dual.set_aspect("equal")
                ax_dual.set_title(f"Dual graph  ({len(edges)} edges)")

            fig.tight_layout()
            plt.show()
            plt.close(fig)

        total_neurons = self.net.hidden_neuron_count
        self._info.value = (
            f"<div style='font-family:monospace'>"
            f"architecture = {self.architecture} &nbsp;|&nbsp; "
            f"hidden neurons = {total_neurons} &nbsp;|&nbsp; "
            f"regions = {n_regions} &nbsp;|&nbsp; "
            f"theoretical max (Zaslavsky) = {_zaslavsky_bound(total_neurons)}"
            f"</div>"
        )

    # -------------------------------------------------------------- show

    def display(self) -> None:
        display(self._root)

    def _ipython_display_(self) -> None:
        self.display()


def _zaslavsky_bound(h: int) -> int:
    """Max number of regions h lines in general position can induce in the plane: 1 + h + C(h,2)."""
    if h <= 0:
        return 1
    return 1 + h + h * (h - 1) // 2
