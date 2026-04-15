"""ipywidgets UI for adjusting weights/biases of a `MultiLayerNetwork` and watching the
exact polyhedral decomposition update live.

Rendering strategy: the figure is built on an Agg canvas (fully independent of pyplot's display
machinery) and rasterized to a PNG that we push into an ``ipywidgets.Image``. That means:
  * No `plt.show()` / inline-backend double display (no duplicated figures).
  * Updates swap the image's bytes in place — the DOM image element is replaced atomically, so
    there is no visible blank frame between slider ticks (no flicker).
"""

from __future__ import annotations

from io import BytesIO
from typing import Sequence

import ipywidgets as widgets
import numpy as np
from IPython.display import display
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon

from .decomposition import build_dual_graph, compute_decomposition
from .network import MultiLayerNetwork
from .pp import to_json_string

_PALETTE = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F8C471", "#82E0AA", "#F1948A", "#AED6F1", "#A9DFBF",
    "#F9E79F", "#D7BDE2", "#A3E4D7", "#F5B7B1", "#FAD7A0",
]


def _region_color(pattern: tuple[int, ...]) -> str:
    """Stable color: hash the binary pattern so region colors persist as sliders move."""
    if not pattern:
        return _PALETTE[0]
    idx = 0
    for bit in pattern:
        idx = (idx * 2 + int(bit)) % (1 << 30)
    return _PALETTE[idx % len(_PALETTE)]


class PolyhedraWidget:
    """Interactive widget over a ReLU `MultiLayerNetwork`.

    Args:
        architecture: sizes of each layer, input → output. Ignored if ``state`` is given.
        state: optional dict (from `MultiLayerNetwork.to_state`) to seed weights/biases from.
        view_half: half-width of the square viewing window.
        slider_range: (min, max) for weight/bias sliders.
        slider_step: slider granularity.
        seed: RNG seed for initial random weights (only used when ``state`` is None).
        show_dual_graph: whether to render the dual graph alongside the decomposition.
        show_neuron_lines: whether to overlay the neuron activation boundaries.

    Example::

        from src import PolyhedraWidget
        w = PolyhedraWidget(architecture=[2, 4, 1])
        w.display()

        # Later, after tweaking sliders:
        state = w.state
        w2 = PolyhedraWidget(state=state)   # reproduce exactly
    """

    def __init__(
        self,
        architecture: Sequence[int] = (2, 4, 1),
        *,
        state: dict | None = None,
        view_half: float = 3.0,
        slider_range: tuple[float, float] = (-3.0, 3.0),
        slider_step: float = 0.1,
        seed: int | None = 0,
        show_dual_graph: bool = False,
        show_neuron_lines: bool = True,
    ):
        if state is not None:
            self.net = MultiLayerNetwork.from_state(state)
            self.architecture = list(self.net.architecture)
        else:
            if architecture[0] != 2:
                raise ValueError("PolyhedraWidget requires a 2-input network")
            self.architecture = list(architecture)
            rng = np.random.default_rng(seed) if seed is not None else None
            self.net = MultiLayerNetwork(self.architecture, rng=rng)

        if self.architecture[0] != 2:
            raise ValueError("PolyhedraWidget requires a 2-input network")

        self.view_half = view_half
        self.slider_range = slider_range
        self.slider_step = slider_step
        self.show_dual_graph = show_dual_graph
        self.show_neuron_lines = show_neuron_lines

        self._weight_sliders: list[list[list[widgets.FloatSlider]]] = []
        self._bias_sliders: list[list[widgets.FloatSlider]] = []
        self._suppress_callbacks = False

        self._setup_figure()
        self._image = widgets.Image(format="png")
        self._export_area = widgets.Textarea(
            value="",
            placeholder="Click 'Export weights & biases' to populate…",
            layout=widgets.Layout(width="340px", height="140px"),
        )
        self._info = widgets.HTML()

        self._build_ui()
        self._render_plot()

    # ------------------------------------------------------ public access

    @property
    def state(self) -> dict:
        """Round-trippable dict of the current weights/biases/architecture."""
        return self.net.to_state()

    # ------------------------------------------------------------- figure

    def _setup_figure(self) -> None:
        if self.show_dual_graph:
            w_in, h_in = 12.0, 6.0
            self._fig = Figure(figsize=(w_in, h_in), dpi=100)
            FigureCanvasAgg(self._fig)
            self._ax_decomp = self._fig.add_subplot(1, 2, 1)
            self._ax_dual = self._fig.add_subplot(1, 2, 2)
        else:
            w_in, h_in = 7.5, 7.5
            self._fig = Figure(figsize=(w_in, h_in), dpi=100)
            FigureCanvasAgg(self._fig)
            self._ax_decomp = self._fig.add_subplot(1, 1, 1)
            self._ax_dual = None

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        slider_panel = self._build_slider_panel()
        controls = self._build_action_buttons()
        right = widgets.VBox(
            [self._image, self._info],
            layout=widgets.Layout(overflow="visible"),
        )
        self._root = widgets.HBox(
            [widgets.VBox([controls, slider_panel],
                          layout=widgets.Layout(width="380px", flex="0 0 auto", overflow="visible")),
             right],
            layout=widgets.Layout(align_items="flex-start", overflow="visible"),
        )

    def _build_slider_panel(self) -> widgets.Widget:
        """Sliders are rendered for hidden layers only — the output layer is linear and does not
        affect the polyhedral decomposition, so it is not controllable from this UI."""
        layer_blocks = []
        # Hidden layers correspond to weight-matrix indices 0 .. layers-3.
        num_hidden = self.net.layers - 2
        for l in range(num_hidden):
            layer_name = f"Hidden {l + 1}"
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
        export_btn = widgets.Button(description="Export weights & biases", icon="download",
                                    layout=widgets.Layout(width="340px"))
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

        randomize_btn.on_click(lambda _: self.randomize())
        zero_btn.on_click(lambda _: self.zero())
        export_btn.on_click(lambda _: self._populate_export())

        def _on_view(change):
            self.view_half = float(change["new"])
            self._render_plot()

        def _on_lines(change):
            self.show_neuron_lines = bool(change["new"])
            self._render_plot()

        view_slider.observe(_on_view, names="value")
        lines_toggle.observe(_on_lines, names="value")

        return widgets.VBox([
            widgets.HBox([randomize_btn, zero_btn]),
            view_slider,
            lines_toggle,
            export_btn,
            self._export_area,
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
            # Only hidden-layer sliders exist; the output layer is intentionally unexposed.
            for l in range(len(self._weight_sliders)):
                for i in range(self.architecture[l + 1]):
                    for j in range(self.architecture[l]):
                        self._weight_sliders[l][i][j].value = float(self.net.weights[l][i, j])
                    self._bias_sliders[l][i].value = float(self.net.biases[l][i, 0])
        finally:
            self._suppress_callbacks = False

    def _populate_export(self) -> None:
        """Write the current network state into the export Textarea using the project's
        canonical JSON save format (matches `multilayer_network.json`)."""
        self._export_area.value = to_json_string(self.state)

    # ---------------------------------------------------------- plotting

    def _render_plot(self) -> None:
        v = self.view_half
        x_min, x_max, y_min, y_max = -v, v, -v, v
        decomp = compute_decomposition(self.net, x_min, x_max, y_min, y_max)
        n_regions = len(decomp.regions)

        ax = self._ax_decomp
        ax.clear()
        for region in decomp.regions:
            color = _region_color(region.pattern)
            ax.add_patch(MplPolygon(
                region.polygon, closed=True,
                facecolor=color, edgecolor="none", alpha=0.85,
            ))
        if self.show_neuron_lines:
            for segs in decomp.neuron_segments:
                for p, q in segs:
                    ax.plot([p[0], q[0]], [p[1], q[1]],
                            color="black", lw=1.0, solid_capstyle="round")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_title(f"Polyhedral decomposition  ({n_regions} regions)")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.axhline(0, color="#333", lw=0.4, zorder=0.5)
        ax.axvline(0, color="#333", lw=0.4, zorder=0.5)

        if self._ax_dual is not None:
            ax_dual = self._ax_dual
            ax_dual.clear()
            centroids, edges = build_dual_graph(decomp)
            for i, j in edges:
                ax_dual.plot([centroids[i, 0], centroids[j, 0]],
                             [centroids[i, 1], centroids[j, 1]],
                             color="#888", lw=1.0, zorder=1)
            for idx, region in enumerate(decomp.regions):
                ax_dual.scatter(centroids[idx, 0], centroids[idx, 1],
                                s=120, c=_region_color(region.pattern),
                                edgecolors="black", linewidths=1.0, zorder=2)
            ax_dual.set_xlim(x_min, x_max)
            ax_dual.set_ylim(y_min, y_max)
            ax_dual.set_aspect("equal")
            ax_dual.set_title(f"Dual graph  ({len(edges)} edges)")

        self._fig.tight_layout()

        buf = BytesIO()
        self._fig.savefig(buf, format="png")
        self._image.value = buf.getvalue()

        total_neurons = self.net.hidden_neuron_count
        self._info.value = (
            f"<div style='font-family:monospace'>"
            f"architecture = {self.architecture} &nbsp;|&nbsp; "
            f"hidden neurons = {total_neurons} &nbsp;|&nbsp; "
            f"regions = {n_regions} &nbsp;|&nbsp; "
            f"max (Zaslavsky) = {_zaslavsky_bound(total_neurons)}"
            f"</div>"
        )

    # -------------------------------------------------------------- show

    def display(self) -> None:
        display(self._root)

    def _ipython_display_(self) -> None:
        self.display()


def _zaslavsky_bound(h: int) -> int:
    """Max regions h lines in general position can induce in the plane: 1 + h + C(h,2)."""
    if h <= 0:
        return 1
    return 1 + h + h * (h - 1) // 2
