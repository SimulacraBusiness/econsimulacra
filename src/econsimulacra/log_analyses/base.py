from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, Optional, TypeAlias, TypeVar, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rich.console import Console, RenderableType
from rich.panel import Panel

from .records import AgentGenerationRecord
from .store import RecordStore

TimeKey: TypeAlias = int | datetime
T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True)
class TimeAxisConfig:
    """Configuration for a shared simulation-step x-axis.

    Produced by :meth:`AnalyzerBase._prepare_time_axis` and consumed by
    :meth:`AnalyzerBase._apply_time_axis`. All figures produced during one
    analysis pass share the same instance, guaranteeing consistent x-axis
    extents and tick positions across figures.

    Attributes:
        x_min (float): Minimum x value (earliest ``time_step`` in the store).
        x_max (float): Maximum x value (latest ``time_step`` in the store).
        tick_positions (list[float]): Evenly-spaced tick positions (as
            ``time_step`` values) sampled from the sorted global step list.
        tick_labels (list[str]): Human-readable labels for each tick.
            Uses ``datetime`` strings formatted as ``%Y-%m-%d\\n %H:%M`` when
            datetime values are available and monotonically non-decreasing;
            falls back to plain step numbers otherwise.
    """

    x_min: float
    x_max: float
    tick_positions: list[float]
    tick_labels: list[str]


class AnalyzerBase(ABC, Generic[T, U]):
    """Base class for all log analyzers in EconSimulacra.

    Each analyzer transforms a :class:`RecordStore` into a typed result
    ``T`` (single run) and ``U`` (multi-run aggregate). The standard
    analysis pipeline is:

    1. Call :meth:`analyze` with a :class:`RecordStore` to obtain a result
       of type ``T``.
    2. Call :meth:`draw_figs` to produce Matplotlib figures from that result.
    3. Optionally call :meth:`build_summary` to render a Rich-formatted
       terminal summary.

    For multi-run experiments, use :meth:`analyze_stores` and
    :meth:`draw_figs_all` instead.

    **Shared time axis**

    Subclasses that produce time-series plots should call
    :meth:`_prepare_time_axis` at the start of :meth:`analyze`. This reads
    all ``time_step`` values from the store and stores a
    :class:`TimeAxisConfig`. Subclasses then call :meth:`_apply_time_axis`
    on each :class:`~matplotlib.axes.Axes` when drawing figures. This ensures
    that plots from different analyzers share the same x-axis range and tick
    labels, making side-by-side comparison straightforward.

    Attributes:
        name (str): Unique analyzer name. Used as a dict key in result
            dicts returned by :class:`AnalysisManager` and as a prefix for
            saved figure file names.

    Type Parameters:
        T: Result type returned by :meth:`analyze` for a single store.
        U: Result type returned by :meth:`analyze_stores` for multiple stores.
    """

    name: str

    _time_axis_config: Optional[TimeAxisConfig] = None

    @abstractmethod
    def analyze(self, store: RecordStore) -> T:
        """Analyse records in *store* and return a structured result.

        Subclasses must implement this method. The result type ``T`` is
        defined by the subclass and should contain all intermediate data
        needed by :meth:`draw_figs` and :meth:`build_summary`.

        Args:
            store (RecordStore): The record store to analyse.

        Returns:
            T: Structured analysis result.
        """

    @abstractmethod
    def analyze_stores(self, stores: list[RecordStore]) -> U:
        """Analyse a list of stores and return an aggregate result.

        This is the multi-run counterpart of :meth:`analyze`. The aggregate
        result type ``U`` is defined by the subclass and may differ from
        ``T``.

        Args:
            stores (list[RecordStore]): One store per simulation run.

        Returns:
            U: Aggregated analysis result across all stores.
        """
        pass

    @abstractmethod
    def draw_figs(self, result: T) -> dict[str, Figure]:
        """Draw diagnostic figures from a single-run analysis result.

        Args:
            result (T): Output of :meth:`analyze`.

        Returns:
            dict[str, Figure]: Mapping from figure name to Matplotlib figure.
        """

    @abstractmethod
    def draw_figs_all(self, individual_results: list[T]) -> dict[str, Figure]:
        """Draw diagnostic figures across multiple runs.

        Args:
            individual_results (list[T]): One result per store, each
                produced by :meth:`analyze`.

        Returns:
            dict[str, Figure]: Mapping from figure name to Matplotlib figure.
        """
        return {}

    def build_summary(self, result: T) -> RenderableType:
        """Build a rich-renderable summary of the analysis result.

        Subclasses can override this method.
        """
        return Panel.fit(
            f"No summary is implemented for [bold]{self.name}[/bold].",
            title="Analysis Summary",
            border_style="yellow",
        )

    def summarize_results(
        self,
        result: T,
        console: Optional[Console] = None,
    ) -> None:
        """Print the analysis summary using rich."""
        console = console or Console()
        console.print(self.build_summary(result))

    def build_summary_all(self, results: U) -> RenderableType:
        """Build a rich-renderable summary for multiple stores.

        Subclasses can override this method.
        """
        return Panel.fit(
            f"No multi-store summary is implemented for [bold]{self.name}[/bold].",
            title="Multi-Store Analysis Summary",
            border_style="yellow",
        )

    def summarize_results_all(
        self,
        results: U,
        console: Optional[Console] = None,
    ) -> None:
        """Summarize results for multiple stores."""
        console = console or Console()
        console.print(self.build_summary_all(results))

    def _prepare_time_axis(
        self,
        store: RecordStore,
        *,
        max_ticks: int = 10,
    ) -> None:
        """Prepare a shared x-axis from the simulation steps in *store*.

        The x-coordinate is always the integer ``time_step``, so that plots
        are consistent even when ``datetime`` values are absent or
        discontinuous. Datetime values are used only as tick *labels*.

        Tick positions are chosen by linearly spacing ``max_ticks`` indices
        over the sorted list of unique steps (using
        :func:`numpy.linspace`). When datetime labels are available and
        monotonically non-decreasing they are formatted as
        ``%Y-%m-%d\\n %H:%M``; otherwise, plain step numbers are used.

        The result is stored in ``self._time_axis_config`` and applied to
        axes by :meth:`_apply_time_axis`.

        Args:
            store (RecordStore): Record store from which ``time_step`` and
                ``time`` fields are extracted.
            max_ticks (int): Maximum number of tick marks on the x-axis.
                Must be positive.

        Raises:
            ValueError: If ``max_ticks`` is not positive.
        """
        if max_ticks <= 0:
            raise ValueError("max_ticks must be positive.")

        step2time: dict[int, datetime] = {}
        step2fallback: set[int] = set()

        for record in store.records:
            maybe_step: object = getattr(record, "time_step", None)
            maybe_time: object = getattr(record, "time", None)
            if maybe_step == -1:
                continue

            if not isinstance(maybe_step, int):
                continue

            step2fallback.add(maybe_step)

            if isinstance(maybe_time, datetime):
                step2time[maybe_step] = maybe_time

        sorted_steps: list[int] = sorted(step2time)

        # Check datetime monotonicity. If broken, fall back to step labels.
        datetime_values: list[datetime] = [
            cast(datetime, step2time[step])
            for step in sorted_steps
            if isinstance(step2time[step], datetime)
        ]

        use_datetime_labels: bool = True
        if len(datetime_values) >= 2:
            use_datetime_labels = all(
                left <= right
                for left, right in zip(
                    datetime_values[:-1],
                    datetime_values[1:],
                    strict=True,
                )
            )

        min_step: int = sorted_steps[0]
        max_step: int = sorted_steps[-1]

        num_ticks: int = min(max_ticks, len(sorted_steps))
        tick_indices: list[int] = [
            int(round(value))
            for value in np.linspace(
                0,
                len(sorted_steps) - 1,
                num=num_ticks,
            )
        ]
        tick_indices = sorted(set(tick_indices))

        tick_steps: list[int] = [sorted_steps[index] for index in tick_indices]

        if use_datetime_labels:
            sorted_steps = sorted(step2fallback)
            tick_labels: list[str] = [
                self._format_time_key(step2time[step])
                if step in step2time
                else str(step)
                for step in tick_steps
            ]
        else:
            tick_labels = [str(step) for step in tick_steps]

        self._time_axis_config = TimeAxisConfig(
            x_min=float(min_step),
            x_max=float(max_step),
            tick_positions=[float(step) for step in tick_steps],
            tick_labels=tick_labels,
        )

    def _time_to_x(self, time_key: TimeKey) -> float:
        """Convert a time key to a numeric x-coordinate.

        Args:
            time_key: Integer time step or datetime.

        Returns:
            Numeric x-coordinate used for plotting.

        Raises:
            ValueError: If a datetime key is passed but no prepared time axis is
                available.
        """
        if isinstance(time_key, int):
            return float(time_key)

        if self._time_axis_config is None:
            raise ValueError(
                "Datetime x-values require _prepare_time_axis to be called first."
            )
        return float(time_key.timestamp())

    def _times_to_x(self, time_keys: list[TimeKey]) -> list[float]:
        """Convert time keys to numeric x-coordinates.

        Args:
            time_keys: List of integer or datetime time keys.

        Returns:
            Numeric x-coordinate list.
        """
        return [self._time_to_x(time_key) for time_key in time_keys]

    def _apply_time_axis(
        self,
        ax: Axes,
        *,
        rotate: int = 45,
        grid: bool = True,
    ) -> None:
        """Apply the prepared shared time-axis settings to an axis.

        Args:
            ax: Matplotlib axis to configure.
            rotate: Rotation angle for x-axis tick labels.
            grid: Whether to show a light grid.

        Raises:
            RuntimeError: If :meth:`_prepare_time_axis` has not been called.
        """
        if self._time_axis_config is None:
            raise RuntimeError("_prepare_time_axis must be called before plotting.")
        config: TimeAxisConfig = self._time_axis_config

        ax.set_xlim(config.x_min, config.x_max)
        ax.set_xticks(config.tick_positions)
        ax.set_xticklabels(
            config.tick_labels,
            rotation=rotate,
            ha="right",
            rotation_mode="anchor",
        )

        if grid:
            ax.grid(True, alpha=0.3)

    def _format_time_key(self, time_key: TimeKey) -> str:
        """Format a time key for axis labels and summaries.

        Args:
            time_key: Integer time step or datetime value.

        Returns:
            Human-readable time label.
        """
        if isinstance(time_key, datetime):
            return time_key.strftime("%Y-%m-%d\n %H:%M")

        return str(time_key)

    def get_agent_id2name(self, store: RecordStore) -> dict[int, str]:
        """Utility method to create a mapping from agent IDs to agent names using AgentGenerationRecord.

        Args:
            store (RecordStore): The record store containing the records to analyze.

        Returns:
            dict[int, str]: A dictionary mapping agent IDs to agent names.
        """
        agent_id2name: dict[int, str] = {}
        for record in store.typed(AgentGenerationRecord):
            agent_id2name[record.agent_id] = record.agent_name
        return agent_id2name


class FigureSaver:
    """Utility class for saving analyzer figures to PDF files.

    :class:`FigureSaver` centralises DPI, font size, and LaTeX settings so
    that all figures produced by different analyzers share the same visual
    style. Figures are saved as PDFs; ``tight_layout`` and
    ``bbox_inches="tight"`` prevent tick labels from being clipped.

    Example::

        saver = FigureSaver(dpi=300, font_size=12)
        figures = analyzer.draw_figs(result)
        saver.save_to_pdf(figures, Path("outputs/price_analysis"))
    """

    def __init__(
        self,
        dpi: int = 300,
        use_tex: bool = False,
        font_size: int = 15,
        rc_params: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialization.

        Args:
            dpi (int): The resolution in dots per inch for saved figures. Default is 300.
            use_tex (bool): Whether to use LaTeX for rendering text in figures. Default is False.
            font_size (int): The base font size for figure text. Default is 10.
            rc_params (Optional[dict[str, Any]]): Additional Matplotlib rcParams to customize figure style.
        """
        self.dpi = dpi
        self.use_tex = use_tex
        self.font_size = font_size
        self.set_style(rc_params)

    def set_style(self, rc_params: Optional[dict[str, Any]] = None) -> None:
        """Set the Matplotlib style for figures.

        Args:
            rc_params (Optional[dict[str, Any]]):
                Additional Matplotlib rcParams to customize figure style.
                This will override the default style settings if provided.
        """
        mpl.rcParams.update(
            {
                "font.size": self.font_size,
                "axes.titlesize": self.font_size,
                "axes.labelsize": self.font_size,
                "legend.fontsize": self.font_size - 1,
                "xtick.labelsize": self.font_size - 1,
                "ytick.labelsize": self.font_size - 1,
                "text.usetex": self.use_tex,
            }
        )
        if rc_params is not None:
            mpl.rcParams.update(rc_params)

    def save_to_pdf(
        self,
        figs: dict[str, Figure],
        parent_path: Path,
    ) -> None:
        """Save figures to PDF files in the specified directory.

        Args:
            figs (dict[str, Figure]): A dictionary
                mapping figure names to Matplotlib Figure objects.
            parent_path (Path): The directory path where
                the PDF files will be saved.
                The directory will be created if it does not exist.
        """
        if not parent_path.exists():
            parent_path.mkdir(parents=True)
        for name, fig in figs.items():
            fig.tight_layout()
            fig.savefig(
                parent_path / f"{name}.pdf",
                format="pdf",
                dpi=self.dpi,
                bbox_inches="tight",
                pad_inches=0.02,
            )
        plt.close(fig)


class AnalysisManager:
    """Orchestrate multiple analyzers over one or more record stores.

    :class:`AnalysisManager` iterates over a list of
    :class:`AnalyzerBase` subclasses, runs them on the supplied
    :class:`RecordStore`\\ (s), and optionally renders terminal summaries
    and saves PDF figures.

    Example::

        from econsimulacra.log_analyses import (
            load_from_file,
            RecordStore,
            ActionCounter,
            FollowerCounter,
            StoreSalesAnalyzer,
            AnalysisManager,
        )

        records = load_from_file("path/to/log.txt")
        store = RecordStore(records)
        manager = AnalysisManager([
            ActionCounter(),
            FollowerCounter(),
            StoreSalesAnalyzer(),
        ])
        results = manager.run_all(store, figs_save_path="path/to/save/figs")

    The returned ``results`` dict maps each analyzer's ``name`` attribute
    to the result object produced by :meth:`~AnalyzerBase.analyze`.
    """

    def __init__(self, analyzers: list[AnalyzerBase[Any, Any]]) -> None:
        """Initialization.

        Args:
            analyzers (list[AnalyzerBase[Any, Any]]): A list of analyzer instances to run.
        """
        self.analyzers = analyzers

    def run_all(
        self,
        store: RecordStore,
        render_summary: bool = False,
        figs_save_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run all analyzers and save their figures.

        Args:
            store (RecordStore): The record store to analyze.
            render_summary (bool): Whether to render a summary of the analysis results.
            figs_save_path (str, optional): The directory path to save figures.
        """
        results: dict[str, Any] = {}
        saver: FigureSaver = FigureSaver()
        for analyzer in self.analyzers:
            result: Any = analyzer.analyze(store)
            if render_summary:
                analyzer.summarize_results(result)
            results[analyzer.name] = result
            if figs_save_path is not None:
                figs: dict[str, Figure] = analyzer.draw_figs(result)
                saver.save_to_pdf(figs, pathlib.Path(figs_save_path))
        return results

    def run_all_w_multi_stores(
        self,
        stores: list[RecordStore],
        render_summary: bool = False,
        figs_save_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run all analyzers on multiple stores and save their figures.

        Args:
            stores (list[RecordStore]): The list of record stores to analyze.
            render_summary (bool): Whether to render a summary of the analysis results.
            figs_save_path (str, optional): The directory path to save figures.
        """
        individual_results: dict[str, list[Any]] = {}
        results: dict[str, Any] = {}
        saver: FigureSaver = FigureSaver()
        for analyzer in self.analyzers:
            for store in stores:
                result: Any = analyzer.analyze(store)
                if analyzer.name not in individual_results:
                    individual_results[analyzer.name] = []
                individual_results[analyzer.name].append(result)
            result = analyzer.analyze_stores(stores)
            if render_summary:
                analyzer.summarize_results_all(result)
            results[analyzer.name] = result
            if figs_save_path is not None:
                figs: dict[str, Figure] = analyzer.draw_figs_all(
                    individual_results[analyzer.name]
                )
                saver.save_to_pdf(figs, pathlib.Path(figs_save_path))
        return results
