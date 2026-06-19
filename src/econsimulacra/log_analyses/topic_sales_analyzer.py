from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from scipy.optimize import curve_fit  # type: ignore

from .base import AnalyzerBase
from .records import InnerThoughtRecord, OrderReactionRecord, TweetRecord
from .store import RecordStore

TopicSalesResult: TypeAlias = dict[int, dict[str, float]]


@dataclass
class QuadraticFitResult:
    """Quadratic regression result for a single store.

    Stores the fitted coefficients of the model:

    .. math::

        \\hat{y}_{\\text{norm}} =
        a \\cdot x_{\\text{norm}}^2 + b \\cdot x_{\\text{norm}} + c

    where :math:`x` and :math:`y` are z-scored word count and sales:

    .. math::

        x_{\\text{norm}} = \\frac{x - \\bar{x}}{\\sigma_x},
        \\qquad
        y_{\\text{norm}} = \\frac{y - \\bar{y}}{\\sigma_y}

    Attributes:
        store_name (str): Name of the store.
        a (float): Coefficient of the quadratic term :math:`x^2`.
        b (float): Coefficient of the linear term :math:`x`.
        c (float): Constant term (intercept).
        r2 (float): Coefficient of determination :math:`R^2`.
        mse (float): Mean squared error on the normalised scale.
        rmse (float): Root mean squared error on the normalised scale.
        n_samples (int): Number of ``(word_count, sales)`` data points.
        x_mean (float): Sample mean of the raw word-count values.
        x_std (float): Sample standard deviation of the raw word-count values.
        y_mean (float): Sample mean of the raw sales values.
        y_std (float): Sample standard deviation of the raw sales values.
    """

    store_name: str
    a: float
    b: float
    c: float
    r2: float
    mse: float
    rmse: float
    n_samples: int
    x_mean: float
    x_std: float
    y_mean: float
    y_std: float


class TopicSalesAnalyzer(AnalyzerBase[TopicSalesResult, dict[str, QuadraticFitResult]]):
    """Estimate the relationship between topic word count and store sales.

    :class:`TopicSalesAnalyzer` counts how often a set of *topic words*
    appear in agent tweets (or inner thoughts) within each time window, and
    cross-references those counts with the concurrent sales of each store.
    Across multiple runs, :meth:`analyze_stores` pools
    ``(word_count, sales)`` data points and fits the quadratic model on the
    z-scored variables:

    .. math::

        \\hat{y}_{\\text{norm}} =
        a \\cdot x_{\\text{norm}}^2 + b \\cdot x_{\\text{norm}} + c

    **Interpretation of curvature**

    * :math:`a > 0` (convex ↑): sales accelerate with topic buzz — a
      super-linear amplification effect.
    * :math:`a < 0` (concave ∩): the relationship saturates or reverses at
      high word counts — possible over-saturation.
    * :math:`a \\approx 0` (linear): a proportional relationship, well
      captured by the linear term :math:`b`.

    Goodness of fit is reported via :math:`R^2` and RMSE. A low
    :math:`R^2` suggests store sales are driven by factors beyond topic
    word count alone.

    Note:
        The single-run :meth:`analyze` method collects the raw
        ``(word_count, sales)`` data and is intended as a building block
        for :meth:`analyze_stores`.
    """

    name = "topic_sales"

    def __init__(
        self,
        topic_words: list[str],
        window_size: int = 24,
        is_inner_thought: bool = False,
        use_amount: bool = False,
        exclude_agent_ids: list[int] = [],
    ):
        """Initialization.

        Args:
            topic_words (list[str]): A list of topic words to count.
            window_size (int, optional): The size of the time window to analyze. Defaults to 24.
            is_inner_thought:
                Whether the target record is inner thought or tweet.
                If True, the analyzer will look for InnerThoughtRecord instead of TweetRecord.
                Defaults to False.
            use_amount:
                Whether to use the accepted amount instead of sales (price * accepted amount) for the analysis. Defaults to False.
            exclude_agent_ids (list[int], optional):
                A list of agent IDs to exclude from the analysis. Defaults to an empty list.
        """
        self.topic_words = topic_words
        self.window_size = window_size
        self.is_inner_thought = is_inner_thought
        self.use_amount = use_amount
        self.exclude_agent_ids = exclude_agent_ids

    def analyze(self, store: RecordStore) -> TopicSalesResult:
        """Collect word-count and sales data per time window from *store*.

        Groups :class:`~econsimulacra.log_analyses.records.TweetRecord` (or
        :class:`~econsimulacra.log_analyses.records.InnerThoughtRecord`) and
        :class:`~econsimulacra.log_analyses.records.OrderReactionRecord`
        entries into non-overlapping windows of ``window_size`` steps:

        .. math::

            w(t) = \\left\\lfloor \\frac{t}{\\text{window\\_size}}
                   \\right\\rfloor \\times \\text{window\\_size}

        For each window, the total occurrence count of ``topic_words`` in
        agent messages is accumulated as ``"word_count"``, and the revenue
        of each store is accumulated as ``"sales_{firm_name}"`` (or
        ``"sales_amount_{firm_name}"`` when ``use_amount=True``).

        Args:
            store (RecordStore): Record store to analyse.

        Returns:
            TopicSalesResult: Dict mapping window-start step to a nested
            dict. Example::

                {
                    0:  {"word_count": 0,  "sales_Pizza Place": 100.5, ...},
                    24: {"word_count": 5,  "sales_Pizza Place": 50.0,  ...},
                    ...
                }
        """
        self._prepare_time_axis(store)
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        agent_id2name: dict[int, str] = self.get_agent_id2name(store)
        records: list[InnerThoughtRecord | TweetRecord | OrderReactionRecord] = [
            *(
                store.typed(InnerThoughtRecord)
                if self.is_inner_thought
                else store.typed(TweetRecord)
            ),
            *store.typed(OrderReactionRecord),
        ]
        result: TopicSalesResult = {}
        for record in records:
            agent_id: int
            if isinstance(record, OrderReactionRecord):
                agent_id = record.counterparty_id
            else:
                agent_id = record.agent_id
            if agent_id in self.exclude_agent_ids:
                continue
            record_time_step: int = int(record.time_step)
            window_start: int = (
                record_time_step // self.window_size
            ) * self.window_size
            if window_start not in result:
                result[window_start] = {"word_count": 0}
            if isinstance(record, InnerThoughtRecord):
                word_count: int = sum(
                    record.inner_thought.count(word) for word in self.topic_words
                )
                result[window_start]["word_count"] += word_count
            elif isinstance(record, TweetRecord):
                word_count = sum(
                    record.message.count(word) for word in self.topic_words
                )
                result[window_start]["word_count"] += word_count
            else:
                if agent_id not in agent_id2name:
                    raise ValueError(f"Agent ID {agent_id} not found in agent_id2name.")
                firm_name: str = agent_id2name[agent_id]
                sales_key: str = (
                    f"sales_amount_{firm_name}"
                    if self.use_amount
                    else f"sales_{firm_name}"
                )
                if sales_key not in result[window_start]:
                    result[window_start][sales_key] = 0.0
                if self.use_amount:
                    result[window_start][sales_key] += record.accept_amount
                else:
                    result[window_start][sales_key] += (
                        record.accept_amount * record.price
                    )
        return result

    def draw_figs(self, result: TopicSalesResult) -> dict[str, Figure]:
        """Draw scatter plots of topic word count vs sales for each store.

        Produces one scatter plot per store found in *result*, with the raw
        (non-normalised) word-count on the x-axis and revenue (or accepted
        amount when ``use_amount=True``) on the y-axis.

        Args:
            result (TopicSalesResult): Output of :meth:`analyze`.

        Returns:
            dict[str, Figure]: Mapping from firm name to Matplotlib figure.
        """
        figures: dict[str, Figure] = {}
        for key in result[next(iter(result))].keys():
            if key.startswith("sales_"):
                firm_name: str = key[len("sales_") :]
                x: list[float] = []
                y: list[float] = []
                for _, values in result.items():
                    if key in values:
                        x.append(float(values["word_count"]))
                        y.append(float(values[key]))
                fig, ax = plt.subplots()
                ax.scatter(x, y)
                ax.set_xlabel("Topic Word Count")
                ax.set_ylabel("Sales" if not self.use_amount else "Accepted Amount")
                figures[firm_name] = fig
        return figures

    def draw_figs_all(
        self, individual_results: list[TopicSalesResult]
    ) -> dict[str, Figure]:
        """Draw pooled scatter plots across multiple runs.

        Aggregates the data from all :class:`TopicSalesResult` instances and
        draws one scatter plot per firm showing all ``(word_count, sales)``
        data points from all runs on the same axes. Both axes are z-scored
        (standardised) to make cross-firm comparison easier.

        Args:
            individual_results (list[TopicSalesResult]): One result per
                simulation run, each produced by :meth:`analyze`.

        Returns:
            dict[str, Figure]: Mapping from firm name to Matplotlib figure.
        """
        figures: dict[str, Figure] = {}
        firm_names: set[str] = set()
        for result in individual_results:
            for _, values in result.items():
                for key in values.keys():
                    if key.startswith("sales_"):
                        firm_names.add(key[len("sales_") :])
        for firm_name in firm_names:
            x: list[float] = []
            y: list[float] = []
            sales_key: str = f"sales_{firm_name}"
            for result in individual_results:
                for _, values in result.items():
                    if sales_key in values:
                        x.append(float(values["word_count"]))
                        y.append(float(values[sales_key]))
            x_arr: NDArray[np.float64] = np.array(x)
            x_arr = (
                (x_arr - np.mean(x_arr)) / np.std(x_arr) if np.std(x_arr) > 0 else x_arr
            )
            y_arr: NDArray[np.float64] = np.array(y)
            y_arr = (
                (y_arr - np.mean(y_arr)) / np.std(y_arr) if np.std(y_arr) > 0 else y_arr
            )
            fig, ax = plt.subplots()
            ax.scatter(x_arr, y_arr, alpha=0.6, s=2)
            ax.set_xlabel("Topic Word Count")
            ax.set_ylabel("Sales" if not self.use_amount else "Accepted Amount")
            figures[firm_name] = fig
        return figures

    def analyze_stores(
        self, stores: list[RecordStore]
    ) -> dict[str, QuadraticFitResult]:
        """Fit a quadratic model to pooled (word_count, sales) data.

        Calls :meth:`analyze` on each store, pools the resulting data
        points per firm, z-scores both variables, and fits the quadratic
        model:

        .. math::

            \\hat{y}_{\\text{norm}} =
            a \\cdot x_{\\text{norm}}^2 + b \\cdot x_{\\text{norm}} + c

        using :func:`scipy.optimize.curve_fit`. Stores with fewer than
        2 data points or zero variance in :math:`x` or :math:`y` are
        skipped silently.

        Args:
            stores (list[RecordStore]): One record store per simulation
                run.

        Returns:
            dict[str, QuadraticFitResult]: Mapping from firm name to the
            corresponding :class:`QuadraticFitResult`. Firms for which a
            fit could not be obtained are absent from the dict.
        """
        topic_sales_results: list[TopicSalesResult] = [
            self.analyze(store) for store in stores
        ]
        firm_names: set[str] = set()
        for result in topic_sales_results:
            for _, values in result.items():
                for key in values.keys():
                    if key.startswith("sales_"):
                        firm_names.add(key[len("sales_") :])
        fit_results: dict[str, QuadraticFitResult] = {}

        def quadratic_model(
            x: NDArray[np.float64], a: float, b: float, c: float
        ) -> NDArray[np.float64]:
            return a * x**2 + b * x + c

        for firm_name in firm_names:
            xs: list[float] = []
            ys: list[float] = []
            sales_key: str = f"sales_{firm_name}"
            for result in topic_sales_results:
                for _, values in result.items():
                    if sales_key in values:
                        xs.append(float(values["word_count"]))
                        ys.append(float(values[sales_key]))
            x_arr: NDArray[np.float64] = np.array(xs)
            y_arr: NDArray[np.float64] = np.array(ys)
            if len(x_arr) < 2:
                continue
            x_mean: float = float(np.mean(x_arr))
            x_std: float = float(np.std(x_arr))
            y_mean: float = float(np.mean(y_arr))
            y_std: float = float(np.std(y_arr))
            if x_std == 0.0 or y_std == 0.0:
                continue
            x_norm: NDArray[np.float64] = (x_arr - x_mean) / x_std
            y_norm: NDArray[np.float64] = (y_arr - y_mean) / y_std
            try:
                popt, _ = curve_fit(
                    quadratic_model, x_norm, y_norm, p0=(0.0, 0.0, 0.0), maxfev=10_000
                )
            except RuntimeError:
                continue
            a, b, c = map(float, popt)
            y_pred: NDArray[np.float64] = quadratic_model(x_norm, a, b, c)
            residual: NDArray[np.float64] = y_norm - y_pred
            mse: float = float(np.mean(residual**2))
            rmse: float = float(np.sqrt(mse))
            ss_res = float(np.sum(residual**2))
            ss_tot = float(np.sum((y_norm - np.mean(y_norm)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            fit_results[firm_name] = QuadraticFitResult(
                store_name=firm_name,
                a=a,
                b=b,
                c=c,
                r2=r2,
                mse=mse,
                rmse=rmse,
                n_samples=len(x_arr),
                x_mean=x_mean,
                x_std=x_std,
                y_mean=y_mean,
                y_std=y_std,
            )
        return fit_results

    def build_summary_all(
        self,
        results: dict[str, QuadraticFitResult],
    ) -> RenderableType:
        """Summarise quadratic regression results in a Rich table.

        Displays one row per store, ranked by :math:`R^2` descending, with
        columns for :math:`a`, :math:`b`, :math:`c`, :math:`R^2`, RMSE,
        sample count, and a trend label:

        * *Convex ↑* when :math:`a > 0` (accelerating returns with buzz)
        * *Concave ∩* when :math:`a < 0` (diminishing / saturating returns)
        * *Linear* when :math:`a \\approx 0`

        Args:
            results (dict[str, QuadraticFitResult]): Mapping from store
                name to regression result, as returned by
                :meth:`analyze_stores`.

        Returns:
            RenderableType: Rich panel containing the summary table.
        """

        table = Table(
            title="SNS Topic vs Store Sales Analysis",
            show_lines=True,
        )

        table.add_column("Store", style="bold cyan")
        table.add_column("a", justify="right")
        table.add_column("b", justify="right")
        table.add_column("c", justify="right")
        table.add_column("R²", justify="right")
        table.add_column("RMSE", justify="right")
        table.add_column("N", justify="right")
        table.add_column("Trend", justify="left")

        sorted_results = sorted(
            results.values(),
            key=lambda r: r.r2,
            reverse=True,
        )

        for result in sorted_results:
            # -----------------------------------------------------
            # Interpret quadratic shape
            # -----------------------------------------------------
            if result.a > 0:
                trend = "Convex ↑"
            elif result.a < 0:
                trend = "Concave ∩"
            else:
                trend = "Linear"

            table.add_row(
                result.store_name,
                f"{result.a:+.4f}",
                f"{result.b:+.4f}",
                f"{result.c:+.4f}",
                f"{result.r2:.4f}",
                f"{result.rmse:.4f}",
                str(result.n_samples),
                trend,
            )

        return Panel(table, title="Quadratic Fit Results", border_style="green")
