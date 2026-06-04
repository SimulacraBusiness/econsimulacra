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
from scipy.optimize import curve_fit

from .base import AnalyzerBase
from .records import InnerThoughtRecord, OrderReactionRecord, TweetRecord
from .store import RecordStore

TopicSalesResult: TypeAlias = dict[int, dict[str, float]]


@dataclass
class QuadraticFitResult:
    """Quadratic regression result (y = a * x^2 + b * x + c) for a single store.

    Attributes:
        store_name (str): The name of the store.
        a (float): Coefficient of the quadratic term.
        b (float): Coefficient of the linear term.
        c (float): Coefficient of the constant term.
        r2 (float): R-squared value of the fit.
        mse (float): Mean squared error of the fit.
        rmse (float): Root mean squared error of the fit.
        n_samples (int): Number of samples used in the fit.
        x_mean (float): Mean of the x values (word counts).
        x_std (float): Standard deviation of the x values (word counts).
        y_mean (float): Mean of the y values (sales).
        y_std (float): Standard deviation of the y values (sales).
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
    """Topic sales analyzer.

    TopicSalesAnalyzer records the word count of specified topic words
    and the sales of each store within a certain time window,
    and analyzes the relationship between them using the formula:

    ``math``:
        sales = a \\cdot \\text{word count}^2 + b \\cdot \\text{word count} + c

    Note:
        This analyzer estimates the relationship between topic word count and sales
        via analyze_stores() method, which aggregates the results of multiple stores and fits a quadratic regression.
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
        """Analyze the relationship between topic word count and sales.

        Args:
            store (RecordStore): The record store to analyze.

        Returns:
            TopicSalesResult:
                A dictionary mapping store names to another dictionary that maps time steps to a dictionary containing 'word_count' and 'sales'.
                Ex)
                {
                    0: {
                        "word_count": 0,
                        "sales_Pizza Place": 100.5,
                        "sales_amount_Pizza Place": 5.0,
                        "sales_Burger Joint": 200.0,
                        "sales_amount_Burger Joint": 10.0,
                        ...
                    },
                    0+window_size: {
                        "word_count": 5,
                        "sales_Pizza Place": 50.0,
                        "sales_Burger Joint": 150.0,
                        "sales_amount_Pizza Place": 2.0,
                        "sales_amount_Burger Joint": 7.0,
                        ...
                    },
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
        """Draw scatter plots to visualize the relationship between topic word count and sales.

        Args:
            result (TopicSalesResult): The analysis result to visualize.
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
        """Draw figures for multiple analysis results.

        This method aggregates the data from multiple TopicSalesResult instances and draws a single scatter plot for each firm, showing all data points across the results.

        Args:
            results (list[TopicSalesResult]): A list of analysis results to visualize.

        Returns:
            dict[str, Figure]: A dictionary mapping firm names to their respective scatter plot figures.
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
        """Analyze multiple stores and aggregate the results.

        Args:
            stores (list[RecordStore]): A list of record stores to analyze.

        Returns:
            dict[str, QuadraticFitResult]: A dictionary mapping store names to their respective quadratic fit results.
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
        """
        Summarize quadratic regression results using rich tables.

        Parameters
        ----------
        results :
            Mapping from store name to regression result.

        console :
            Rich console instance. If None, a new Console is created.
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
