from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table

from .base import AnalyzerBase
from .records import InnerThoughtRecord, TweetRecord
from .store import RecordStore

try:
    from scipy.stats import pearsonr  # type: ignore
except ImportError:
    pearsonr = None  # type: ignore[assignment]


@dataclass(frozen=True)
class TopicSentimentWindowStats:
    """Per-window statistics for topic word count and sentiment distribution.

    Attributes:
        word_count (int): Total occurrences of topic words across all
            matching records in this window (same counting convention as
            :class:`TopicSalesAnalyzer`).
        mean_sentiment (float, optional): Mean sentiment score of matching
            records in this window. ``None`` if no records had a valid
            (non-``None``) sentiment score.
        std_sentiment (float, optional): Sample standard deviation (ddof=1)
            of sentiment scores of matching records. ``None`` if fewer than 2
            records had a valid sentiment score.
        n_sentiment (int): Number of matching records with a non-``None``
            sentiment score used to compute ``mean_sentiment`` and
            ``std_sentiment``.
    """

    word_count: int
    mean_sentiment: Optional[float]
    std_sentiment: Optional[float]
    n_sentiment: int


TopicSentimentResult: TypeAlias = dict[int, TopicSentimentWindowStats]


@dataclass(frozen=True)
class TopicSentimentCorrelationResult:
    """Cross-store correlation result between topic buzz and sentiment polarisation.

    This is the output of :meth:`TopicSentimentAnalyzer.analyze_stores`. It
    pools ``(word_count, std_sentiment)`` pairs across all windows and all
    stores to estimate the Pearson correlation coefficient :math:`r`.

    Attributes:
        r (float, optional): Pearson correlation coefficient between
            ``word_count`` and ``std_sentiment`` across all pooled
            ``(window, store)`` data points:

            .. math::

                r = \\frac{\\sum_i (x_i - \\bar{x})(y_i - \\bar{y})}
                          {\\sqrt{\\sum_i (x_i - \\bar{x})^2}\\;
                           \\sqrt{\\sum_i (y_i - \\bar{y})^2}}

            ``None`` if fewer than 2 valid data points are available, or
            if either variable has zero variance.
        p_value (float, optional): Two-tailed p-value for the null hypothesis
            :math:`r = 0`. ``None`` when ``r`` is ``None``.
        n_samples (int): Number of ``(word_count, std_sentiment)`` pairs used
            in the correlation computation. Only windows with
            ``std_sentiment is not None`` contribute.
        word_counts (list[float]): Pooled word-count values (:math:`x`).
        std_sentiments (list[float]): Pooled sentiment std-deviation values
            (:math:`y`).
    """

    r: Optional[float]
    p_value: Optional[float]
    n_samples: int
    word_counts: list[float]
    std_sentiments: list[float]


class TopicSentimentAnalyzer(
    AnalyzerBase[TopicSentimentResult, TopicSentimentCorrelationResult]
):
    """Measure how topic buzz intensity relates to sentiment polarisation.

    This analyzer tests the hypothesis:

        *The more agents discuss a topic in a time window, the more polarised
        (higher standard deviation) the sentiment of those discussions becomes.*

    For each time window of ``window_size`` steps, it counts how often the
    ``topic_words`` appear in agent tweets (or inner thoughts) and computes the
    distribution of sentiment scores for those matching records.

    **Per-window computation**

    Let :math:`\\mathcal{R}_w` be the set of records in window :math:`w`
    that contain at least one word from ``topic_words``, and let
    :math:`\\mathcal{S}_w \\subseteq \\mathcal{R}_w` be the subset with a
    non-``None`` sentiment score.

    * **Word count** – total topic-word occurrences (buzz intensity):

      .. math::

          c_w = \\sum_{r \\in \\mathcal{R}_w}
                \\sum_{k} \\text{count}(\\text{word}_k,\\, \\text{message}_r)

    * **Sentiment mean**:

      .. math::

          \\bar{s}_w = \\frac{1}{|\\mathcal{S}_w|}
                       \\sum_{r \\in \\mathcal{S}_w} s_r

    * **Sentiment std** (sample standard deviation, ddof=1):

      .. math::

          \\sigma_w = \\sqrt{\\frac{1}{|\\mathcal{S}_w|-1}
                      \\sum_{r \\in \\mathcal{S}_w} (s_r - \\bar{s}_w)^2}

    **Cross-store correlation**

    :meth:`analyze_stores` pools all ``(c_w, \\sigma_w)`` pairs across every
    window and every run, then computes the Pearson correlation coefficient:

    .. math::

        r = \\text{corr}(c_w,\\; \\sigma_w)

    A significantly positive :math:`r` supports the polarisation hypothesis.
    Use :meth:`build_summary_all` to display :math:`r` and its p-value.

    Note:
        The ``sentiment`` field of
        :class:`~econsimulacra.log_analyses.records.TweetRecord` and
        :class:`~econsimulacra.log_analyses.records.InnerThoughtRecord` must
        be populated (not ``None``) for sentiment statistics to be meaningful.
        If sentiment scores are absent, ``std_sentiment`` will be ``None`` for
        all windows and the correlation cannot be estimated.
    """

    name = "topic_sentiment"

    def __init__(
        self,
        topic_words: list[str],
        window_size: int = 24,
        is_inner_thought: bool = False,
        exclude_agent_ids: list[int] = [],
    ) -> None:
        """Initialise the analyzer.

        Args:
            topic_words (list[str]): Words whose presence in a message marks
                it as on-topic. Word count per record is the sum of
                :py:meth:`str.count` for each word, matching the convention of
                :class:`TopicSalesAnalyzer`.
            window_size (int): Number of simulation steps per aggregation
                window. Must be positive. Defaults to 24.
            is_inner_thought (bool): If ``True``, analyse
                :class:`~econsimulacra.log_analyses.records.InnerThoughtRecord`
                instead of
                :class:`~econsimulacra.log_analyses.records.TweetRecord`.
                Defaults to ``False``.
            exclude_agent_ids (list[int]): Agent IDs excluded from the
                analysis (e.g., non-consumer agents). Defaults to ``[]``.
        """
        self.topic_words = topic_words
        self.window_size = window_size
        self.is_inner_thought = is_inner_thought
        self.exclude_agent_ids = exclude_agent_ids

    def analyze(self, store: RecordStore) -> TopicSentimentResult:
        """Compute per-window buzz intensity and sentiment statistics.

        For each time window the method:

        1. Filters records to those containing at least one occurrence of a
           word from ``topic_words``.
        2. Accumulates the total word-occurrence count as ``word_count``.
        3. Collects non-``None`` sentiment scores from matching records.
        4. Computes the mean and sample standard deviation (ddof=1) of those
           scores.

        Windows with no matching records are omitted from the result.

        Args:
            store (RecordStore): Record store containing tweet or inner-thought
                records.

        Returns:
            TopicSentimentResult: Mapping from window-start step to a
            :class:`TopicSentimentWindowStats`. Only windows that contain at
            least one matching record are included.

        Raises:
            ValueError: If ``window_size`` is not positive.
        """
        self._prepare_time_axis(store)
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")

        records: list[TweetRecord] | list[InnerThoughtRecord] = (
            store.typed(InnerThoughtRecord)
            if self.is_inner_thought
            else store.typed(TweetRecord)
        )

        window_word_counts: dict[int, int] = {}
        window_sentiments: dict[int, list[float]] = {}

        for record in records:
            if record.agent_id in self.exclude_agent_ids:
                continue

            message: str = (
                record.inner_thought
                if isinstance(record, InnerThoughtRecord)
                else record.message
            )

            total_count: int = sum(message.count(word) for word in self.topic_words)
            if total_count == 0:
                continue

            window_start: int = (
                record.time_step // self.window_size
            ) * self.window_size

            if window_start not in window_word_counts:
                window_word_counts[window_start] = 0
                window_sentiments[window_start] = []

            window_word_counts[window_start] += total_count

            if record.sentiment is not None:
                window_sentiments[window_start].append(record.sentiment)

        result: TopicSentimentResult = {}
        for window_start, word_count in window_word_counts.items():
            sentiments: list[float] = window_sentiments[window_start]
            n: int = len(sentiments)
            mean_sentiment: Optional[float] = (
                float(np.mean(sentiments)) if n >= 1 else None
            )
            std_sentiment: Optional[float] = (
                float(np.std(sentiments, ddof=1)) if n >= 2 else None
            )
            result[window_start] = TopicSentimentWindowStats(
                word_count=word_count,
                mean_sentiment=mean_sentiment,
                std_sentiment=std_sentiment,
                n_sentiment=n,
            )

        return result

    def analyze_stores(
        self, stores: list[RecordStore]
    ) -> TopicSentimentCorrelationResult:
        """Estimate the Pearson correlation between buzz and sentiment polarisation.

        Calls :meth:`analyze` on each store, pools all
        ``(word_count, std_sentiment)`` pairs where ``std_sentiment`` is not
        ``None``, and computes :math:`r = \\text{corr}(c_w, \\sigma_w)` using
        :func:`scipy.stats.pearsonr`.

        Args:
            stores (list[RecordStore]): One record store per simulation run.

        Returns:
            TopicSentimentCorrelationResult: Correlation result including
            :math:`r`, the two-tailed p-value, the sample size, and the raw
            pooled vectors for custom plotting.
        """
        word_counts: list[float] = []
        std_sentiments: list[float] = []

        for store in stores:
            result = self.analyze(store)
            for stats in result.values():
                if stats.std_sentiment is not None:
                    word_counts.append(float(stats.word_count))
                    std_sentiments.append(stats.std_sentiment)

        n: int = len(word_counts)
        r: Optional[float] = None
        p_value: Optional[float] = None

        if n >= 2 and pearsonr is not None:
            x: np.ndarray = np.array(word_counts)
            y: np.ndarray = np.array(std_sentiments)
            if float(np.std(x)) > 0.0 and float(np.std(y)) > 0.0:
                corr = pearsonr(x, y)
                r = float(corr[0])  # type: ignore[arg-type]
                p_value = float(corr[1])  # type: ignore[arg-type]

        return TopicSentimentCorrelationResult(
            r=r,
            p_value=p_value,
            n_samples=n,
            word_counts=word_counts,
            std_sentiments=std_sentiments,
        )

    def draw_figs(self, result: TopicSentimentResult) -> dict[str, Figure]:
        """Draw per-run diagnostic figures.

        Produces up to two figures:

        * ``topic_sentiment_timeseries`` – two-row subplot: (top) word count
          as a bar chart over time; (bottom) mean sentiment ± std as a line
          with a shaded band.
        * ``word_count_vs_std_sentiment`` – scatter plot of word count (:math:`x`)
          vs sentiment standard deviation (:math:`y`) across windows within
          this run, directly visualising the within-run polarisation trend.

        Args:
            result (TopicSentimentResult): Output of :meth:`analyze`.

        Returns:
            dict[str, Figure]: Figure name → Matplotlib figure. Empty if
            *result* contains no windows.
        """
        figures: dict[str, Figure] = {}

        if not result:
            return figures

        sorted_windows: list[int] = sorted(result.keys())

        fig_ts: Figure
        axes_ts: np.ndarray
        fig_ts, axes_ts = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax_count: Axes = axes_ts[0]
        ax_sent: Axes = axes_ts[1]

        word_counts_plot: list[int] = [result[w].word_count for w in sorted_windows]
        ax_count.bar(
            sorted_windows,
            word_counts_plot,
            width=self.window_size * 0.8,
            align="edge",
        )
        ax_count.set_ylabel("Topic word count")
        ax_count.grid(True, alpha=0.3)
        self._apply_time_axis(ax_count)

        valid_windows: list[int] = [
            w for w in sorted_windows if result[w].mean_sentiment is not None
        ]
        if valid_windows:
            valid_means: list[float] = [
                cast(float, result[w].mean_sentiment) for w in valid_windows
            ]
            valid_stds: list[float] = []
            for w in valid_windows:
                std = result[w].std_sentiment
                valid_stds.append(std if std is not None else 0.0)
            ax_sent.plot(valid_windows, valid_means, marker="o", label="mean")
            ax_sent.fill_between(
                valid_windows,
                [m - s for m, s in zip(valid_means, valid_stds)],
                [m + s for m, s in zip(valid_means, valid_stds)],
                alpha=0.3,
                label="±std",
            )
            ax_sent.legend()
        ax_sent.set_ylabel("Sentiment score")
        ax_sent.grid(True, alpha=0.3)
        self._apply_time_axis(ax_sent)

        fig_ts.tight_layout()
        figures["topic_sentiment_timeseries"] = fig_ts

        scatter_x: list[float] = [
            float(result[w].word_count)
            for w in sorted_windows
            if result[w].std_sentiment is not None
        ]
        scatter_y: list[float] = [
            result[w].std_sentiment  # type: ignore[misc]
            for w in sorted_windows
            if result[w].std_sentiment is not None
        ]
        if scatter_x:
            fig_sc: Figure
            ax_sc: Axes
            fig_sc, ax_sc = plt.subplots(figsize=(8, 6))
            ax_sc.scatter(scatter_x, scatter_y, alpha=0.7)
            ax_sc.set_xlabel("Topic word count")
            ax_sc.set_ylabel("Sentiment std deviation")
            ax_sc.grid(True, alpha=0.3)
            fig_sc.tight_layout()
            figures["word_count_vs_std_sentiment"] = fig_sc

        return figures

    def draw_figs_all(
        self,
        individual_results: list[TopicSentimentResult],
    ) -> dict[str, Figure]:
        """Draw a pooled scatter of word count vs sentiment std across all runs.

        Aggregates ``(word_count, std_sentiment)`` pairs from all runs and
        plots them in a single scatter. A linear regression line is overlaid
        and the Pearson :math:`r` and p-value are shown in the legend when
        :mod:`scipy` is available.

        Args:
            individual_results (list[TopicSentimentResult]): One result per
                simulation run, each produced by :meth:`analyze`.

        Returns:
            dict[str, Figure]: ``{"word_count_vs_std_sentiment_all": fig}``
            or empty if no valid data points exist.
        """
        figures: dict[str, Figure] = {}

        xs: list[float] = []
        ys: list[float] = []
        for result in individual_results:
            for stats in result.values():
                if stats.std_sentiment is not None:
                    xs.append(float(stats.word_count))
                    ys.append(stats.std_sentiment)

        if not xs:
            return figures

        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(xs, ys, alpha=0.4, s=20)
        ax.set_xlabel("Topic word count")
        ax.set_ylabel("Sentiment std deviation")
        ax.grid(True, alpha=0.3)

        x_arr: np.ndarray = np.array(xs)
        y_arr: np.ndarray = np.array(ys)
        if len(xs) >= 2 and float(np.std(x_arr)) > 0.0:
            coeffs = np.polyfit(x_arr, y_arr, 1)
            x_line = np.linspace(float(x_arr.min()), float(x_arr.max()), 200)
            y_line = np.polyval(coeffs, x_line)
            label: str = "regression"
            if pearsonr is not None and float(np.std(y_arr)) > 0.0:
                corr = pearsonr(x_arr, y_arr)
                label = f"r = {float(corr[0]):.3f},  p = {float(corr[1]):.3f}"  # type: ignore[arg-type]
            ax.plot(x_line, y_line, "r-", label=label)
            ax.legend()

        fig.tight_layout()
        figures["word_count_vs_std_sentiment_all"] = fig
        return figures

    def build_summary(self, result: TopicSentimentResult) -> RenderableType:
        """Build a Rich summary table for a single-run analysis.

        Displays one row per time window with word count, number of records
        with valid sentiment, mean sentiment, and sentiment standard deviation.

        Args:
            result (TopicSentimentResult): Output of :meth:`analyze`.

        Returns:
            RenderableType: Rich panel containing the per-window summary table.
        """
        table = Table(
            title="Topic Sentiment (per window)",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Window start", justify="right")
        table.add_column("Word count", justify="right")
        table.add_column("N (sentiment)", justify="right")
        table.add_column("Mean sentiment", justify="right")
        table.add_column("Std sentiment", justify="right")

        for window_start in sorted(result.keys()):
            stats: TopicSentimentWindowStats = result[window_start]
            table.add_row(
                str(window_start),
                str(stats.word_count),
                str(stats.n_sentiment),
                f"{stats.mean_sentiment:.3f}"
                if stats.mean_sentiment is not None
                else "-",
                f"{stats.std_sentiment:.3f}"
                if stats.std_sentiment is not None
                else "-",
            )

        return Panel(table, title=f"{self.name} Summary", border_style="blue")

    def build_summary_all(
        self,
        results: TopicSentimentCorrelationResult,
    ) -> RenderableType:
        """Build a Rich summary panel for the cross-store correlation result.

        Displays the sample size, Pearson :math:`r`, p-value, and an
        interpretation of the sign of :math:`r`. A positive and statistically
        significant :math:`r` supports the hypothesis that higher topic buzz
        leads to more polarised sentiment.

        Args:
            results (TopicSentimentCorrelationResult): Output of
                :meth:`analyze_stores`.

        Returns:
            RenderableType: Rich panel containing the correlation summary.
        """
        table = Table(
            title="Topic Sentiment Correlation (buzz vs polarisation)",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Metric")
        table.add_column("Value", justify="right")

        table.add_row(
            "N samples  (windows with std_sentiment)",
            str(results.n_samples),
        )
        table.add_row(
            "Pearson r  (word_count vs std_sentiment)",
            f"{results.r:.4f}" if results.r is not None else "-",
        )
        table.add_row(
            "p-value  (two-tailed, H₀: r = 0)",
            f"{results.p_value:.4f}" if results.p_value is not None else "-",
        )

        if results.r is not None:
            if results.r > 0:
                interpretation = "Positive: higher buzz → higher sentiment variance"
            elif results.r < 0:
                interpretation = "Negative: higher buzz → lower sentiment variance"
            else:
                interpretation = "No linear correlation detected"
            table.add_row("Interpretation", interpretation)

        return Panel(
            table,
            title=f"{self.name} Correlation Summary",
            border_style="blue",
        )
