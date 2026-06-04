from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from pandas import DataFrame
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .base import AnalyzerBase
from .records import InnerThoughtRecord, TweetRecord
from .store import RecordStore

try:
    from bertopic import BERTopic
except ImportError:  # pragma: no cover
    BERTopic = None  # type: ignore[assignment]


TimeKey = int | datetime


@dataclass(frozen=True)
class TopicResult:
    """Result of tweet topic analysis."""

    tweets: DataFrame
    topic_counts: DataFrame
    topic_shares: DataFrame
    topic_summary: DataFrame
    agent_topic_counts: DataFrame


class TopicAnalyzer(
    AnalyzerBase[
        TopicResult,
        list[TopicResult],
    ]
):
    """Analyze tweet logs using BERTopic.

    This analyzer:
    - extracts TweetRecord from RecordStore,
    - assigns a topic to each tweet using BERTopic,
    - aggregates topic dynamics over time windows,
    - summarizes topic frequencies and agent-topic distributions.
    """

    name = "tweet_topic"

    def __init__(
        self,
        topic_model: Optional[object] = None,
        time_window_size: int = 1,
        top_n_topics: int = 10,
        min_topic_size: int = 5,
        calculate_probabilities: bool = False,
        is_inner_thought: bool = False,
        exclude_agent_ids: list[int] = [],
    ) -> None:
        """Initialize analyzer.

        Args:
            topic_model:
                Optional preconfigured BERTopic instance.
                If None, a default BERTopic model is created.
            time_window_size:
                Window size for aggregating integer time_step.
                For example, 6 means time_step 0-5, 6-11, ...
            top_n_topics:
                Number of topics to show in figures and summaries.
            min_topic_size:
                Minimum topic size used when creating default BERTopic.
            calculate_probabilities:
                Whether BERTopic calculates topic probabilities.
                False is faster and usually enough for this analyzer.
            is_inner_thought:
                Whether the target record is inner thought or tweet.
                If True, the analyzer will look for InnerThoughtRecord instead of TweetRecord.
            exclude_agent_ids:
                List of agent IDs to exclude from analysis (e.g., non-agent entities).
        """
        self.topic_model = topic_model
        self.time_window_size = time_window_size
        self.top_n_topics = top_n_topics
        self.min_topic_size = min_topic_size
        self.calculate_probabilities = calculate_probabilities
        self.is_inner_thought = is_inner_thought
        self.exclude_agent_ids = exclude_agent_ids

    def analyze(
        self,
        store: RecordStore,
    ) -> TopicResult:
        self._prepare_time_axis(store)

        records = (
            store.typed(InnerThoughtRecord)
            if self.is_inner_thought
            else store.typed(TweetRecord)
        )

        if len(records) == 0:
            return self._empty_result()

        df = self._records_to_dataframe(records)

        topic_model = self._get_topic_model()

        df = self._fit_and_assign_topics(
            df=df,
            topic_model=topic_model,
        )

        return self._build_topic_result(df)

    def analyze_stores(
        self,
        stores: list[RecordStore],
    ) -> list[TopicResult]:
        self._prepare_time_axis(stores[0])

        all_dfs: list[pd.DataFrame] = []
        all_docs: list[str] = []

        for store in stores:
            records = (
                store.typed(InnerThoughtRecord)
                if self.is_inner_thought
                else store.typed(TweetRecord)
            )

            df = self._records_to_dataframe(records)

            all_dfs.append(df)

            if not df.empty:
                all_docs.extend(df["message"].astype(str).tolist())

        if len(all_docs) == 0:
            return [self._empty_result() for _ in stores]

        topic_model = self._get_topic_model()

        topic_model.fit(all_docs)  # type: ignore

        results: list[TopicResult] = []

        for df in all_dfs:
            if df.empty:
                results.append(self._empty_result())
                continue

            df_with_topics = self._assign_topics(
                df=df,
                topic_model=topic_model,
            )

            results.append(self._build_topic_result(df_with_topics))

        return results

    def draw_figs(self, result: TopicResult) -> dict[str, Figure]:
        """Draw topic analysis figures."""
        figs: dict[str, Figure] = {}

        if result.tweets.empty:
            return figs

        top_topics = self._get_top_topic_names(result)
        inner_str = "inner_thought" if self.is_inner_thought else "tweet"
        figs[f"topic_counts_over_time_{inner_str}"] = self._draw_topic_counts_over_time(
            result.topic_counts,
            top_topics,
        )
        figs[f"topic_shares_over_time_{inner_str}"] = self._draw_topic_shares_over_time(
            result.topic_shares,
            top_topics,
        )
        figs[f"top_topic_counts_{inner_str}"] = self._draw_top_topic_counts(
            result.topic_summary
        )

        return figs

    def draw_figs_all(
        self,
        individual_results: list[TopicResult],
    ) -> dict[str, Figure]:
        return {}

    def build_summary(self, result: TopicResult):
        """Build rich summary."""
        if result.tweets.empty:
            return Panel.fit(
                "No tweet records found.",
                title="Tweet Topic Analysis",
                border_style="yellow",
            )

        total_tweets = len(result.tweets)
        total_agents = result.tweets["agent_id"].nunique()
        total_topics = result.tweets.loc[
            result.tweets["topic"] != -1, "topic"
        ].nunique()
        noise_tweets = int((result.tweets["topic"] == -1).sum())

        overview = Table(title="Overview")
        overview.add_column("Metric")
        overview.add_column("Value", justify="right")
        overview.add_row("Total tweets", str(total_tweets))
        overview.add_row("Tweeting agents", str(total_agents))
        overview.add_row("Unique topics", str(total_topics))
        overview.add_row("Noise tweets", str(noise_tweets))

        top_table = Table(title=f"Top {self.top_n_topics} Topics")
        top_table.add_column("Topic")
        top_table.add_column("#Tweets", justify="right")
        top_table.add_column("#Agents", justify="right")
        top_table.add_column("First step", justify="right")
        top_table.add_column("Last step", justify="right")

        for _, row in result.topic_summary.head(self.top_n_topics).iterrows():
            top_table.add_row(
                str(row["topic_name"]),
                str(int(row["num_tweets"])),
                str(int(row["num_agents"])),
                str(int(row["first_time_step"])),
                str(int(row["last_time_step"])),
            )

        note = Text(
            "Topic -1 is BERTopic's outlier/noise topic.",
            style="dim",
        )
        inner_str = "Inner thought" if self.is_inner_thought else "Tweet"

        return Panel.fit(
            Group(overview, top_table, note),
            title=f"{inner_str} Topic Analysis",
            border_style="green",
        )

    def _empty_result(self) -> TopicResult:
        empty = pd.DataFrame()

        return TopicResult(
            tweets=empty,
            topic_counts=empty,
            topic_shares=empty,
            topic_summary=empty,
            agent_topic_counts=empty,
        )

    def _build_topic_result(
        self,
        df: pd.DataFrame,
    ) -> TopicResult:
        df = df.copy()

        df["time_window"] = df["time_step"].map(self._to_time_window)

        topic_counts = (
            df.groupby(["time_window", "topic_name"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )

        topic_shares = topic_counts.div(
            topic_counts.sum(axis=1),
            axis=0,
        ).fillna(0.0)

        topic_summary = (
            df.groupby(["topic", "topic_name"])
            .agg(
                num_tweets=("message", "size"),
                num_agents=("agent_id", "nunique"),
                first_time_step=("time_step", "min"),
                last_time_step=("time_step", "max"),
                avg_followers=("num_followers", "mean"),
                avg_follows=("num_follows", "mean"),
            )
            .reset_index()
            .sort_values(
                "num_tweets",
                ascending=False,
            )
        )

        agent_topic_counts = (
            df.groupby(["agent_id", "topic_name"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )

        return TopicResult(
            tweets=df,
            topic_counts=topic_counts,
            topic_shares=topic_shares,
            topic_summary=topic_summary,
            agent_topic_counts=agent_topic_counts,
        )

    def _get_topic_model(self) -> object:
        if self.topic_model is not None:
            return self.topic_model

        if BERTopic is None:
            raise ImportError(
                "BERTopic is required for TweetTopicAnalyzer. "
                "Install it with `pip install bertopic`."
            )

        return BERTopic(
            min_topic_size=self.min_topic_size,
            calculate_probabilities=self.calculate_probabilities,
            verbose=False,
        )

    def _assign_topics(
        self,
        df: pd.DataFrame,
        topic_model,
    ) -> pd.DataFrame:
        if df.empty:
            return df

        docs = df["message"].astype(str).tolist()

        topics, _ = topic_model.transform(docs)

        df = df.copy()
        df["topic"] = topics

        topic_names = self._get_topic_names(topic_model)

        df["topic_name"] = df["topic"].map(topic_names).fillna("Unknown")

        return df

    def _fit_and_assign_topics(
        self,
        df: pd.DataFrame,
        topic_model,
    ) -> pd.DataFrame:
        docs = df["message"].astype(str).tolist()

        topics, _ = topic_model.fit_transform(docs)

        df = df.copy()
        df["topic"] = topics

        topic_names = self._get_topic_names(topic_model)

        df["topic_name"] = df["topic"].map(topic_names).fillna("Unknown")

        return df

    def _records_to_dataframe(
        self,
        records: list[TweetRecord] | list[InnerThoughtRecord],
    ) -> pd.DataFrame:
        rows = []
        for record in records:
            if record.agent_id in self.exclude_agent_ids:
                continue
            if not isinstance(record, (TweetRecord, InnerThoughtRecord)):
                continue
            elif isinstance(record, TweetRecord):
                rows.append(
                    {
                        "time": record.time,
                        "time_step": record.time_step,
                        "agent_id": record.agent_id,
                        "message": record.message,
                        "num_follows": record.num_follows,
                        "num_followers": record.num_followers,
                    }
                )
            elif isinstance(record, InnerThoughtRecord):
                rows.append(
                    {
                        "time": record.time,
                        "time_step": record.time_step,
                        "agent_id": record.agent_id,
                        "message": record.inner_thought,
                        "num_follows": 0,
                        "num_followers": 0,
                    }
                )

        return pd.DataFrame(rows).sort_values(["time_step", "agent_id"])

    def _to_time_window(self, time_step: int) -> int:
        if self.time_window_size <= 1:
            return time_step
        return (time_step // self.time_window_size) * self.time_window_size

    def _get_topic_names(self, topic_model) -> dict[int, str]:
        topic_info = topic_model.get_topic_info()
        topic_names: dict[int, str] = {}

        for _, row in topic_info.iterrows():
            topic_id = int(row["Topic"])
            name = str(row["Name"])
            topic_names[topic_id] = name

        return topic_names

    def _get_top_topic_names(self, result: TopicResult) -> list[str]:
        topic_summary = result.topic_summary
        topic_summary = topic_summary[topic_summary["topic"] != -1]
        return topic_summary.head(self.top_n_topics)["topic_name"].tolist()

    def _draw_topic_counts_over_time(
        self,
        topic_counts: DataFrame,
        top_topics: list[str],
    ) -> Figure:
        fig, ax = plt.subplots(figsize=(8, 6))
        topic_counts[top_topics].plot(ax=ax)
        ax.set_xlabel("Time window")
        ax.set_ylabel("#Tweets")
        ax.legend(title="Topic", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        self._apply_time_axis(ax)
        fig.tight_layout()
        return fig

    def _draw_topic_shares_over_time(
        self,
        topic_shares: DataFrame,
        top_topics: list[str],
    ) -> Figure:
        fig, ax = plt.subplots(figsize=(12, 6))
        topic_shares[top_topics].plot.area(ax=ax)
        ax.set_xlabel("Time window")
        ax.set_ylabel("Topic share")
        ax.set_ylim(0, min(1.0, topic_shares[top_topics].sum(axis=1).max() * 1.1))
        ax.legend(title="Topic", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        self._apply_time_axis(ax)
        fig.tight_layout()
        return fig

    def _draw_top_topic_counts(self, topic_summary: DataFrame) -> Figure:
        fig, ax = plt.subplots(figsize=(8, 6))
        summary = topic_summary[topic_summary["topic"] != -1].head(self.top_n_topics)
        ax.barh(summary["topic_name"], summary["num_tweets"])
        ax.invert_yaxis()
        ax.set_xlabel("#Tweets")
        ax.set_ylabel("Topic")
        ax.grid(True, alpha=0.3)
        self._apply_time_axis(ax)
        fig.tight_layout()
        return fig
