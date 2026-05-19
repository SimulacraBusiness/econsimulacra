from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.dates import DateFormatter, date2num
from matplotlib.figure import Figure
from numpy.typing import NDArray
from rich.panel import Panel
from rich.table import Table
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    adjusted_rand_score,
    silhouette_score,
)

from .base import AnalyzerBase
from .records import ConsumptionRecord, ItemGenerationRecord, OrderRecord
from .store import RecordStore

try:
    from umap import UMAP  # type: ignore[import-untyped]
except ImportError:
    UMAP = None

TimeKey: TypeAlias = int | datetime
ConsumerVector: TypeAlias = NDArray[np.float32]
Embedding: TypeAlias = NDArray[np.float64]
ClusterLabelArray: TypeAlias = NDArray[np.int_]
ConsumerClusterResult: TypeAlias = dict[int, dict[TimeKey, tuple[int, ConsumerVector]]]


class FitTransform2D(Protocol):
    def fit_transform(self, x: NDArray[np.float32]) -> Any:
        """Project input vectors into a lower-dimensional representation."""
        ...


@dataclass
class ConsumerClusterAnalyzer(AnalyzerBase[ConsumerClusterResult]):
    """Cluster agents' purchase or consumption behavior over rolling time windows.

    This analyzer converts each ``(agent, time-window)`` pair into an item-wise
    behavior vector, where each vector element represents the total amount of a
    specific item consumed or ordered during that window. The resulting vectors
    are clustered with :class:`sklearn.cluster.KMeans`.

    The number of clusters is selected from ``k_candidates`` by maximizing
    clustering stability, measured as the mean pairwise Adjusted Rand Index
    across multiple KMeans runs with different random seeds. Silhouette scores
    are also computed for diagnostics, but are not used for model selection.

    The analyzer stores intermediate fitted attributes, such as vectors, labels,
    cluster centers, selected ``k``, and diagnostic scores. These attributes are
    used by :meth:`draw_figs` and :meth:`build_summary`, so those methods must be
    called only after :meth:`analyze`.

    Attributes:
        name: Name of this analyzer.
        window_size: Number of simulation time steps included in each window.
        total_time_steps: Total number of simulation steps. If ``None``, this is
            inferred from the maximum ``time_step`` in the target records.
        k_candidates: Candidate numbers of clusters.
        exclude_items: Item names excluded from vectorization.
        is_consumption: If ``True``, cluster consumption records. If ``False``,
            cluster order records.
        random_state: Base random seed used for KMeans and dimensionality
            reduction.
        n_stability_runs: Number of KMeans runs used for stability estimation.
        normalize: If ``True``, normalize each non-zero vector to sum to one.
    """

    name: str = "consumer_cluster"

    window_size: int = 10
    total_time_steps: int | None = None
    k_candidates: tuple[int, ...] = (2, 3, 4, 5, 6)
    exclude_items: tuple[str, ...] = ("Yen",)
    is_consumption: bool = True
    random_state: int = 42
    n_stability_runs: int = 10
    normalize: bool = False

    item_list: list[str] | None = None
    selected_k: int | None = None
    cluster_centers_: ConsumerVector | None = None
    labels_: ClusterLabelArray | None = None
    vectors_: ConsumerVector | None = None
    agent_ids_: list[int] | None = None
    window_starts_: list[TimeKey] | None = None
    stability_scores_: dict[int, float] | None = None
    silhouette_scores_: dict[int, float] | None = None

    def analyze(self, store: RecordStore) -> ConsumerClusterResult:
        """Analyze agent-level behavior and return cluster assignments.

        Args:
            store: Record store containing item generation records and either
                consumption records or order records.

        Returns:
            Nested dictionary mapping ``agent_id`` to ``window_start`` to a pair
            of ``(cluster_label, behavior_vector)``.

        Raises:
            ValueError: If no valid target items, records, vectors, or cluster
                candidates are available.
        """
        exclude_items: set[str] = set(self.exclude_items)

        item_names: set[str] = {
            record.item_name
            for record in store.typed(ItemGenerationRecord)
            if record.item_name not in exclude_items
        }
        self.item_list = sorted(item_names)

        if not self.item_list:
            raise ValueError(
                "No target items found. Check ItemGenerationRecord or exclude_items."
            )

        item2idx: dict[str, int] = {
            item_name: item_idx for item_idx, item_name in enumerate(self.item_list)
        }

        if self.is_consumption:
            raw_target_records: list[ConsumptionRecord | OrderRecord] = list(
                store.typed(ConsumptionRecord)
            )
        else:
            raw_target_records = list(store.typed(OrderRecord))

        target_records: list[ConsumptionRecord | OrderRecord] = [
            record for record in raw_target_records if record.item_name in item2idx
        ]

        if not target_records:
            raise ValueError("No consumption/order records found for target items.")

        agent_ids: list[int] = sorted(
            {int(record.agent_id) for record in target_records}
        )

        if self.total_time_steps is None:
            max_time_step: int = max(int(record.time_step) for record in target_records)
            total_time_steps: int = max_time_step + 1
        else:
            total_time_steps = self.total_time_steps

        window_starts_int: list[int] = list(
            range(0, total_time_steps, self.window_size)
        )

        time_step2time: dict[int, TimeKey] = {}

        for record in store.records:
            maybe_time_step: object = getattr(record, "time_step", None)
            maybe_time: object = getattr(record, "time", None)

            if not isinstance(maybe_time_step, int):
                continue

            if not isinstance(maybe_time, int | datetime):
                continue

            time_step2time[maybe_time_step] = maybe_time

        records_by_agent: defaultdict[int, list[ConsumptionRecord | OrderRecord]]
        records_by_agent = defaultdict(list)

        for record in target_records:
            records_by_agent[int(record.agent_id)].append(record)

        vectors: list[ConsumerVector] = []
        vector_agent_ids: list[int] = []
        vector_window_starts: list[TimeKey] = []

        for agent_id in agent_ids:
            agent_records: list[ConsumptionRecord | OrderRecord] = records_by_agent[
                agent_id
            ]

            for start in window_starts_int:
                end: int = start + self.window_size
                vector: ConsumerVector = np.zeros(len(self.item_list), dtype=np.float32)

                for record in agent_records:
                    record_time_step: int = int(record.time_step)
                    if start <= record_time_step < end:
                        item_idx: int = item2idx[record.item_name]
                        vector[item_idx] += np.float32(record.item_amount)

                vector_sum: np.float32 = np.float32(vector.sum())
                if self.normalize and vector_sum > np.float32(0.0):
                    vector = cast(ConsumerVector, vector / vector_sum)

                vectors.append(vector)
                vector_agent_ids.append(agent_id)
                vector_window_starts.append(time_step2time.get(start, start))

        x_matrix: ConsumerVector = np.vstack(vectors).astype(np.float32)

        if len(x_matrix) < 2:
            raise ValueError("At least two vectors are required for clustering.")

        valid_k_candidates: list[int] = [
            k for k in self.k_candidates if 2 <= k < len(x_matrix)
        ]
        if not valid_k_candidates:
            raise ValueError(
                f"No valid k candidates. Got k_candidates={self.k_candidates}, "
                f"but sample size is {len(x_matrix)}."
            )

        stability_scores: dict[int, float] = {}
        silhouette_scores: dict[int, float] = {}

        for k in valid_k_candidates:
            labels_list: list[ClusterLabelArray] = []

            for seed_offset in range(self.n_stability_runs):
                kmeans: KMeans = KMeans(
                    n_clusters=k,
                    random_state=self.random_state + seed_offset,
                    n_init="auto",
                )
                labels: ClusterLabelArray = cast(
                    ClusterLabelArray, kmeans.fit_predict(x_matrix)
                )
                labels_list.append(labels)

            ari_scores: list[float] = []
            for left_idx in range(len(labels_list)):
                for right_idx in range(left_idx + 1, len(labels_list)):
                    ari_score: float = float(
                        adjusted_rand_score(
                            labels_list[left_idx],
                            labels_list[right_idx],
                        )
                    )
                    ari_scores.append(ari_score)

            stability_scores[k] = float(np.mean(ari_scores)) if ari_scores else 0.0

            first_labels: ClusterLabelArray = labels_list[0]
            if len(set(int(label) for label in first_labels)) > 1:
                silhouette_scores[k] = float(silhouette_score(x_matrix, first_labels))
            else:
                silhouette_scores[k] = float("nan")

        self.stability_scores_ = stability_scores
        self.silhouette_scores_ = silhouette_scores
        self.selected_k = max(
            valid_k_candidates,
            key=lambda k: (stability_scores[k], -k),
        )

        final_kmeans: KMeans = KMeans(
            n_clusters=self.selected_k,
            random_state=self.random_state,
            n_init="auto",
        )
        final_labels: ClusterLabelArray = cast(
            ClusterLabelArray, final_kmeans.fit_predict(x_matrix)
        )

        self.cluster_centers_ = final_kmeans.cluster_centers_.astype(np.float32)
        self.labels_ = final_labels
        self.vectors_ = x_matrix
        self.agent_ids_ = vector_agent_ids
        self.window_starts_ = vector_window_starts

        result: ConsumerClusterResult = {}

        for agent_id, window_start, label, vector in zip(
            vector_agent_ids,
            vector_window_starts,
            final_labels,
            x_matrix,
            strict=True,
        ):
            if agent_id not in result:
                result[agent_id] = {}

            result[agent_id][window_start] = (
                int(label),
                vector.astype(np.float32),
            )

        return result

    def draw_figs(self, result: ConsumerClusterResult) -> dict[str, Figure]:
        """Create diagnostic figures for the clustering result.

        Args:
            result: Output returned by :meth:`analyze`.

        Returns:
            Dictionary from figure names to Matplotlib figures.

        Raises:
            RuntimeError: If this method is called before :meth:`analyze`.
        """
        if (
            self.vectors_ is None
            or self.labels_ is None
            or self.agent_ids_ is None
            or self.window_starts_ is None
            or self.cluster_centers_ is None
            or self.item_list is None
            or self.selected_k is None
        ):
            raise RuntimeError("draw_figs must be called after analyze.")

        figures: dict[str, Figure] = {}
        x_matrix: ConsumerVector = self.vectors_

        if len(x_matrix) >= 3 and UMAP is not None:
            reducer: FitTransform2D = cast(
                FitTransform2D,
                UMAP(
                    n_components=2,
                    random_state=self.random_state,
                    n_neighbors=min(15, max(2, len(x_matrix) - 1)),
                ),
            )
            raw_embedding: Any = reducer.fit_transform(x_matrix)
            embedding_title: str = "UMAP"
        else:
            reducer = cast(
                FitTransform2D,
                PCA(n_components=2, random_state=self.random_state),
            )
            raw_embedding = reducer.fit_transform(x_matrix)
            embedding_title = "PCA fallback"

        embedding: Embedding = np.asarray(raw_embedding, dtype=np.float64)

        fig_by_agent: Figure
        ax_by_agent: Axes
        fig_by_agent, ax_by_agent = plt.subplots(figsize=(7, 5))

        scatter_by_agent: PathCollection = ax_by_agent.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=np.asarray(self.agent_ids_, dtype=np.int_),
            alpha=0.8,
        )
        ax_by_agent.set_title(f"Consumer vectors by agent ID ({embedding_title})")
        ax_by_agent.set_xlabel("dim 1")
        ax_by_agent.set_ylabel("dim 2")
        fig_by_agent.colorbar(scatter_by_agent, ax=ax_by_agent, label="agent_id")
        figures["embedding_by_agent"] = fig_by_agent

        time_values: list[float] = []
        for window_start in self.window_starts_:
            if isinstance(window_start, datetime):
                time_values.append(float(window_start.timestamp()))
            else:
                time_values.append(float(window_start))

        fig_by_time: Figure
        ax_by_time: Axes
        fig_by_time, ax_by_time = plt.subplots(figsize=(7, 5))

        # Convert time labels into matplotlib date numbers
        time_numeric: NDArray[np.float64]

        if all(isinstance(t, datetime) for t in self.window_starts_):
            datetime_windows: list[datetime] = [
                cast(datetime, t) for t in self.window_starts_
            ]
            time_numeric = np.asarray(
                date2num(datetime_windows),
                dtype=np.float64,
            )
        else:
            time_numeric = np.asarray(
                [
                    float(t.timestamp() if isinstance(t, datetime) else t)
                    for t in self.window_starts_
                ],
                dtype=np.float64,
            )

        scatter_by_time: PathCollection = ax_by_time.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=time_numeric,
            alpha=0.8,
        )

        ax_by_time.set_title(f"Consumer vectors by time ({embedding_title})")
        ax_by_time.set_xlabel("dim 1")
        ax_by_time.set_ylabel("dim 2")

        cbar = fig_by_time.colorbar(
            scatter_by_time,
            ax=ax_by_time,
        )
        cbar.set_label("window_start")

        # Format datetime ticks nicely
        if all(isinstance(t, datetime) for t in self.window_starts_):
            cbar.ax.yaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

        fig_by_time.tight_layout()

        figures["embedding_by_time"] = fig_by_time

        transition_counts: Counter[tuple[int, int]] = Counter()

        for time2cluster_vec in result.values():
            ordered_items: list[tuple[TimeKey, tuple[int, ConsumerVector]]] = sorted(
                time2cluster_vec.items(),
                key=lambda item: (
                    item[0].timestamp() if isinstance(item[0], datetime) else item[0]
                ),
            )
            ordered_labels: list[int] = [
                cluster_vec[0] for _, cluster_vec in ordered_items
            ]

            for src_label, dst_label in zip(
                ordered_labels[:-1],
                ordered_labels[1:],
                strict=True,
            ):
                transition_counts[(src_label, dst_label)] += 1

        n_clusters: int = self.selected_k
        theta_values: NDArray[np.float64] = np.linspace(
            0.0,
            2.0 * np.pi,
            n_clusters,
            endpoint=False,
        )
        positions: dict[int, NDArray[np.float64]] = {
            cluster_id: np.asarray(
                [np.cos(theta_values[cluster_id]), np.sin(theta_values[cluster_id])],
                dtype=np.float64,
            )
            for cluster_id in range(n_clusters)
        }

        fig_network: Figure
        ax_network: Axes
        fig_network, ax_network = plt.subplots(figsize=(6, 6))
        ax_network.set_title("Cluster transition network")

        for cluster_id, position in positions.items():
            ax_network.scatter(float(position[0]), float(position[1]), s=900)
            ax_network.text(
                float(position[0]),
                float(position[1]),
                str(cluster_id),
                ha="center",
                va="center",
            )

        max_count: int = max(transition_counts.values(), default=1)

        def as_xy(array: NDArray[np.float64]) -> tuple[float, float]:
            """Convert a two-dimensional NumPy vector to a Matplotlib coordinate."""
            return (float(array[0]), float(array[1]))

        for (src_label, dst_label), count in transition_counts.items():
            src_position: NDArray[np.float64] = positions[src_label]
            dst_position: NDArray[np.float64] = positions[dst_label]
            line_width: float = 1.0 + 5.0 * float(count) / float(max_count)

            if src_label == dst_label:
                ax_network.annotate(
                    "",
                    xy=as_xy(src_position + np.asarray([0.05, 0.05])),
                    xytext=as_xy(src_position + np.asarray([-0.05, -0.05])),
                    arrowprops={
                        "arrowstyle": "->",
                        "lw": line_width,
                        "connectionstyle": "arc3,rad=0.4",
                    },
                )
            else:
                ax_network.annotate(
                    "",
                    xy=as_xy(dst_position),
                    xytext=as_xy(src_position),
                    arrowprops={
                        "arrowstyle": "->",
                        "lw": line_width,
                        "alpha": 0.7,
                        "connectionstyle": "arc3,rad=0.15",
                    },
                )

        ax_network.set_aspect("equal")
        ax_network.axis("off")
        figures["cluster_transition_network"] = fig_network

        centers: ConsumerVector = self.cluster_centers_
        n_center_clusters: int = int(centers.shape[0])

        fig_representatives: Figure
        axes_raw: Axes | NDArray[np.object_]
        fig_representatives, axes_raw = plt.subplots(
            n_center_clusters,
            1,
            figsize=(8, max(2.5, 2.0 * n_center_clusters)),
            sharex=True,
        )

        if isinstance(axes_raw, np.ndarray):
            axes: list[Axes] = [cast(Axes, axis) for axis in axes_raw.ravel()]
        else:
            axes = [axes_raw]

        for cluster_id, axis in enumerate(axes):
            axis.bar(self.item_list, centers[cluster_id])
            axis.set_title(f"Cluster {cluster_id}")
            axis.set_ylabel("amount")
            axis.tick_params(axis="x", rotation=45)

        axes[-1].set_xlabel("item")
        fig_representatives.tight_layout()
        figures["cluster_representatives"] = fig_representatives

        return figures

    def build_summary(self, result: ConsumerClusterResult) -> Panel:
        """Build a Rich summary panel for clustering diagnostics.

        The summary includes the clustering mode, sample size, item count,
        selected number of clusters, stability scores, silhouette scores, sample
        counts per cluster, and the most frequent periods for each cluster.

        Args:
            result: Output returned by :meth:`analyze`.

        Returns:
            Rich panel containing a summary table.
        """
        del result

        if self.labels_ is None or self.vectors_ is None:
            return Panel.fit(
                "No clustering result is available.",
                title="Consumer Cluster Summary",
                border_style="yellow",
            )

        labels: ClusterLabelArray = self.labels_
        vectors: ConsumerVector = self.vectors_
        window_starts: list[TimeKey] = self.window_starts_ or []
        item_list: list[str] = self.item_list or []

        table: Table = Table(title="Consumer Cluster Summary")
        table.add_column("Metric")
        table.add_column("Value")

        mode_name: str = "consumption" if self.is_consumption else "purchase/order"
        table.add_row("Mode", mode_name)
        table.add_row("Sample size", str(len(vectors)))
        table.add_row("Number of items", str(len(item_list)))
        table.add_row("Selected k", str(self.selected_k))
        table.add_row("Window size", str(self.window_size))

        if self.stability_scores_ is not None:
            stability_text: str = ", ".join(
                f"k={k}: {score:.3f}" for k, score in self.stability_scores_.items()
            )
            table.add_row("Stability ARI", stability_text)

        if self.silhouette_scores_ is not None:
            silhouette_text: str = ", ".join(
                f"k={k}: {score:.3f}" for k, score in self.silhouette_scores_.items()
            )
            table.add_row("Silhouette", silhouette_text)

        cluster_counts: Counter[int] = Counter(int(label) for label in labels)

        for cluster_id in sorted(cluster_counts):
            table.add_row(
                f"Cluster {cluster_id} samples",
                str(cluster_counts[cluster_id]),
            )

            time_counter: Counter[str] = Counter()

            for label, window_start in zip(labels, window_starts, strict=True):
                if int(label) != cluster_id:
                    continue

                if isinstance(window_start, datetime):
                    window_start_text: str = window_start.strftime("%Y-%m-%d %H:%M")
                else:
                    window_start_text = str(window_start)

                time_counter[window_start_text] += 1

            top_periods: list[tuple[str, int]] = time_counter.most_common(5)
            ranking_text: str = (
                ", ".join(f"{period} ({count})" for period, count in top_periods)
                if top_periods
                else "-"
            )

            table.add_row(
                f"Cluster {cluster_id} top periods",
                ranking_text,
            )

        return Panel.fit(table, title="Analysis Summary", border_style="cyan")
