import pytest
from rich.console import Console

from econsimulacra.log_analyses import StressAnalyzer


def _render_summary(analyzer: StressAnalyzer, result: dict) -> str:
    """Render a stress summary as plain text for assertions.

    Args:
        analyzer (StressAnalyzer): Analyzer that builds the summary.
        result (dict): Stress analysis result to render.

    Returns:
        str: Plain-text representation of the Rich summary.

    Note:
        A fixed console width keeps table rendering deterministic.
    """
    console = Console(record=True, width=120)
    console.print(analyzer.build_summary(result))
    return console.export_text()


def test_build_summary_displays_high_stress_metrics() -> None:
    """Test exposure and persistence values shown in the summary.

    Args:
        None.

    Returns:
        None: This test returns no value.

    Note:
        For normalized stresses ``[0.8, 0.8, 0.2, 0.9]`` and threshold
        ``0.7``, exposure is ``0.1`` and persistence is ``1.25``.
    """
    analyzer = StressAnalyzer()
    result = {
        4: {
            "move": {
                3: 20.0,
                0: 80.0,
                1: 80.0,
                4: 90.0,
            }
        }
    }

    rendered = _render_summary(analyzer, result)

    assert "HighStressExposure" in rendered
    assert "StressPersistence" in rendered
    assert "move" in rendered
    assert "0.1000" in rendered
    assert "1.2500" in rendered


def test_high_stress_metrics_weight_consecutive_exposure() -> None:
    """Test that consecutive high stress increases persistence.

    Args:
        None.

    Returns:
        None: This test returns no value.

    Note:
        Both series have equal exposure, but only the first contains a
        two-observation consecutive high-stress period.
    """
    analyzer = StressAnalyzer()
    consecutive = {0: 80.0, 1: 80.0, 2: 20.0, 3: 20.0}
    intermittent = {0: 80.0, 1: 20.0, 2: 80.0, 3: 20.0}

    consecutive_metrics = analyzer._calc_high_stress_metrics(consecutive)
    intermittent_metrics = analyzer._calc_high_stress_metrics(intermittent)

    assert consecutive_metrics[0] == intermittent_metrics[0]
    assert consecutive_metrics[1] == 1.5
    assert intermittent_metrics[1] == 1.0


def test_high_stress_metrics_are_zero_without_exceedance() -> None:
    """Test the zero-exposure case without division by zero.

    Args:
        None.

    Returns:
        None: This test returns no value.

    Note:
        A value equal to the threshold is not an exceedance.
    """
    analyzer = StressAnalyzer()

    assert analyzer._calc_high_stress_metrics({0: 10.0, 1: 70.0}) == (0.0, 0.0)
    assert analyzer._calc_high_stress_metrics({}) == (0.0, 0.0)


def test_high_stress_metrics_use_configured_scale_and_threshold() -> None:
    """Test custom stress normalization and high-stress threshold.

    Args:
        None.

    Returns:
        None: This test returns no value.

    Note:
        With maximum ``10`` and threshold ``0.5``, values ``6`` and ``7``
        have normalized exceedances ``0.1`` and ``0.2``.
    """
    analyzer = StressAnalyzer(max_stress=10.0, high_stress_threshold=0.5)

    exposure, persistence = analyzer._calc_high_stress_metrics({0: 6.0, 1: 7.0})

    assert exposure == pytest.approx(0.15)
    assert persistence == pytest.approx(5.0 / 3.0)


def test_high_stress_metrics_clamp_normalized_values() -> None:
    """Test normalization remains in the documented unit interval.

    Args:
        None.

    Returns:
        None: This test returns no value.

    Note:
        Out-of-range source data is clamped before applying the metric.
    """
    analyzer = StressAnalyzer(max_stress=100.0, high_stress_threshold=0.7)

    exposure, persistence = analyzer._calc_high_stress_metrics({0: -10.0, 1: 120.0})

    assert exposure == pytest.approx(0.15)
    assert persistence == pytest.approx(1.0)


def test_stress_analyzer_validates_metric_configuration() -> None:
    """Test validation of stress normalization and threshold settings.

    Args:
        None.

    Returns:
        None: This test returns no value.

    Note:
        Both ends of the normalized threshold interval are accepted.
    """
    for max_stress in (0.0, -1.0):
        with pytest.raises(ValueError, match="max_stress must be positive"):
            StressAnalyzer(max_stress=max_stress)

    for threshold in (-0.1, 1.1):
        with pytest.raises(ValueError, match="high_stress_threshold"):
            StressAnalyzer(high_stress_threshold=threshold)

    StressAnalyzer(high_stress_threshold=0.0)
    StressAnalyzer(high_stress_threshold=1.0)


def test_build_summary_handles_empty_result() -> None:
    """Test summary rendering when no stress data is available.

    Args:
        None.

    Returns:
        None: This test returns no value.

    Note:
        Empty analysis results should remain renderable.
    """
    rendered = _render_summary(StressAnalyzer(), {})

    assert "No stress result is available." in rendered
