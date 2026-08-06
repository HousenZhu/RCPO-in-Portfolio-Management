from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from rcpo_portfolio.evaluation import (
    apply_signed_percent_axis,
    format_signed_percent,
    relative_wealth_path,
)


def test_relative_wealth_uses_wealth_ratio_not_return_difference() -> None:
    relative = relative_wealth_path(
        np.asarray([1.5], dtype=np.float64),
        np.asarray([1.0], dtype=np.float64),
    )

    assert np.isclose(relative[-1], 0.25)


def test_relative_wealth_handles_positive_and_negative_returns() -> None:
    positive = relative_wealth_path(
        np.asarray([0.25], dtype=np.float64),
        np.asarray([0.20], dtype=np.float64),
    )
    negative = relative_wealth_path(
        np.asarray([-0.40], dtype=np.float64),
        np.asarray([-0.50], dtype=np.float64),
    )

    assert np.isclose(positive[-1], 1.25 / 1.20 - 1.0)
    assert np.isclose(negative[-1], 0.60 / 0.50 - 1.0)


def test_signed_percent_presentation_includes_direction() -> None:
    assert format_signed_percent(0.25) == "+25.00%"
    assert format_signed_percent(-0.0417) == "-4.17%"
    assert format_signed_percent(0.0) == "0.00%"

    figure, axis = plt.subplots()
    apply_signed_percent_axis(axis)
    formatter = axis.yaxis.get_major_formatter()
    assert formatter(0.10, 0) == "+10%"
    assert formatter(-0.05, 0) == "-5%"
    assert formatter(0.0, 0) == "0%"
    plt.close(figure)
