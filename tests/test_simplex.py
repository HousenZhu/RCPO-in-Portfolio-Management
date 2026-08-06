from __future__ import annotations

import numpy as np

from rcpo_portfolio.simplex import build_simplex_decomposition


def test_simplex_decomposition_maps_logits_to_feasible_weights() -> None:
    mapper = build_simplex_decomposition(
        num_assets=6,
        constraint_1_indices=[1, 2, 4],
        constraint_2_indices=[0, 4, 5],
        constraint_1_min_weight=0.45,
        constraint_2_min_weight=0.45,
    )

    result = mapper.map_logits(np.linspace(-1.0, 1.0, mapper.action_dim, dtype=np.float32))
    weights = result.weights

    assert weights.shape == (6,)
    assert np.all(weights >= 0.0)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[[1, 2, 4]].sum() >= 0.45
    assert weights[[0, 4, 5]].sum() >= 0.45
    assert set(result.diagnostics) >= {"simplex_z1", "simplex_z2", "simplex_z3", "simplex_z4"}


def test_simplex_decomposition_maps_branch_weights_to_feasible_weights() -> None:
    mapper = build_simplex_decomposition(
        num_assets=6,
        constraint_1_indices=[1, 2, 4],
        constraint_2_indices=[0, 4, 5],
        constraint_1_min_weight=0.45,
        constraint_2_min_weight=0.45,
    )

    result = mapper.map_branch_weights(mapper.neutral_branch_weights())

    assert result.weights.shape == (6,)
    assert np.all(result.weights >= 0.0)
    assert np.isclose(result.weights.sum(), 1.0)
    assert result.weights[[1, 2, 4]].sum() >= 0.45
    assert result.weights[[0, 4, 5]].sum() >= 0.45


def test_simplex_decomposition_c3_forces_overlap_branch() -> None:
    mapper = build_simplex_decomposition(
        num_assets=6,
        constraint_1_indices=[1, 2, 4],
        constraint_2_indices=[0, 4, 5],
        constraint_1_min_weight=0.55,
        constraint_2_min_weight=0.55,
    )

    result = mapper.map_logits(np.zeros(mapper.action_dim, dtype=np.float32))

    assert mapper.branch_indices[0] == (4,)
    assert np.isclose(result.diagnostics["simplex_z1"], 0.10)
    assert np.isclose(result.diagnostics["simplex_z2"], 0.45)
    assert np.isclose(result.diagnostics["simplex_z3"], 0.30)
    assert np.isclose(result.diagnostics["simplex_z4"], 0.15)
    np.testing.assert_allclose(
        result.weights,
        np.asarray([0.125, 0.175, 0.175, 0.025, 0.375, 0.125], dtype=np.float32),
        atol=1e-6,
    )


def test_simplex_decomposition_rejects_infeasible_disjoint_overlap() -> None:
    try:
        build_simplex_decomposition(
            num_assets=4,
            constraint_1_indices=[1],
            constraint_2_indices=[2],
            constraint_1_min_weight=0.60,
            constraint_2_min_weight=0.60,
        )
    except ValueError as error:
        assert "thresholds sum above 1" in str(error)
    else:
        raise AssertionError("Expected disjoint constraints above total weight 1 to fail.")

def test_branch_training_mask_skips_zero_mass_overlap_branch() -> None:
    mapper = build_simplex_decomposition(
        num_assets=6,
        constraint_1_indices=[1, 2, 3],
        constraint_2_indices=[2, 3, 4],
        constraint_1_min_weight=0.50,
        constraint_2_min_weight=0.40,
    )

    assert mapper.branch_indices[0] == (2, 3)
    assert mapper.branch_training_mask() == (False, True, True, True)


def test_branch_training_mask_enables_positive_multi_asset_overlap_branch() -> None:
    mapper = build_simplex_decomposition(
        num_assets=6,
        constraint_1_indices=[1, 2, 3],
        constraint_2_indices=[2, 3, 4],
        constraint_1_min_weight=0.60,
        constraint_2_min_weight=0.50,
    )

    assert mapper.branch_indices[0] == (2, 3)
    assert mapper.branch_training_mask() == (True, True, True, True)


def test_branch_training_mask_skips_singleton_even_with_positive_mass() -> None:
    mapper = build_simplex_decomposition(
        num_assets=6,
        constraint_1_indices=[1, 2, 4],
        constraint_2_indices=[0, 4, 5],
        constraint_1_min_weight=0.55,
        constraint_2_min_weight=0.55,
    )

    assert mapper.branch_indices[0] == (4,)
    assert mapper.branch_training_mask() == (False, True, True, True)
