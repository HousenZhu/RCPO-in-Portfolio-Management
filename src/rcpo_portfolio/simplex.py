from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimplexDecompositionResult:
    weights: np.ndarray
    branch_weights: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    diagnostics: dict[str, float]


@dataclass(frozen=True)
class SimplexDecomposition:
    num_assets: int
    constraint_1_indices: tuple[int, ...]
    constraint_2_indices: tuple[int, ...]
    constraint_1_min_weight: float
    constraint_2_min_weight: float
    branch_indices: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]

    @property
    def action_dim(self) -> int:
        return int(sum(len(indices) for indices in self.branch_indices))

    def branch_training_mask(
        self,
        epsilon: float = 1e-8,
    ) -> tuple[bool, bool, bool, bool]:
        """Return branches with both portfolio mass and allocation freedom.

        Only z1 is fixed entirely by the two constraint thresholds. The other
        CAOSD coefficients depend on sampled upstream branch allocations.
        """
        z1 = max(
            0.0,
            self.constraint_1_min_weight + self.constraint_2_min_weight - 1.0,
        )
        train_branch_1 = z1 > epsilon and len(self.branch_indices[0]) > 1
        return (train_branch_1, True, True, True)

    def map_logits(self, action: np.ndarray) -> SimplexDecompositionResult:
        logits = np.asarray(action, dtype=np.float32)
        if logits.shape != (self.action_dim,):
            raise ValueError(
                f"Expected simplex-decomposition action shape {(self.action_dim,)}, "
                f"got {logits.shape}."
            )

        padded_branches = [
            self._padded_softmax(branch_action, indices)
            for branch_action, indices in zip(
                self._split_action(logits),
                self.branch_indices,
                strict=True,
            )
        ]
        return self._combine_padded_branches(padded_branches)

    def map_branch_weights(self, action: np.ndarray) -> SimplexDecompositionResult:
        branch_action = np.asarray(action, dtype=np.float32)
        if branch_action.shape != (self.action_dim,):
            raise ValueError(
                f"Expected simplex-decomposition branch-weight action shape "
                f"{(self.action_dim,)}, got {branch_action.shape}."
            )
        padded_branches = [
            self._padded_branch_weights(weights, indices)
            for weights, indices in zip(
                self._split_action(branch_action),
                self.branch_indices,
                strict=True,
            )
        ]
        return self._combine_padded_branches(padded_branches)

    def neutral_branch_weights(self) -> np.ndarray:
        branches: list[np.ndarray] = []
        for indices in self.branch_indices:
            if not indices:
                continue
            branches.append(
                np.full(len(indices), 1.0 / float(len(indices)), dtype=np.float32)
            )
        if not branches:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(branches).astype(np.float32)

    def neutral_padded_branches(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        branches = []
        for indices in self.branch_indices:
            padded = np.zeros(self.num_assets, dtype=np.float32)
            if indices:
                padded[np.asarray(indices, dtype=np.int64)] = 1.0 / float(len(indices))
            branches.append(padded)
        return tuple(branches)  # type: ignore[return-value]

    def _split_action(self, action: np.ndarray) -> list[np.ndarray]:
        branches: list[np.ndarray] = []
        offset = 0
        for indices in self.branch_indices:
            branch_size = len(indices)
            branches.append(action[offset : offset + branch_size])
            offset += branch_size
        return branches

    def _combine_padded_branches(
        self,
        padded_branches: list[np.ndarray],
    ) -> SimplexDecompositionResult:
        y1, y2, y3, y4 = padded_branches
        c1 = float(self.constraint_1_min_weight)
        c2 = float(self.constraint_2_min_weight)
        intersection = set(self.branch_indices[0])
        z1 = max(0.0, c1 + c2 - 1.0)
        z2 = max(0.0, c1 - z1)
        z2_intersection = float(z2 * sum(y2[index] for index in intersection))
        z3 = max(0.0, c2 - z1 - z2_intersection)
        z4 = 1.0 - z1 - z2 - z3
        if z4 < 0.0 and abs(z4) < 1e-7:
            z4 = 0.0
        if z4 < -1e-7:
            raise ValueError(
                "Simplex-decomposition weights became infeasible; check allocation constraints."
            )

        weights = z1 * y1 + z2 * y2 + z3 * y3 + z4 * y4
        weights = np.clip(weights, 0.0, None)
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            raise ValueError("Simplex-decomposition mapping produced zero total weight.")
        weights = (weights / weight_sum).astype(np.float32)

        return SimplexDecompositionResult(
            weights=weights,
            branch_weights=tuple(
                branch.astype(np.float32, copy=True) for branch in padded_branches
            ),
            diagnostics={
                "simplex_z1": float(z1),
                "simplex_z2": float(z2),
                "simplex_z3": float(z3),
                "simplex_z4": float(z4),
                "simplex_z2_intersection": z2_intersection,
            },
        )

    def _padded_softmax(self, logits: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
        padded = np.zeros(self.num_assets, dtype=np.float32)
        if not indices:
            return padded
        centered = logits - np.max(logits)
        branch_weights = np.exp(centered)
        branch_weights = branch_weights / np.sum(branch_weights)
        padded[np.asarray(indices, dtype=np.int64)] = branch_weights.astype(np.float32)
        return padded

    def _padded_branch_weights(
        self,
        branch_weights: np.ndarray,
        indices: tuple[int, ...],
    ) -> np.ndarray:
        padded = np.zeros(self.num_assets, dtype=np.float32)
        if not indices:
            return padded
        weights = np.asarray(branch_weights, dtype=np.float32)
        if weights.shape != (len(indices),):
            raise ValueError(
                f"Expected branch weight shape {(len(indices),)}, got {weights.shape}."
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("Branch weights must be finite.")
        if np.any(weights < -1e-6):
            raise ValueError("Branch weights must be nonnegative.")
        weights = np.clip(weights, 0.0, None)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12:
            raise ValueError("Branch weights must have positive total mass.")
        padded[np.asarray(indices, dtype=np.int64)] = (weights / weight_sum).astype(np.float32)
        return padded


def build_simplex_decomposition(
    *,
    num_assets: int,
    constraint_1_indices: list[int],
    constraint_2_indices: list[int],
    constraint_1_min_weight: float,
    constraint_2_min_weight: float,
) -> SimplexDecomposition:
    c1 = float(constraint_1_min_weight)
    c2 = float(constraint_2_min_weight)
    if num_assets < 1:
        raise ValueError("num_assets must be positive.")
    if not 0.0 <= c1 <= 1.0:
        raise ValueError("constraint_1_min_weight must be between 0 and 1.")
    if not 0.0 <= c2 <= 1.0:
        raise ValueError("constraint_2_min_weight must be between 0 and 1.")

    v1 = _normalize_indices(constraint_1_indices, num_assets, "constraint_1_indices")
    v2 = _normalize_indices(constraint_2_indices, num_assets, "constraint_2_indices")
    if c1 > 0.0 and not v1:
        raise ValueError("constraint_1_indices cannot be empty when constraint 1 is active.")
    if c2 > 0.0 and not v2:
        raise ValueError("constraint_2_indices cannot be empty when constraint 2 is active.")

    k1 = tuple(index for index in v1 if index in set(v2))
    if c1 + c2 > 1.0 + 1e-12 and not k1:
        raise ValueError(
            "Infeasible allocation constraints: thresholds sum above 1 but groups do not overlap."
        )
    all_indices = tuple(range(num_assets))
    return SimplexDecomposition(
        num_assets=num_assets,
        constraint_1_indices=v1,
        constraint_2_indices=v2,
        constraint_1_min_weight=c1,
        constraint_2_min_weight=c2,
        branch_indices=(k1, v1, v2, all_indices),
    )


def _normalize_indices(indices: list[int], num_assets: int, field_name: str) -> tuple[int, ...]:
    normalized: list[int] = []
    seen: set[int] = set()
    for raw_index in indices:
        index = int(raw_index)
        if index < 0 or index >= num_assets:
            raise ValueError(
                f"{field_name} contains invalid asset index {index}; valid range is "
                f"0..{num_assets - 1}."
            )
        if index not in seen:
            normalized.append(index)
            seen.add(index)
    return tuple(sorted(normalized))
