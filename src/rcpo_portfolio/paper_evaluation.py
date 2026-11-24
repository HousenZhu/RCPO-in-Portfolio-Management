"""Trajectory accounting and statistical outputs for the project report."""

from __future__ import annotations

import csv
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import torch

from .evaluation import plt
from .market import MarketSlice


METHODS = {
    "constrained_neutral": "Constrained-neutral",
    "rcpo_allocation_penalty": "RCPO with allocation penalties",
    "caosd_ppo": "CAOSD + PPO",
    "caosd_rcpo_shared": "CAOSD + RCPO (shared global)",
    "caosd_rcpo_standalone": "CAOSD + RCPO (standalone)",
    "caosd_rcpo_cf_r": "CAOSD + RCPO (CF-R)",
    "caosd_rcpo_cf_c": "CAOSD + RCPO (CF-C)",
    "caosd_rcpo_cf_rc": "CAOSD + RCPO (CF-RC)",
}
BASELINE = "constrained_neutral"
STANDALONE = "caosd_rcpo_standalone"
CREDIT_METHODS = tuple(METHODS)[3:]
OVERALL_METHODS = (*tuple(METHODS)[:3], STANDALONE)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def slice_market(market: MarketSlice, lookback: int, horizon: int) -> MarketSlice:
    return MarketSlice(
        market.risky_returns[-(lookback + horizon):],
        market.regimes[-(lookback + horizon):],
    )


def rollout(env, policy, horizon: int) -> dict[str, np.ndarray]:
    obs, _ = env.reset(options={"start_index": env.lookback})
    wealth, turnovers, net_returns, weights = [], [], [], []
    for _ in range(horizon):
        obs, _, terminated, truncated, info = env.step(policy(obs))
        wealth.append(info["portfolio_value"])
        turnovers.append(info["turnover"])
        net_returns.append(info["net_return"])
        weights.append(info["weights"])
        if terminated or truncated:
            break
    if len(wealth) != horizon or not terminated or truncated:
        raise ValueError(f"Expected a complete {horizon}-step evaluation episode.")
    return _verified_path(wealth, turnovers, net_returns, weights)


def _verified_path(wealth, turnovers, net_returns, weights):
    result = {
        "wealth": np.r_[1.0, np.asarray(wealth, dtype=np.float64)],
        "turnover": np.asarray(turnovers, dtype=np.float64),
        "net_return": np.asarray(net_returns, dtype=np.float64),
        "weights": np.asarray(weights, dtype=np.float64),
    }
    if not all(np.all(np.isfinite(values)) for values in result.values()):
        raise ValueError("Evaluation trajectory contains nonfinite values.")
    if np.any(result["wealth"] <= 0):
        raise ValueError("Evaluation wealth must be positive.")
    np.testing.assert_allclose(
        result["wealth"][1:],
        np.cumprod(np.maximum(1e-12, 1 + result["net_return"])),
        rtol=1e-10,
        atol=0,
    )
    return result


@torch.inference_mode()
def batched_rollout(environments, model, horizon: int) -> list[dict[str, np.ndarray]]:
    """Batch deterministic policy inference on a set of equal-length markets."""
    observations = [env.reset(options={"start_index": env.lookback})[0] for env in environments]
    count = len(environments)
    wealth, turnover, returns = (np.empty((count, horizon)) for _ in range(3))
    weights = np.empty((count, horizon, environments[0].num_assets))
    device = next(model.parameters()).device
    for t in range(horizon):
        inputs = torch.as_tensor(np.stack(observations), device=device, dtype=torch.float32)
        actions = model.get_policy_output(inputs, deterministic=True).action.cpu().numpy()
        for i, env in enumerate(environments):
            observations[i], _, terminated, truncated, info = env.step(actions[i])
            if truncated or terminated != (t == horizon - 1):
                raise ValueError(f"Expected a complete {horizon}-step evaluation episode.")
            wealth[i, t] = info["portfolio_value"]
            turnover[i, t] = info["turnover"]
            returns[i, t] = info["net_return"]
            weights[i, t] = info["weights"]
    return [_verified_path(wealth[i], turnover[i], returns[i], weights[i]) for i in range(count)]


def drawdown(wealth: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(wealth)
    return ((peak - wealth) / np.maximum(peak, 1e-12))[1:]


def score_trajectory(path, baseline, *, risk, groups, cost_tolerance, allocation_tolerance):
    wealth = path["wealth"]
    bw = baseline["wealth"]
    dd = drawdown(wealth)
    benchmark_dd = drawdown(bw)
    budget = np.maximum(risk["floor"], risk["margin"] * benchmark_dd)
    cost = np.maximum(0.0, dd - budget) ** 2 / risk["cost_scale"]
    allowance = (risk["alpha_budget_ratio"] * budget) ** 2 / risk["cost_scale"]
    weights = path["weights"]
    allocation_checks = {
        "negative_weight_violation": np.maximum(0, -weights.min(axis=1)),
        "weight_sum_violation": np.abs(weights.sum(axis=1) - 1),
        **{
            f"group_{i}_violation": np.maximum(0, target - weights[:, indices].sum(axis=1))
            for i, (indices, target) in enumerate(groups, start=1)
        },
    }
    violation = np.maximum.reduce(list(allocation_checks.values()))
    gap = float(np.mean(cost - allowance))
    summary = {
        "episode_length": len(dd),
        "initial_wealth": float(wealth[0]),
        "terminal_wealth": float(wealth[-1]),
        "baseline_initial_wealth": float(bw[0]),
        "baseline_terminal_wealth": float(bw[-1]),
        "actual_cumulative_return": float(wealth[-1] / wealth[0] - 1),
        "relative_wealth": float((wealth[-1] / wealth[0]) / (bw[-1] / bw[0]) - 1),
        "mean_cost": float(cost.mean()),
        "mean_cost_allowance": float(allowance.mean()),
        "cost_gap": gap,
        "path_feasible": gap <= cost_tolerance,
        "max_drawdown": float(dd.max()),
        "average_turnover": float(path["turnover"].mean()),
        "mean_concentration": float(np.square(weights).sum(axis=1).mean()),
        "allocation_violation_step_count": int(np.count_nonzero(violation > allocation_tolerance)),
        "max_allocation_violation": float(violation.max()),
    }
    steps = {
        **path, "baseline_wealth": bw, "growth": wealth[1:] / wealth[:-1],
        "drawdown": dd, "benchmark_drawdown": benchmark_dd, "budget": budget,
        "cost": cost, "allowance": allowance, "allocation_violation": violation,
        **allocation_checks,
    }
    return summary, steps


def summarize(rows: list[dict], fields: tuple[str, ...], bootstrap: int, seed: int) -> dict:
    """Resample training seeds and shared anchor blocks after averaging action repeats."""
    result = {}
    if not rows:
        return result
    cells = defaultdict(list)
    for row in rows:
        cells[(row["training_seed"], row["anchor_seed"], row["market_id"])].append(
            [float(row[field]) for field in fields]
        )
    seeds = sorted({key[0] for key in cells})
    anchors = sorted({key[1] for key in cells})
    blocks = defaultdict(list)
    for (training_seed, anchor, _), values in cells.items():
        blocks[training_seed, anchor].append(np.mean(values, axis=0))
    if set(product(seeds, anchors)) != blocks.keys():
        raise ValueError("Each included training seed must cover all anchor blocks.")
    sizes = np.asarray([[len(blocks[s, a]) for a in anchors] for s in seeds])
    if not np.all(sizes == sizes[0]):
        raise ValueError("Training seeds must cover equally sized market blocks.")
    sums = np.asarray([[np.sum(blocks[s, a], axis=0) for a in anchors] for s in seeds])
    means = (sums.sum(axis=1) / sizes.sum(axis=1)[:, None]).mean(axis=0)
    rng = np.random.default_rng(seed)
    sampled_seeds = rng.integers(len(seeds), size=(bootstrap, len(seeds), 1))
    sampled_anchors = rng.integers(len(anchors), size=(bootstrap, 1, len(anchors)))
    resampled = (
        sums[sampled_seeds, sampled_anchors].sum(axis=2)
        / sizes[sampled_seeds, sampled_anchors].sum(axis=2)[..., None]
    ).mean(axis=1)
    low, high = np.percentile(resampled, [2.5, 97.5], axis=0)
    has_repeats = len(seeds) > 1 or len(anchors) > 1
    for i, field in enumerate(fields):
        result[f"mean_{field}"] = float(means[i])
        identifiable_zero = (
            field in {"relative_wealth", "delta_relative_wealth", "delta_cost_gap"}
            and np.all(sums[:, :, i] == 0)
        )
        result[f"ci_low_{field}"] = float(low[i]) if has_repeats or identifiable_zero else None
        result[f"ci_high_{field}"] = float(high[i]) if has_repeats or identifiable_zero else None
    return result


def paired_differences(rows: list[dict], reference: list[dict]) -> list[dict]:
    def index(records):
        matches = {
            (r["training_seed"], r["market_id"], r["action_seed"]): r for r in records
        }
        if len(matches) != len(records):
            raise ValueError("Duplicate seed, market and action-repeat pair.")
        return matches

    left = index(rows)
    right = index(reference)
    if left.keys() != right.keys():
        raise ValueError("Table 2 requires identical seed, market and action-repeat pairs.")
    return [
        {
            **row,
            "delta_relative_wealth": row["relative_wealth"] - right[key]["relative_wealth"],
            "delta_cost_gap": row["cost_gap"] - right[key]["cost_gap"],
        }
        for key, row in left.items()
    ]


def plot_learning(rows: list[dict], baseline: dict, output: Path) -> None:
    colors = ("#252525", "#98733F", "#587DA3", "#A44638", "#218175", "#75549A", "#B96535", "#577F42")
    styles = ("--", "-.", ":", "--", "-", "-.", ":", "-")
    style = dict(zip(METHODS, zip(colors, styles)))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey="row", layout="constrained")
    for i, (row_axes, metric, scale) in enumerate(
        zip(axes, ("relative_wealth", "cost_gap"), (100, 1000))
    ):
        for j, (ax, methods) in enumerate(zip(row_axes, (OVERALL_METHODS, CREDIT_METHODS))):
            for method in methods:
                color, line = style[method]
                series = sorted(
                    (row for row in rows if row["algorithm_id"] == method),
                    key=lambda row: row["training_real_transition_count"],
                )
                if method == BASELINE:
                    ax.axhline(scale * baseline[f"mean_{metric}"], color=color, ls=line,
                               label=METHODS[method])
                elif series:
                    x = np.asarray([r["training_real_transition_count"] for r in series]) / 1e6
                    def values(field):
                        return scale * np.asarray(
                            [r.get(field, np.nan) for r in series], dtype=float
                        )
                    counts = sorted({r["effective_seed_count"] for r in series if r["effective_seed_count"]})
                    seed_label = f"n={counts[0]}" if len(counts) == 1 else (
                        f"n={counts[0]}–{counts[-1]}" if counts else "missing"
                    )
                    ax.plot(x, values(f"mean_{metric}"), color=color, ls=line,
                            marker=".", label=f"{METHODS[method]} ({seed_label})")
                    ax.fill_between(x, values(f"ci_low_{metric}"), values(f"ci_high_{metric}"),
                                    color=color, alpha=0.10)
                else:
                    ax.plot([], [], color=color, ls=line, label=f"{METHODS[method]} (missing)")
            ax.set_title(f"({chr(97 + 2*i + j)}) " + ("Overall comparison" if j == 0 else "Credit comparison"))
            ax.grid(alpha=0.2)
            if i:
                ax.axhline(0, color="#999999", lw=0.8, zorder=0)
                ax.set_xlabel("Real environment transitions (M)")
            if not j:
                ax.set_ylabel("Relative wealth (%)" if i == 0 else r"Mean cost gap (×$10^{-3}$)")
            if i == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Validation relative wealth and path-cost gap during training")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"figure_1_credit_learning.{extension}", dpi=220)
    plt.close(fig)
