from __future__ import annotations


GLOBAL_BRANCH_CREDIT = "global"
SHARED_GLOBAL_BRANCH_CREDIT = "shared_global_branch"
STANDALONE_BRANCH_CREDIT = "standalone"
STANDALONE_REWARD_GLOBAL_COST = "standalone_reward_global_cost"
COUNTERFACTUAL_REWARD_GLOBAL_COST = "counterfactual_open_loop_reward_global_cost"
STANDALONE_REWARD_COUNTERFACTUAL_COST = (
    "standalone_reward_counterfactual_open_loop_cost"
)
COUNTERFACTUAL_REWARD_COST = "counterfactual_open_loop_reward_cost"

VALID_BRANCH_CREDIT_MODES = {
    GLOBAL_BRANCH_CREDIT,
    SHARED_GLOBAL_BRANCH_CREDIT,
    STANDALONE_BRANCH_CREDIT,
    STANDALONE_REWARD_GLOBAL_COST,
    COUNTERFACTUAL_REWARD_GLOBAL_COST,
    STANDALONE_REWARD_COUNTERFACTUAL_COST,
    COUNTERFACTUAL_REWARD_COST,
}

COUNTERFACTUAL_REWARD_MODES = {
    COUNTERFACTUAL_REWARD_GLOBAL_COST,
    COUNTERFACTUAL_REWARD_COST,
}

COUNTERFACTUAL_COST_MODES = {
    STANDALONE_REWARD_COUNTERFACTUAL_COST,
    COUNTERFACTUAL_REWARD_COST,
}

GLOBAL_COST_MODES = {
    SHARED_GLOBAL_BRANCH_CREDIT,
    STANDALONE_REWARD_GLOBAL_COST,
    COUNTERFACTUAL_REWARD_GLOBAL_COST,
}

MATCHED_CREDIT_MODES = {
    SHARED_GLOBAL_BRANCH_CREDIT,
    STANDALONE_REWARD_GLOBAL_COST,
    COUNTERFACTUAL_REWARD_GLOBAL_COST,
    STANDALONE_REWARD_COUNTERFACTUAL_COST,
    COUNTERFACTUAL_REWARD_COST,
}


def uses_branch_credit(mode: str) -> bool:
    return mode != GLOBAL_BRANCH_CREDIT


def uses_counterfactual_reward(mode: str) -> bool:
    return mode in COUNTERFACTUAL_REWARD_MODES


def uses_counterfactual_cost(mode: str) -> bool:
    return mode in COUNTERFACTUAL_COST_MODES


def uses_counterfactual_credit(mode: str) -> bool:
    return uses_counterfactual_reward(mode) or uses_counterfactual_cost(mode)


def uses_global_cost_credit(mode: str) -> bool:
    return mode in GLOBAL_COST_MODES

