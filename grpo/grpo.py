from typing import Callable
import torch
from einops import rearrange


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float = 1e-5,
    normalize_by_std: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, normalized by the group size.

    Args:
      * reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against
    the ground truths, producing a dict with keys: "reward", "format_reward" and "answer_reward".
      * rollout_responses: list[str] Rollouts from the policy. The length of this list is
    rollout_batch_size = n_prompts_per_rollout_batch * group_size.
      * repeated_ground_truths: list[str] The ground truths for the examples. The length of this
    list is rollout_batch_size, because the ground truth for each example is repeated
    group_size times.
      * group_size: int Number of responses per question (group).
      * advantage_eps: float Small constant to avoid division by zero in normalization.
      * normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise
    subtract only the group mean.

    Returns:
      * advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout
    response.
      * raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout
    response.
      * metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    """
    assert len(rollout_responses) == len(repeated_ground_truths)
    all_rewards: list[dict[str, float]] = [
        reward_fn(rollout, ground_truth)
        for rollout, ground_truth in zip(rollout_responses, repeated_ground_truths)
    ]
    total_rewards: torch.Tensor = torch.tensor([x["reward"] for x in all_rewards])
    total_rewards = rearrange(
        total_rewards,
        "(n_prompts_per_rollout_batch group_size) -> n_prompts_per_rollout_batch group_size",
        group_size=group_size,
    )  # n_prompts_per_rollout_batch group_size
    means = total_rewards.mean(-1, keepdim=True)
    normed_rewards = total_rewards - means
    if normalize_by_std:
        normed_rewards /= total_rewards.std(-1, keepdim=True) + advantage_eps
    metadata = {
        "mean": total_rewards.mean().item(),
        "max": total_rewards.max().item(),
        "min": total_rewards.min().item(),
        "std": total_rewards.std().item(),
    }
    return (normed_rewards.flatten(), total_rewards.flatten(), metadata)
