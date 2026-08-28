from __future__ import annotations

from collections import defaultdict
from typing import Optional, cast

from .memory_items import ObsHistoryItem


def summarize_observed_price_changes(
    obs_items: list[ObsHistoryItem],
    relative_threshold: float = 0.01,
) -> str:
    """Summarizes significant price changes in the observed inventory over time.

    Args:
        obs_items (list[ObsHistoryItem]): A list of observation items to summarize.

    Returns:
        str: A summarized string of inventory price changes.

    Note:
        The components in obs_items must be: ObsHistoryItem with obs_type=='others_inventory'.
        The function identifies significant price changes and summarizes them in a human-readable format.
        If no significant changes are found, it returns an empty string.
    """
    if len(obs_items) < 2:
        return "No significant price changes observed."
    item2prices: dict[str, list[float]] = defaultdict(list)
    for obs_item in obs_items:
        if obs_item.obs_type != "others_inventory":
            raise ValueError(
                f"Expected obs_type 'others_inventory', but got '{obs_item.obs_type}'"
            )
        for other_agent_inventory in obs_item.obs:
            if not isinstance(other_agent_inventory, dict):
                raise ValueError(
                    f"Expected inventory item to be a dict, but got {other_agent_inventory}"
                )
            for key, value in other_agent_inventory.items():
                if key in {"agent_id", "agent_name", "is_household"}:
                    continue
                elif isinstance(value, dict):
                    item_name: str = key
                    d: dict[str, str | int | float] = cast(
                        dict[str, str | int | float], value
                    )
                    price = d["price"]
                    item2prices[item_name].append(float(price))
                else:
                    raise ValueError(
                        f"Expected inventory item value to be a dict, but got {value}"
                    )
    messages: list[str] = []
    for item_name, prices in item2prices.items():
        if len(prices) < 2:
            continue
        min_price = min(prices)
        max_price = max(prices)
        final_price = prices[-1]
        if min_price == 0:
            continue
        price_increased = (final_price - min_price) / min_price
        price_decreased = (max_price - final_price) / max_price
        if price_increased >= relative_threshold:
            messages.append(
                f"The price of {item_name} increased from {min_price:.1f} to {final_price:.1f}"
            )
        elif price_decreased >= relative_threshold:
            messages.append(
                f"The price of {item_name} decreased from {max_price:.1f} to {final_price:.1f}"
            )
        else:
            continue
    return (
        "; ".join(messages) + "."
        if messages
        else "No significant price changes observed."
    )


def summarize_num_changes(
    obs_items: list[ObsHistoryItem],
    is_follow: bool = True,
) -> str:
    """Summarizes significant changes in the number of follows/followers over time.

    Args:
        obs_items (list[ObsHistoryItem]): A list of observation items to summarize.

    Returns:
        str: A summarized string of follows changes.

    Note:
        The components in obs_items must be: ObsHistoryItem with obs_type=='num_follows' or 'num_followers'.
    """
    if not obs_items:
        return ""
    sorted_items: list[ObsHistoryItem] = sorted(
        obs_items, key=lambda item: item.time_step
    )
    oldest_item: ObsHistoryItem = sorted_items[0]
    latest_item: ObsHistoryItem = sorted_items[-1]
    oldest_num: int = int(oldest_item.obs)
    latest_num: int = int(latest_item.obs)
    diff = latest_num - oldest_num
    if is_follow:
        target = "people you follow"
    else:
        target = "followers"
    if diff > 0:
        return (
            f"From time {oldest_item.time} to {latest_item.time}, "
            f"you have increased the number of {target} "
            f"from {oldest_num} to {latest_num} "
            f"(+{diff})."
        )
    elif diff < 0:
        return (
            f"From time {oldest_item.time} to {latest_item.time}, "
            f"you have decreased the number of {target} "
            f"from {oldest_num} to {latest_num} "
            f"({diff})."
        )
    else:
        return (
            f"From time {oldest_item.time} to {latest_item.time}, "
            f"you have not changed the number of {target} "
            f"at {latest_num}."
        )


def summarize_self_tweet_frequency(obs_items: list[ObsHistoryItem]) -> str:
    """Summarizes the frequency of the agent's own tweets over time.

    Args:
        obs_items (list[ObsHistoryItem]): A list of observation items to summarize.

    Returns:
        str: A summary of the agent's tweet frequency.

    Note:
        The components in obs_items must be: ObsHistoryItem with obs_type=='self_tweet'.
        - obs == None means the agent has never tweeted yet.
        - obs stores the latest tweet content.
        - If obs is unchanged from the previous observation,
          it means no new tweet was posted.
    """
    if not obs_items:
        return ""
    sorted_items: list[ObsHistoryItem] = sorted(
        obs_items, key=lambda item: item.time_step
    )
    tweet_count: int = 0
    prev_obs: Optional[str] = None
    for item in sorted_items:
        current_obs: Optional[str] = item.obs
        if current_obs is None:
            continue
        if prev_obs is None:
            tweet_count += 1
        elif current_obs != prev_obs:
            tweet_count += 1
        prev_obs = current_obs
    num_steps: int = len(sorted_items)
    if tweet_count == 0:
        return f"You have not posted any tweets in the last {num_steps} steps."
    freq: float = tweet_count / max(num_steps, 1)
    return (
        f"You have posted {tweet_count} tweets in the last "
        f"{num_steps} steps, with a frequency of {freq:.2f} tweets per step."
    )
