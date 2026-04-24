from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ..agents import Agent, Government
from ..date_utils import get_corresponding_value
from ..logs import Log, OrderReactionLog
from .base import Event, EventTrigger

if TYPE_CHECKING:
    from ..envs import Environment


class Subsidy4SpecificOrder(Event):
    """Subsidy for specific orders class.

    This event provides subsidies to the agents who purchase
    specific items during specific periods.
    The subsidy amount is calculated as:
        subsidy_amount = accept_amount * subsidy_rate.
    """

    def __init__(
        self,
        trigger: EventTrigger,
        config: dict[str, Any],
    ) -> None:
        """Initialization.

        Args:
            trigger: The trigger for this event. It should be triggered by logs.
            config: The configuration for this event. It should contain:
                - governmentName: The name of the government agent
                    providing the subsidy.
                - itemNames: The list of item names that are eligible for the subsidy.
                - subsidyRates: A list of dictionaries, each containing:
                    - "start": The start time of the subsidy period.
                    - "end": The end time of the subsidy period.
                    - "rate": The subsidy rate for that period.
                    The time can be represented as either
                    int (timestamp) or str (ISO format).
                    For example:
                    [
                        {
                            "start": "2025-03-01 00:00:00",
                            "end": "2025-04-30 23:59:59",
                            "rate": 0.1
                        },
                        {
                            "start": "2025-05-01 00:00:00",
                            "end": "2025-05-31 23:59:59",
                            "rate": 0.15
                        }
                    ]
        """
        super().__init__(trigger, config)
        self._validate_trigger(trigger)
        if "governmentName" not in config:
            raise ValueError(
                "Subsidy4SpecificOrder requires 'governmentName' in config."
            )
        self.gov_name: str = config["governmentName"]
        if "itemNames" not in config:
            raise ValueError("Subsidy4SpecificOrder requires 'itemNames' in config.")
        self.item_names: list[str] = config["itemNames"]
        self.items_validated: bool = False
        if "subsidyRates" not in config:
            raise ValueError("Subsidy4SpecificOrder requires 'subsidyRates' in config.")
        self.period2subsidy_rate: dict[tuple[int | str, int | str], float] = {
            (entry["start"], entry["end"]): entry["rate"]
            for entry in config["subsidyRates"]
        }

    def _validate_trigger(self, trigger: EventTrigger) -> None:
        if trigger.at is not None:
            raise ValueError(
                "Subsidy4SpecificOrder should only be triggered with logs, "
                + f"found at={trigger.at}."
            )
        if trigger.every is not None:
            raise ValueError(
                "Subsidy4SpecificOrder should only be triggered with logs, "
                + f"found every={trigger.every}."
            )
        if trigger.between is not None:
            raise ValueError(
                "Subsidy4SpecificOrder should only be triggered with logs, "
                + f"found between={trigger.between}."
            )
        if len(trigger.logs) == 0:
            raise ValueError(
                "Subsidy4SpecificOrder should be triggered by logs, but no logs found."
            )

    def _validate_item_names(self, env: Environment[Any]) -> None:
        for item_name in self.item_names:
            if item_name not in env.item_name2item:
                raise ValueError(
                    f"Item name '{item_name}' in Subsidy4SpecificOrder config "
                    f"is not found in the environment."
                )
        self.items_validated = True

    def get_current_subsidy_rate(self, current_time: int | str) -> float:
        return get_corresponding_value(
            current_time, self.period2subsidy_rate, default_value=0.0
        )

    def execute(self, env: Environment[Any], log: Optional[Log] = None) -> None:
        """Provide subsidies to the agents who purchase specific items during specific periods.

        Args:
            env: The environment in which this event is executed.
            log: The log that triggers this event. It should be an OrderReactionLog.
        """
        if log is None:
            raise ValueError(
                "Subsidy4SpecificOrder should be triggered by a log, but got None."
            )
        if not self.items_validated:
            self._validate_item_names(env)
        subsidy_rate: float = self.get_current_subsidy_rate(env.get_time())
        if subsidy_rate == 0.0:
            return
        if not isinstance(log, OrderReactionLog):
            raise ValueError(
                "Unexpected log type for Subsidy4SpecificOrder: "
                + f"expected OrderReactionLog, got {type(log)}."
            )
        item_name: str = log.item_name
        accept_amount: float | int = log.accept_amount
        agent_id: int = log.agent_id
        price: float = log.price
        if item_name in self.item_names:
            cash_name: str = env.cash_name
            subsidy_amount: float = accept_amount * price * subsidy_rate
            agent: Agent[Any] = env.agent_id2agent[agent_id]
            government: Agent[Any] = env.agent_id2agent[
                env.agent_name2agent_id[self.gov_name]
            ]
            if not isinstance(government, Government):
                raise ValueError(
                    f"Agent with name '{self.gov_name}' is expected "
                    + f"to be a Government, but got {type(government)}."
                )
            government.exchange_goods(
                give_item_name=cash_name,
                give_item_amount=subsidy_amount,
            )
            agent.exchange_goods(
                get_item_name=cash_name,
                get_item_amount=subsidy_amount,
            )
