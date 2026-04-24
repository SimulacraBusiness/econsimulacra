from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ..agents import Agent, Government
from ..date_utils import get_corresponding_value
from ..logs import Log, OrderReactionLog
from .base import Event, EventTrigger

if TYPE_CHECKING:
    from ..envs import Environment


class ConsumptionTax(Event):
    """Consumption tax class.

    This event levies a consumption tax on all orders.
    When an order is executed, the buyer pays an additional tax amount
    to the government, calculated as
    ``tax_amount = execution_amount * price * tax_rate``.
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
                - taxRates: a list of time-period / rate mappings::

                        [
                            {
                                "start": "2025-03-01 00:00:00",
                                "end":   "2025-04-30 23:59:59",
                                "rate":  0.1
                            },
                            {
                                "start": "2025-05-01 00:00:00",
                                "end":   "2025-05-31 23:59:59",
                                "rate":  0.15
                            }
                        ]
        """
        super().__init__(trigger, config)
        self._validate_trigger(trigger)
        if "governmentName" not in config:
            raise ValueError("ConsumptionTax requires 'governmentName' in config.")
        self.gov_name: str = config["governmentName"]
        if "taxRates" not in config:
            raise ValueError("ConsumptionTax requires 'taxRates' in config.")
        self.period2tax_rates: dict[tuple[int | str, int | str], float] = {
            (entry["start"], entry["end"]): entry["rate"]
            for entry in config["taxRates"]
        }

    def _validate_trigger(self, trigger: EventTrigger) -> None:
        if trigger.at is not None:
            raise ValueError(
                "ConsumptionTax should only be triggered with logs, "
                + f"found at={trigger.at}."
            )
        if trigger.every is not None:
            raise ValueError(
                "ConsumptionTax should only be triggered with logs, "
                + f"found every={trigger.every}."
            )
        if trigger.between is not None:
            raise ValueError(
                "ConsumptionTax should only be triggered with logs, "
                + f"found between={trigger.between}."
            )
        if len(trigger.logs) == 0:
            raise ValueError(
                "ConsumptionTax should be triggered by logs, but no logs found."
            )

    def get_current_tax_rate(self, current_time: int | str) -> float:
        return get_corresponding_value(
            current_time, self.period2tax_rates, default_value=0.0
        )

    def execute(self, env: Environment[Any], log: Optional[Log] = None) -> None:
        """Impose a consumption tax on the order and transfer the tax amount from the buyer to the government.

        Args:
            env: The environment in which the event is executed.
            log: The log that triggered the event. It should be an instance of OrderReactionLog.
        """
        if log is None:
            raise ValueError(
                "ConsumptionTax should be triggered by a log, but got None."
            )
        if not isinstance(log, OrderReactionLog):
            raise ValueError(
                "Unexpected log type for ConsumptionTax: "
                + f"expected OrderReactionLog, got {type(log)}."
            )
        accept_amount: float | int = log.accept_amount
        agent_id: int = log.agent_id
        price: float = log.price
        cash_name: str = env.cash_name
        tax_amount: float = accept_amount * price * self.get_current_tax_rate(log.time)
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
            get_item_name=cash_name,
            get_item_amount=tax_amount,
        )
        agent.exchange_goods(
            give_item_name=cash_name,
            give_item_amount=tax_amount,
        )
