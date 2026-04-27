from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ..agents import Agent
from ..logs import AgentGenerationLog, Log
from .base import Event, EventTrigger

if TYPE_CHECKING:
    from ..envs import Environment


class ConstantSupply(Event):
    """Constant supply class.

    This event provides a constant supply to the specified agents at regular intervals.
    The supply amount is determined by the configuration and does not depend on any
    other factors.
    """

    def __init__(
        self,
        trigger: EventTrigger,
        config: dict[str, Any],
    ) -> None:
        """Initialization.

        Args:
            trigger: The trigger for this event.
                It should be triggered at regular intervals.
            config: The configuration for this event. It should contain:
                - suppliedAgentNames: The list of agent names who
                    receive the supply.

                Example::

                    "constantSupply": {
                        "type": "ConstantSupply",
                        "trigger": {
                            "with": ["AgentGenerationLog"],
                            "every": 60
                        },
                        "suppliedAgentNames": ["Daily Mart", "QuickBite"]
                    }

        Note:
            The supply amount for each agent is stored in agent_id2supply_dic.
        """
        super().__init__(trigger, config)
        self._validate_trigger(trigger)
        if "suppliedAgentNames" not in config:
            raise ValueError("suppliedAgentNames not found in config.")
        self.supplied_agent_names: list[str] = config["suppliedAgentNames"]
        self.agent_id2supply_dic: dict[int, dict[str, float]] = {}

    def _validate_trigger(self, trigger: EventTrigger) -> None:
        if trigger.at is not None:
            raise ValueError(
                "ConstantSupply should be triggered at regular intervals, "
                + f"found at={trigger.at}."
            )
        if trigger.every is None:
            raise ValueError("ConstantSupply requires 'every' in trigger.")
        if trigger.between is not None:
            raise ValueError(
                "ConstantSupply should be triggered at regular intervals, "
                + f"found between={trigger.between}."
            )
        if len(trigger.logs) == 0:
            raise ValueError(
                "ConstantSupply should be triggered when each agent is generated, "
                "i.e. triggered by AgentGenerationLog, but no logs found."
            )

    def execute(self, env: Environment[Any], log: Optional[Log] = None) -> None:
        """Provide a constant supply to the specified agents at regular intervals.

        Args:
            env: The environment in which the event is executed.
            log: The log that triggered the event.
                It should be an instance of AgentGenerationLog.
                If provided, the supply for the agent will be stored.
                If not provided, the supply will be added to all agents based on the stored supplies.
        """
        cash_name: str = env.cash_name
        agent: Agent[Any]
        if log is not None:
            if not isinstance(log, AgentGenerationLog):
                raise ValueError(
                    "Unexpected log type for ConstantSupply: "
                    + f"expected AgentGenerationLog, got {type(log)}."
                )
            else:
                found: bool = False
                for agent_name in self.supplied_agent_names:
                    if agent_name in log.agent_name:
                        found = True
                        break
                if found:
                    initial_inventory_dic: dict[str, float | int] = {
                        k: v for k, v in log.inventory_dic.items() if k != cash_name
                    }
                    self.agent_id2supply_dic[log.agent_id] = initial_inventory_dic
        else:
            for agent_id, supply_dic in self.agent_id2supply_dic.items():
                agent = env.agent_id2agent[agent_id]
                for item_name, supply_amount in supply_dic.items():
                    agent.inventory_dic[item_name] = (
                        agent.inventory_dic.get(item_name, 0.0) + supply_amount
                    )
