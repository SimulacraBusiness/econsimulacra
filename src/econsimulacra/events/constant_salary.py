from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ..agents import Agent
from ..logs import AgentGenerationLog, Log
from .base import Event, EventTrigger

if TYPE_CHECKING:
    from ..envs import Environment


class ConstantSalary(Event):
    """Constant salary class.

    This event provides a constant salary to the specified agents at regular intervals.
    The salary amount is determined by the configuration and does not depend on any
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
                - unpaidAgentNames: The list of agent names who
                    do not receive the salary.

                Example::

                    "constantSalary": {
                        "type": "ConstantSalary",
                        "trigger": {
                            "with": ["AgentGenerationLog"],
                            "every": 60
                        },
                        "unpaidAgentNames": ["Retailer", "Restaurant", "Government"]
                    }

        Note:
            The salary amount for each agent is stored in agent_id2salary.
        """
        super().__init__(trigger, config)
        self._validate_trigger(trigger)
        if "unpaidAgentNames" not in config:
            raise ValueError("unpaidAgentNames not found in config.")
        self.unpaid_agent_names: list[str] = config["unpaidAgentNames"]
        self.agent_id2salary: dict[int, float] = {}

    def _validate_trigger(self, trigger: EventTrigger) -> None:
        if trigger.at is not None:
            raise ValueError(
                "ConstantSalary should be triggered at regular intervals, "
                + f"found at={trigger.at}."
            )
        if trigger.every is None:
            raise ValueError("ConstantSalary requires 'every' in trigger.")
        if trigger.between is not None:
            raise ValueError(
                "ConstantSalary should be triggered at regular intervals, "
                + f"found between={trigger.between}."
            )
        if len(trigger.logs) == 0:
            raise ValueError(
                "ConstantSalary should be triggered when each agent is generated, "
                "i.e. triggered by AgentGenerationLog, but no logs found."
            )

    def execute(self, env: Environment[Any], log: Optional[Log] = None) -> None:
        """Provide a constant salary to the specified agents at regular intervals.

        Args:
            env: The environment in which the event is executed.
            log: The log that triggered the event.
                It should be an instance of AgentGenerationLog.
                If provided, the salary for the agent will be stored.
                If not provided, the salary will be paid to all agents based on the stored salaries.
        """
        cash_name: str = env.cash_name
        agent: Agent[Any]
        if log is not None:
            if not isinstance(log, AgentGenerationLog):
                raise ValueError(
                    "ConstantSalary should be triggered by AgentGenerationLog, "
                    + f"found log of type {type(log)}."
                )
            agent_id: int = log.agent_id
            inventory_dic: dict[str, float | int] = log.inventory_dic
            self.agent_id2salary[agent_id] = inventory_dic.get(cash_name, 0.0)
        else:
            for agent_id, salary in self.agent_id2salary.items():
                agent = env.agent_id2agent[agent_id]
                agent_name: str = agent.agent_name
                is_unpaid: bool = False
                for unpaid_agent_name in self.unpaid_agent_names:
                    if unpaid_agent_name in agent_name:
                        is_unpaid = True
                        break
                if not is_unpaid:
                    agent.inventory_dic[cash_name] += salary
