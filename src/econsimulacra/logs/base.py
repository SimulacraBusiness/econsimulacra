from typing import Any, Callable, Optional


class Log:
    """Base class for simulation event logs.

    A Log instance represents a single event that occurred during the simulation
    (e.g., agent generation, movement, consumption, orders, proposals, etc.).

    In typical usage, agents or the environment create a log object and record it
    via read_and_write(). The logger stores logs in Logger.pending_logs until
    Logger.process_logs() is called, which dispatches each pending log to a handler
    and then clears the pending list.
    """

    def read_and_write(self, logger: "Logger") -> None:
        """Append this log to the logger's pending logs.

        Args:
            logger (Logger): The logger that stores pending logs.
        """
        logger.write_log(self)

    def to_dict(self) -> dict[str, object]:
        """Convert this log object into a dictionary.

        Returns:
            dict[str, object]: The dictionary representation of the log.
        """
        return self.__dict__


class AgentGenerationLog(Log):
    def __init__(
        self,
        time: int | str,
        time_step: int,
        agent_id: int,
        agent_type: str,
        agent_name: str,
        wealth: float,
        inventory_dic: dict[str, float | int],
        persona_dic: Optional[dict[str, Any]],
    ) -> None:
        """Initialization.

        Args:
            time (int | str): Current time of the environment.
            time_step (int): Current integer step index of the environment.
            agent_id (int): The unique id of the generated agent.
            agent_type (str): The type of the generated agent.
            agent_name (str): The name of the generated agent.
            wealth (float): The initial wealth of the generated agent.
            inventory_dic (dict[str, float | int]): The initial inventory of the generated agent.
                The keys are item names, and the values are the amounts.
            persona_dic (Optional[dict[str, Any]]): The persona details of the generated agent.
                The keys are persona attributes, and the values are the attribute values.

        Note:
            ```wealth``` is a value that represents the agent's total assets
            at the time of generation.

            It is defined as:

            .. math::

                w_i = c_i + \\sum_{j \\in \\mathcal{I}} p_j q_j

            where:

            - :math:`w_i` is the total wealth of agent :math:`i`
            - :math:`c_i` is the initial cash of agent :math:`i`
            - :math:`p_j` is the price of item :math:`j`
            - :math:`q_j` is the quantity of item :math:`j` in the agent's inventory
            - :math:`\\mathcal{I}` is the set of items
        """
        self.type: str = "agent_generation"
        self.time: int | str = time
        self.time_step: int = time_step
        self.agent_id: int = agent_id
        self.agent_type: str = agent_type
        self.agent_name: str = agent_name
        self.wealth: float = wealth
        self.inventory_dic: dict[str, float | int] = inventory_dic.copy()
        self.persona_dic: Optional[dict[str, Any]] = (
            persona_dic.copy() if persona_dic is not None else None
        )

    def to_dict(self) -> dict[str, object]:
        """Convert this log object into a dictionary.

        Overrides the base method to include inventory abd persona details.

        Returns:
            dict[str, object]: The dictionary representation of the log,
                including inventory and persona details.
        """
        d: dict[str, object] = {
            "type": self.type,
            "time": self.time,
            "time_step": self.time_step,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "agent_name": self.agent_name,
            "wealth": self.wealth,
        }
        for item_name, item_amount in self.inventory_dic.items():
            d[f"inventory_{item_name}"] = item_amount
        for persona_key, persona_value in (self.persona_dic or {}).items():
            d[f"persona_{persona_key}"] = persona_value
        return d


class ItemGenerationLog(Log):
    def __init__(
        self,
        time: int | str,
        time_step: int,
        item_name: str,
        price: float,
    ) -> None:
        """Initialization.

        Args:
            time (int | str): Current time of the environment.
            time_step (int): Current integer step index of the environment.
            item_name (str): The name of the generated item.
            price (float): The initial price of the generated item.
        """
        self.type: str = "item_generation"
        self.time: int | str = time
        self.time_step: int = time_step
        self.item_name: str = item_name
        self.price: float = price


class SpaceAssignLog(Log):
    def __init__(self, agent_id: int, pos: tuple[int, ...]) -> None:
        """Initialization.

        Args:
            agent_id (int): The unique id of the agent.
            pos (tuple[int, ...]): The assigned position for the agent. The length of the tuple should match the dimension of the environment's space.
            (e.g., (x, y) for 2D, (x, y, z) for 3D).
        """
        self.type: str = "space_assign"
        self.agent_id: int = agent_id
        self.pos: tuple[int, ...] = pos


class MoveLog(Log):
    def __init__(
        self,
        time: int | str,
        time_step: int,
        agent_id: int,
        old_pos: tuple[int, ...],
        new_pos: tuple[int, ...],
        init_pos: tuple[int, ...],
    ) -> None:
        """Initialization.

        Args:
            time (int | str): Current time of the environment when the move occurs.
            time_step (int): Current integer step index of the environment when the move occurs.
            agent_id (int): The unique id of the agent.
            old_pos (tuple[int, ...]): The previous position of the agent. The length of the tuple should match the dimension of the environment's space.
            new_pos (tuple[int, ...]): The new position of the agent. The length of the tuple should match the dimension of the environment's space.
        """
        self.type: str = "move"
        self.time: int | str = time
        self.time_step: int = time_step
        self.agent_id: int = agent_id
        self.old_pos: tuple[int, ...] = old_pos
        self.new_pos: tuple[int, ...] = new_pos
        self.init_pos: tuple[int, ...] = init_pos


class ConsumptionLog(Log):
    def __init__(
        self,
        time: int | str,
        time_step: int,
        agent_id: int,
        item_name: str,
        item_amount: float | int,
    ) -> None:
        """Initialization.

        Args:
            time (int | str): Current time of the environment when the consumption occurs.
            time_step (int): Current integer step index of the environment when the consumption occurs.
            agent_id (int): The unique id of the agent.
            item_name (str): The name of the consumed item.
            item_amount (float | int): The amount of the consumed item.
        """
        self.type: str = "consumption"
        self.time: int | str = time
        self.time_step: int = time_step
        self.agent_id: int = agent_id
        self.item_name: str = item_name
        self.item_amount: float | int = item_amount


class OrderLog(Log):
    def __init__(
        self,
        time: int | str,
        time_step: int,
        agent_id: int,
        counterparty_id: int,
        item_name: str,
        item_amount: float | int,
        price: Optional[float],
        order_id: Optional[int],
    ) -> None:
        """Initialization.

        Args:
            time (int | str): Current time of the environment when the order was made.
            time_step (int): Current integer step index of the environment when the order was made.
            agent_id (int): The unique id of the agent.
            counterparty_id (int): The unique id of the counterparty.
            item_name (str): The name of the item.
            item_amount (float | int): The amount of the item.
            price (Optional[float]): The price of the item.
            order_id (Optional[int]): The unique id of the order.
        """

        self.type: str = "order"
        self.time: int | str = time
        self.time_step: int = time_step
        self.agent_id: int = agent_id
        self.counterparty_id: int = counterparty_id
        self.item_name: str = item_name
        self.item_amount: float | int = item_amount
        self.price: Optional[float] = price
        self.order_id: Optional[int] = order_id


class ProposalLog(Log):
    def __init__(
        self,
        time: int | str,
        time_step: int,
        proposal_id: int,
        proposer_agent_id: int,
        responder_agent_id: int,
        give_item_name: str,
        give_item_amount: float | int,
        get_item_name: str,
        get_item_amount: float | int,
    ) -> None:
        """Initialization.

        Args:
            time (int | str): Current time of the environment when the proposal is made.
            time_step (int): Current integer step index of the environment when the proposal is made.
            proposal_id (int): The unique id of the proposal.
            proposer_agent_id (int): The unique id of the proposer.
            responder_agent_id (int): The unique id of the responder.
            give_item_name (str): The name of the item to give.
            give_item_amount (float | int): The amount of the item to give.
            get_item_name (str): The name of the item to get.
            get_item_amount (float | int): The amount of the item to get.
        """

        self.type: str = "proposal"
        self.time: int | str = time
        self.time_step: int = time_step
        self.proposal_id: int = proposal_id
        self.proposer_agent_id: int = proposer_agent_id
        self.responder_agent_id: int = responder_agent_id
        self.give_item_name: str = give_item_name
        self.give_item_amount: float | int = give_item_amount
        self.get_item_name: str = get_item_name
        self.get_item_amount: float | int = get_item_amount


class OrderReactionLog(Log):
    def __init__(
        self,
        time: int | str,
        time_step: int,
        agent_id: int,
        counterparty_id: int,
        item_name: str,
        item_amount: float | int,
        price: float,
        order_id: Optional[int],
        accept_amount: float | int,
    ) -> None:
        """Initialization.

        Args:
            time (int | str): Current time of the environment when the order reaction occurs.
            time_step (int): Current integer step index of the environment when the order reaction occurs.
            agent_id (int): The unique id of the agent reacting to the order.
            counterparty_id (int): The unique id of the counterparty in the order.
            item_name (str): The name of the item in the order.
            item_amount (float | int): The amount of the item in the order.
            price (float): The price of the item in the order.
            order_id (Optional[int]): The unique id of the order.
            accept_amount (float | int): The amount accepted in reaction to the order. It can be less than or equal to item_amount.
        """
        self.type: str = "order_reaction"
        self.time: int | str = time
        self.time_step: int = time_step
        self.agent_id: int = agent_id
        self.counterparty_id: int = counterparty_id
        self.item_name: str = item_name
        self.item_amount: float | int = item_amount
        self.price: float = price
        self.order_id: Optional[int] = order_id
        self.accept_amount: float | int = accept_amount


class OrderExpirationLog(Log):
    def __init__(
        self,
        time: int | str,
        time_step: int,
        order_id: int,
    ) -> None:
        """Initialization.

        Args:
            time (int | str): Current time of the environment when the order expires.
            time_step (int): Current integer step index of the environment when the order expires.
            order_id (int): The unique id of the expired order.
        """
        self.type: str = "order_expiration"
        self.time: int | str = time
        self.time_step: int = time_step
        self.order_id: int = order_id


class ProposalReactionLog(Log):
    def __init__(
        self,
        time: int | str,
        time_step: int,
        proposal_id: int,
        proposer_agent_id: int,
        responder_agent_id: int,
        give_item_name: str,
        give_item_amount: float | int,
        get_item_name: str,
        get_item_amount: float | int,
        accept: bool,
    ) -> None:
        """Initialization.

        Args:
            time (int | str): Current time of the environment when the proposal reaction occurs.
            time_step (int): Current integer step index of the environment when the proposal reaction occurs.
            proposal_id (int): The unique id of the proposal.
            proposer_agent_id (int): The unique id of the proposer.
            responder_agent_id (int): The unique id of the responder.
            give_item_name (str): The name of the item the proposer offers to give.
            give_item_amount (float | int): The amount of the item the proposer offers to give.
            get_item_name (str): The name of the item the proposer requests in return.
            get_item_amount (float | int): The amount of the item the proposer requests in return.
            accept (bool): Whether the responder accepted the proposal.
        """
        self.type: str = "proposal_reaction"
        self.time: int | str = time
        self.time_step: int = time_step
        self.proposal_id: int = proposal_id
        self.proposer_agent_id: int = proposer_agent_id
        self.responder_agent_id: int = responder_agent_id
        self.give_item_name: str = give_item_name
        self.give_item_amount: float | int = give_item_amount
        self.get_item_name: str = get_item_name
        self.get_item_amount: float | int = get_item_amount
        self.accept: bool = accept


class ProposalExpirationLog(Log):
    def __init__(
        self,
        time: int | str,
        time_step: int,
        proposal_id: int,
    ) -> None:
        """Initialization.

        Args:
            time (int | str): Current time of the environment when the proposal expires.
            time_step (int): Current integer step index of the environment when the proposal expires.
            proposal_id (int): The unique id of the expired proposal.
        """
        self.type: str = "proposal_expiration"
        self.time: int | str = time
        self.time_step: int = time_step
        self.proposal_id: int = proposal_id


class ChangePriceLog(Log):
    def __init__(
        self,
        time: int | str,
        time_step: int,
        agent_id: int,
        item_name: str,
        old_price: float,
        new_price: float,
    ) -> None:
        """Initialization.

        Args:
            time (int | str): Current time of the environment when the price change occurs.
            time_step (int): Current integer step index of the environment when the price change occurs.
            agent_id (int): The unique id of the agent changing the price.
            item_name (str): The name of the item whose price is changed.
            old_price (float): The previous price of the item.
            new_price (float): The new price of the item.
        """
        self.type: str = "change_price"
        self.time: int | str = time
        self.time_step: int = time_step
        self.agent_id: int = agent_id
        self.item_name: str = item_name
        self.old_price: float = old_price
        self.new_price: float = new_price


class TweetLog(Log):
    def __init__(
        self,
        time: int | str,
        time_step: int,
        agent_id: int,
        message: str,
        num_follows: int,
        num_followers: int,
    ) -> None:
        """Initialization.

        Args:
            time (int | str): Current time of the environment when the tweet is posted.
            time_step (int): Current integer step index of the environment when the tweet is posted.
            agent_id (int): The unique id of the agent posting the tweet.
            message (str): The content of the tweet.
            num_follows (int): The number of agents this agent follows at the time of posting.
            num_followers (int): The number of agents following this agent at the time of posting
        """
        self.type: str = "tweet"
        self.time: int | str = time
        self.time_step: int = time_step
        self.agent_id: int = agent_id
        self.message: str = message
        self.num_follows: int = num_follows
        self.num_followers: int = num_followers


class InnerThoughtLog(Log):
    def __init__(
        self,
        time: int | str,
        time_step: int,
        agent_id: int,
        inner_thought: str,
    ) -> None:
        """Initialization.

        Args:
            time (int | str): Current time of the environment when the inner thought is generated.
            time_step (int): Current integer step index of the environment when the inner thought is generated.
            agent_id (int): The unique id of the agent generating the inner thought.
            inner_thought (str): The content of the inner thought.
        """
        self.type: str = "inner_thought"
        self.time: int | str = time
        self.time_step: int = time_step
        self.agent_id: int = agent_id
        self.inner_thought: str = inner_thought


class FollowLog(Log):
    def __init__(
        self,
        time: int | str,
        time_step: int,
        agent_id: int,
        target_agent_id: int,
        num_follows: int,
        num_followers: int,
    ) -> None:
        """Initialization.

        Args:
            time (int | str): Current time of the environment when the follow action occurs.
            time_step (int): Current integer step index of the environment when the follow action occurs.
            agent_id (int): The unique id of the agent following.
            target_agent_id (int): The unique id of the agent being followed.
            num_follows (int): The number of agents the follower agent follows at the time of following.
            num_followers (int): The number of agents following the target agent at the time of
        """
        self.type: str = "follow"
        self.time: int | str = time
        self.time_step: int = time_step
        self.agent_id: int = agent_id
        self.target_agent_id: int = target_agent_id
        self.num_follows: int = num_follows
        self.num_followers: int = num_followers


class UnfollowLog(Log):
    def __init__(
        self,
        time: int | str,
        time_step: int,
        agent_id: int,
        target_agent_id: int,
        num_follows: int,
        num_followers: int,
    ) -> None:
        """Initialization.

        Args:
            time (int | str): Current time of the environment when the unfollow action occurs.
            time_step (int): Current integer step index of the environment when the unfollow action occurs.
            agent_id (int): The unique id of the agent unfollowing.
            target_agent_id (int): The unique id of the agent being unfollowed.
            num_follows (int): The number of agents the unfollower agent follows at the time of unfollowing.
            num_followers (int): The number of agents following the target agent at the time of unfollowing.
        """
        self.type: str = "unfollow"
        self.time: int | str = time
        self.time_step: int = time_step
        self.agent_id: int = agent_id
        self.target_agent_id: int = target_agent_id
        self.num_follows: int = num_follows
        self.num_followers: int = num_followers


class StateEvaluationLog(Log):
    def __init__(
        self,
        time: int | str,
        time_step: int,
        agent_id: int,
        wealth: float,
        relative_wealth: Optional[float],
        buying_power: Optional[float],
        inventory_dic: dict[str, float | int],
        persona_dic: Optional[dict[str, Any]],
    ) -> None:
        """Initialization.

        Args:
            time (int | str): Current time of the environment when the state evaluation occurs.
            time_step (int): Current integer step index of the environment when the state evaluation occurs.
            agent_id (int): The unique id of the agent being evaluated.
            wealth (float): The wealth of the agent at the time of evaluation.
            relative_wealth (float, optional): The relative wealth of the agent at the time of evaluation.
                Only household agents have this value; for other agent types, it is None.
            buying_power (float, optional): The buying power of the agent at the time of evaluation.
                Only household agents have this value; for other agent types, it is None.
            inventory_dic (dict[str, float | int]): The inventory of the agent at the time of evaluation.
                The keys are item names, and the values are the amounts.
            persona_dic (Optional[dict[str, Any]]): The persona details of the agent at the time of evaluation.
                The keys are persona attributes, and the values are the attribute values.

        Note:
            ```relative_wealth``` is a value that represents the agent's wealth
            relative to others in the environment.

            It is defined as:

            .. math::

                \\text{relative\\_wealth}
                = \\frac{w_i - \\mu_w}{\\sigma_w}

            where:

            - :math:`w_i` is the wealth of agent :math:`i`
            - :math:`\\mu_w` is the average wealth of all agents
            - :math:`\\sigma_w` is the standard deviation of wealth across agents

            ```buying_power``` represents the agent's effective purchasing power in the
            simulation environment.

            It is defined as the weighted sum of the quantities of items that the
            agent can afford with its cash:

            .. math::

                B_i = \\sum_{j \\in \\mathcal{I}} \\alpha_j \\frac{c_i}{p_j}

            where:

            - :math:`B_i` is the buying power of agent :math:`i`
            - :math:`c_i` is the cash held by agent :math:`i`
            - :math:`p_j` is the price of item :math:`j`
            - :math:`\\alpha_j` is the weight of item :math:`j`
            - :math:`\\mathcal{I}` is the set of items
        """
        self.type: str = "state_evaluation"
        self.time: int | str = time
        self.time_step: int = time_step
        self.agent_id: int = agent_id
        self.wealth: float = wealth
        self.relative_wealth: Optional[float] = relative_wealth
        self.buying_power: Optional[float] = buying_power
        self.inventory_dic: dict[str, float | int] = inventory_dic.copy()
        self.persona_dic: Optional[dict[str, Any]] = (
            persona_dic.copy() if persona_dic is not None else None
        )

    def to_dict(self) -> dict[str, object]:
        """Convert this log object into a dictionary. Overrides the base method to include inventory details.

        Returns:
            dict[str, object]: The dictionary representation of the log,
                including inventory and persona details.
        """
        d: dict[str, object] = {
            "type": self.type,
            "time": self.time,
            "time_step": self.time_step,
            "agent_id": self.agent_id,
            "wealth": self.wealth,
        }
        for item_name, item_amount in self.inventory_dic.items():
            d[f"inventory_{item_name}"] = item_amount
        for persona_key, persona_value in (self.persona_dic or {}).items():
            d[f"persona_{persona_key}"] = persona_value
        return d


class ObsLog(Log):
    def __init__(
        self,
        obs_type: str,
        time: int | str,
        time_step: int,
        agent_id: int,
        obs: Any,
    ) -> None:
        """Initialization.

        Args:
            obs_type (str): The type of the observation.
            time (int | str): Current time of the environment when the observation is made.
            time_step (int): Current integer step index of the environment when the observation is made.
            agent_id (int): The unique id of the agent making the observation.
            obs Any: The observation details.

        Note:
            ObsLog is generated by ObsProvider (econsimulacra.envs.ObsProvider) in Environment.get_observation(agent_id).
        """
        self.type: str = "observation"
        self.obs_type: str = obs_type
        self.time: int | str = time
        self.time_step: int = time_step
        self.agent_id: int = agent_id
        self.obs: Any = obs


class Logger:
    """Store pending logs and process them.

    The logger collects logs in pending_logs via write_log() (or Log.read_and_write()).
    When process_logs() is called, each log is dispatched to a handler selected from
    _dispatch_table by its class; otherwise _process_log_default() is used.

    Subclasses can override handlers (e.g., _process_log_default) to implement custom behavior.
    """

    def __init__(self) -> None:
        """Initialization.

        The Logger starts with an empty list of pending logs and an empty dispatch table.
        """
        self.pending_logs: list[Log] = []
        self._dispatch_table: dict[type[Log], Callable] = {}

    def clear(self) -> None:
        """Clear all pending logs."""
        self.pending_logs.clear()

    def write_log(self, log: Log) -> None:
        """Append a log to the pending logs list."""
        self.pending_logs.append(log)

    def process_logs(self) -> None:
        """Process all pending logs.

        Each log is dispatched to a handler selected from _dispatch_table by its class.
        If no handler is registered, _process_log_default() is used.
        After processing, pending_logs is cleared.
        """
        for log in self.pending_logs:
            handler = self._dispatch_table.get(type(log), self._process_log_default)
            handler(log)
        self.pending_logs.clear()

    def _process_log_default(self, log: Log) -> None:
        raise NotImplementedError

    def _process_agent_generation_log(self, log: AgentGenerationLog) -> None:
        pass

    def _process_space_assign_log(self, log: SpaceAssignLog) -> None:
        pass

    def _process_move_log(self, log: MoveLog) -> None:
        pass

    def _process_consumption_log(self, log: ConsumptionLog) -> None:
        pass

    def _process_order_log(self, log: OrderLog) -> None:
        pass

    def _process_proposal_log(self, log: ProposalLog) -> None:
        pass

    def _process_order_reaction_log(self, log: OrderReactionLog) -> None:
        pass

    def _process_order_expiration_log(self, log: OrderExpirationLog) -> None:
        pass

    def _process_proposal_reaction_log(self, log: ProposalReactionLog) -> None:
        pass

    def _process_proposal_expiration_log(self, log: ProposalExpirationLog) -> None:
        pass

    def _process_change_price_log(self, log: ChangePriceLog) -> None:
        pass

    def _process_inner_thought_log(self, log: InnerThoughtLog) -> None:
        pass

    def _process_tweet_log(self, log: TweetLog) -> None:
        pass

    def _process_follow_log(self, log: FollowLog) -> None:
        pass

    def _process_unfollow_log(self, log: UnfollowLog) -> None:
        pass

    def _process_state_evaluation_log(self, log: StateEvaluationLog) -> None:
        pass

    def _process_obs_log(self, log: ObsLog) -> None:
        pass

    def save(self) -> None:
        pass
