from typing import Callable
from typing import Optional


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
        time: int,
        agent_id: int,
        agent_type: str,
        agent_name: str,
        inventory_dic: dict[str, float | int],
    ) -> None:
        """Initialization.

        Args:
            time (int): The time step when the agent is generated.
            agent_id (int): The unique id of the generated agent.
            agent_type (str): The type of the generated agent.
            agent_name (str): The name of the generated agent.
            inventory_dic (dict[str, float | int]): The initial inventory of the generated agent. The keys are item names, and the values are the amounts.
        """
        self.type: str = "agent_generation"
        self.time: int = time
        self.agent_id: int = agent_id
        self.agent_type: str = agent_type
        self.agent_name: str = agent_name
        self.inventory_dic: dict[str, float | int] = inventory_dic.copy()

    def to_dict(self) -> dict[str, object]:
        """Convert this log object into a dictionary. Overrides the base method to include inventory details.

        Returns:
            dict[str, object]: The dictionary representation of the log, including inventory details.
        """
        d: dict[str, object] = {
            "type": self.type,
            "time": self.time,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "agent_name": self.agent_name,
        }
        for item_name, item_amount in self.inventory_dic.items():
            d[f"inventory_{item_name}"] = item_amount
        return d


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
        time: int,
        agent_id: int,
        old_pos: tuple[int, ...],
        new_pos: tuple[int, ...],
    ) -> None:
        """Initialization.

        Args:
            time (int): The time step when the move occurs.
            agent_id (int): The unique id of the agent.
            old_pos (tuple[int, ...]): The previous position of the agent. The length of the tuple should match the dimension of the environment's space.
            new_pos (tuple[int, ...]): The new position of the agent. The length of the tuple should match the dimension of the environment's space.
        """
        self.type: str = "move"
        self.time: int = time
        self.agent_id: int = agent_id
        self.old_pos: tuple[int, ...] = old_pos
        self.new_pos: tuple[int, ...] = new_pos


class ConsumptionLog(Log):
    def __init__(
        self, time: int, agent_id: int, item_name: str, item_amount: float | int
    ) -> None:
        """Initialization.

        Args:
            time (int): The time step when the consumption occurs.
            agent_id (int): The unique id of the agent.
            item_name (str): The name of the consumed item.
            item_amount (float | int): The amount of the consumed item.
        """
        self.type: str = "consumption"
        self.time: int = time
        self.agent_id: int = agent_id
        self.item_name: str = item_name
        self.item_amount: float | int = item_amount


class OrderLog(Log):
    def __init__(
        self,
        time: int,
        agent_id: int,
        counterparty_id: int,
        item_name: str,
        item_amount: float | int,
        price: Optional[float],
        order_id: Optional[int],
    ) -> None:
        """Initialization.

        Args:
            time (int): The time step when the order was made.
            agent_id (int): The unique id of the agent.
            counterparty_id (int): The unique id of the counterparty.
            item_name (str): The name of the item.
            item_amount (float | int): The amount of the item.
            price (Optional[float]): The price of the item.
            order_id (Optional[int]): The unique id of the order.
        """

        self.type: str = "order"
        self.time: int = time
        self.agent_id: int = agent_id
        self.counterparty_id: int = counterparty_id
        self.item_name: str = item_name
        self.item_amount: float | int = item_amount
        self.price: Optional[float] = price
        self.order_id: Optional[int] = order_id


class ProposalLog(Log):
    def __init__(
        self,
        time: int,
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
            time (int): The time step when the proposal is made.
            proposal_id (int): The unique id of the proposal.
            proposer_agent_id (int): The unique id of the proposer.
            responder_agent_id (int): The unique id of the responder.
            give_item_name (str): The name of the item to give.
            give_item_amount (float | int): The amount of the item to give.
            get_item_name (str): The name of the item to get.
            get_item_amount (float | int): The amount of the item to get.
        """

        self.type: str = "proposal"
        self.time: int = time
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
        time: int,
        agent_id: int,
        counterparty_id: int,
        item_name: str,
        item_amount: float | int,
        price: Optional[float],
        order_id: Optional[int],
        accept_amount: float | int,
    ) -> None:
        """Initialization.

        Args:
            time (int): The time step when the order reaction occurs.
            agent_id (int): The unique id of the agent reacting to the order.
            counterparty_id (int): The unique id of the counterparty in the order.
            item_name (str): The name of the item in the order.
            item_amount (float | int): The amount of the item in the order.
            price (Optional[float]): The price of the item in the order.
            order_id (Optional[int]): The unique id of the order.
            accept_amount (float | int): The amount accepted in reaction to the order. It can be less than or equal to item_amount.
        """
        self.type: str = "order_reaction"
        self.time: int = time
        self.agent_id: int = agent_id
        self.counterparty_id: int = counterparty_id
        self.item_name: str = item_name
        self.item_amount: float | int = item_amount
        self.price: Optional[float] = price
        self.order_id: Optional[int] = order_id
        self.accept_amount: float | int = accept_amount


class ProposalReactionLog(Log):
    def __init__(
        self,
        time: int,
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
            time (int): The time step when the proposal reaction occurs.
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
        self.time: int = time
        self.proposal_id: int = proposal_id
        self.proposer_agent_id: int = proposer_agent_id
        self.responder_agent_id: int = responder_agent_id
        self.give_item_name: str = give_item_name
        self.give_item_amount: float | int = give_item_amount
        self.get_item_name: str = get_item_name
        self.get_item_amount: float | int = get_item_amount
        self.accept: bool = accept


class ChangePriceLog(Log):
    def __init__(
        self,
        time: int,
        agent_id: int,
        item_name: str,
        old_price: float,
        new_price: float,
    ) -> None:
        """Initialization.

        Args:
            time (int): The time step when the price change occurs.
            agent_id (int): The unique id of the agent changing the price.
            item_name (str): The name of the item whose price is changed.
            old_price (float): The previous price of the item.
            new_price (float): The new price of the item.
        """
        self.type: str = "change_price"
        self.time: int = time
        self.agent_id: int = agent_id
        self.item_name: str = item_name
        self.old_price: float = old_price
        self.new_price: float = new_price


class TweetLog(Log):
    def __init__(self, time: int, agent_id: int, message: str) -> None:
        """Initialization.

        Args:
            time (int): The time step when the tweet is posted.
            agent_id (int): The unique id of the agent posting the tweet.
            message (str): The content of the tweet.
        """
        self.type: str = "tweet"
        self.time: int = time
        self.agent_id: int = agent_id
        self.message: str = message


class FollowLog(Log):
    def __init__(self, time: int, agent_id: int, target_agent_id: int) -> None:
        """Initialization.

        Args:
            time (int): The time step when the follow action occurs.
            agent_id (int): The unique id of the agent following.
            target_agent_id (int): The unique id of the agent being followed.
        """
        self.type: str = "follow"
        self.time: int = time
        self.agent_id: int = agent_id
        self.target_agent_id: int = target_agent_id


class UnfollowLog(Log):
    def __init__(self, time: int, agent_id: int, target_agent_id: int) -> None:
        """Initialization.

        Args:
            time (int): The time step when the unfollow action occurs.
            agent_id (int): The unique id of the agent unfollowing.
            target_agent_id (int): The unique id of the agent being unfollowed.
        """
        self.type: str = "unfollow"
        self.time: int = time
        self.agent_id: int = agent_id
        self.target_agent_id: int = target_agent_id


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

    def _process_proposal_reaction_log(self, log: ProposalReactionLog) -> None:
        pass

    def _process_change_price_log(self, log: ChangePriceLog) -> None:
        pass

    def _process_tweet_log(self, log: TweetLog) -> None:
        pass

    def _process_follow_log(self, log: FollowLog) -> None:
        pass

    def _process_unfollow_log(self, log: UnfollowLog) -> None:
        pass

    def save(self) -> None:
        pass
