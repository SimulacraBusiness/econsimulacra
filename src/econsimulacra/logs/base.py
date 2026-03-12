from typing import Callable
from typing import Optional


class Log:
    def read_and_write(self, logger: "Logger") -> None:
        logger.write_log(self)

    def to_dict(self) -> dict[str, object]:
        return self.__dict__


class AgentGenerationLog(Log):
    def __init__(
        self,
        time: int | str,
        time_step: int,
        agent_id: int,
        agent_type: str,
        agent_name: str,
        inventory_dic: dict[str, float | int],
    ) -> None:
        self.type: str = "agent_generation"
        self.time: int | str = time
        self.time_step: int = time_step
        self.agent_id: int = agent_id
        self.agent_type: str = agent_type
        self.agent_name: str = agent_name
        self.inventory_dic: dict[str, float | int] = inventory_dic.copy()

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "type": self.type,
            "time": self.time,
            "time_step": self.time_step,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "agent_name": self.agent_name,
        }
        for item_name, item_amount in self.inventory_dic.items():
            d[f"inventory_{item_name}"] = item_amount
        return d


class SpaceAssignLog(Log):
    def __init__(self, agent_id: int, pos: tuple[int, ...]) -> None:
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
    ) -> None:
        self.type: str = "move"
        self.time: int | str = time
        self.time_step: int = time_step
        self.agent_id: int = agent_id
        self.old_pos: tuple[int, ...] = old_pos
        self.new_pos: tuple[int, ...] = new_pos


class ConsumptionLog(Log):
    def __init__(
        self,
        time: int | str,
        time_step: int,
        agent_id: int,
        item_name: str,
        item_amount: float | int,
    ) -> None:
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
        order_id: int,
    ) -> None:
        self.type: str = "order"
        self.time: int | str = time
        self.time_step: int = time_step
        self.agent_id: int = agent_id
        self.counterparty_id: int = counterparty_id
        self.item_name: str = item_name
        self.item_amount: float | int = item_amount
        self.price: Optional[float] = price
        self.order_id: int = order_id


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
        order_id: int,
        accept_amount: float | int,
    ) -> None:
        self.type: str = "order_reaction"
        self.time: int | str = time
        self.time_step: int = time_step
        self.agent_id: int = agent_id
        self.counterparty_id: int = counterparty_id
        self.item_name: str = item_name
        self.item_amount: float | int = item_amount
        self.price: float = price
        self.order_id: int = order_id
        self.accept_amount: float | int = accept_amount


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
        self.type: str = "tweet"
        self.time: int | str = time
        self.time_step: int = time_step
        self.agent_id: int = agent_id
        self.message: str = message
        self.num_follows: int = num_follows
        self.num_followers: int = num_followers


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
        self.type: str = "unfollow"
        self.time: int | str = time
        self.time_step: int = time_step
        self.agent_id: int = agent_id
        self.target_agent_id: int = target_agent_id
        self.num_follows: int = num_follows
        self.num_followers: int = num_followers


class Logger:
    def __init__(self) -> None:
        self.pending_logs: list[Log] = []
        self._dispatch_table: dict[type[Log], Callable] = {}

    def clear(self) -> None:
        self.pending_logs.clear()

    def write_log(self, log: Log) -> None:
        self.pending_logs.append(log)

    def process_logs(self) -> None:
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
