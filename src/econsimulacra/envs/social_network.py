class SocialNetwork:
    def __init__(self) -> None:
        self.nodes: set[int] = set()
        self.agent_id2tweet: dict[int, str] = {}
        self.agent_id2followers: dict[int, set[int]] = {}
        self.agent_id2follows: dict[int, set[int]] = {}

    def add_agent(self, agent_id: int) -> None:
        if agent_id in self.nodes:
            raise ValueError(f"Agent ID {agent_id} already exists in the social network.")
        self.nodes.add(agent_id)
        self.agent_id2followers[agent_id] = set()
        self.agent_id2follows[agent_id] = set()

    def tweet(self, agent_id: int, message: str) -> None:
        if agent_id not in self.nodes:
            raise ValueError(f"Agent ID {agent_id} not found in the social network.")
        self.agent_id2tweet[agent_id] = message

    def follow_agent(self, agent_id: int, target_agent_id: int) -> None:
        if agent_id not in self.nodes:
            raise ValueError(f"Agent ID {agent_id} not found in the social network.")
        if target_agent_id not in self.nodes:
            raise ValueError(f"Target Agent ID {target_agent_id} not found in the social network.")
        if target_agent_id == agent_id:
            raise ValueError("An agent cannot follow itself.")
        if target_agent_id in self.agent_id2follows[agent_id]:
            raise ValueError(f"Agent ID {agent_id} already follows Agent ID {target_agent_id}.")
        if agent_id in self.agent_id2followers[target_agent_id]:
            raise ValueError(f"Agent ID {target_agent_id} already has Agent ID {agent_id} as a follower.")
        self.agent_id2follows[agent_id].add(target_agent_id)
        self.agent_id2followers[target_agent_id].add(agent_id)

    def unfollow_agent(self, agent_id: int, target_agent_id: int) -> None:
        if agent_id not in self.nodes:
            raise ValueError(f"Agent ID {agent_id} not found in the social network.")
        if target_agent_id not in self.nodes:
            raise ValueError(f"Target Agent ID {target_agent_id} not found in the social network.")
        if target_agent_id not in self.agent_id2follows[agent_id]:
            raise ValueError(f"Agent ID {agent_id} does not follow Agent ID {target_agent_id}.")
        if agent_id not in self.agent_id2followers[target_agent_id]:
            raise ValueError(f"Agent ID {target_agent_id} does not have Agent ID {agent_id} as a follower.")
        self.agent_id2follows[agent_id].remove(target_agent_id)
        self.agent_id2followers[target_agent_id].remove(agent_id)

    def get_followers(self, agent_id: int) -> set[int]:
        if agent_id not in self.nodes:
            raise ValueError(f"Agent ID {agent_id} not found in the social network.")
        return self.agent_id2followers[agent_id]

    def get_follows(self, agent_id: int) -> set[int]:
        if agent_id not in self.nodes:
            raise ValueError(f"Agent ID {agent_id} not found in the social network.")
        return self.agent_id2follows[agent_id]
