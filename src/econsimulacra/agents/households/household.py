from __future__ import annotations

from random import Random
from typing import Any, Optional

from ..base import Agent
from .policy import (
    ActionCapabilities,
    HouseholdDecisionPolicy,
    ProposalReactionPolicy,
    SupplementalPolicy,
)
from .states import DecisionContext
from .stylized_models import MobilityModel, PhysiologyModel, ShoppingModel


class RuleBasedHousehold(Agent[dict[str, Any]]):
    r"""EconSimulacra adapter for the complete rule-based household.

    The final action combines the core fragment and every supplemental fragment:

    .. math::

       A_t=\Pi_E(a_t^{\rm core})
           \oplus\bigoplus_{j=1}^{m}\Pi_E(a_{t,j}^{\rm supplemental}),

    where :math:`\Pi_E` removes disabled keys and :math:`\oplus` is implemented
    by :meth:`_compose_fragments`.

    .. rubric:: Usage: add a custom action

    .. code-block:: python

       class WeatherTweetPolicy:
           def decide(self, context, state):
               if context.hour == 8:
                   return {"tweet": "Good morning."}
               return {}


       class SocialHousehold(RuleBasedHousehold):
           def __init__(self, *args, **kwargs):
               super().__init__(*args, **kwargs)
               self.add_supplemental_policy(WeatherTweetPolicy())


       simulator.register_classes([SocialHousehold])

    Set the agent config ``type`` to ``SocialHousehold``. A custom scalar key
    must not conflict with another fragment, and any key listed in
    ``disabledActions`` is removed before composition.
    """

    def __init__(
        self,
        agent_id: int,
        agent_name: str,
        env_service_dic: dict[str, Any],
        prng: Optional[Random] = None,
        config: Optional[dict[str, Any]] = None,
        decision_policy: Optional[HouseholdDecisionPolicy] = None,
        supplemental_policies: list[SupplementalPolicy] = [],
    ) -> None:
        """Initialize the EconSimulacra household adapter.

        Args:
            agent_id: Environment-assigned unique identifier.
            agent_name: Base agent name.
            env_service_dic: Environment services available to the agent.
            prng: Optional pseudo-random generator.
            config: Optional household configuration.
            decision_policy: Optional core decision policy. The default is
                constructed from ``config``.
            supplemental_policies: Optional policies appended after the default
                proposal-rejection policy.
        """
        super().__init__(
            agent_id=agent_id,
            agent_name=agent_name,
            env_service_dic=env_service_dic,
            prng=prng,
            config=config,
        )
        self.cash_name = self.config.get("cashName", "Yen")
        self.food_items = tuple(self.config.get("foodItems", ()))
        self.start_hour = float(self.config.get("startHour", 6.0))
        self.step_hours = float(self.config.get("stepHours", 1.0))
        self.decision_policy = decision_policy or self.build_decision_policy()
        self.capabilities = self.decision_policy.capabilities
        self.supplemental_policies: list[SupplementalPolicy] = []
        self.add_supplemental_policy(ProposalReactionPolicy())
        for policy in supplemental_policies:
            self.add_supplemental_policy(policy)
        self.state = self.decision_policy.physiology.initialize_state()

    def build_decision_policy(self) -> HouseholdDecisionPolicy:
        """Build the default core decision policy.

        Returns:
            Policy wired with default physiology, mobility, shopping, and
            action capabilities.
        """
        return HouseholdDecisionPolicy(
            physiology=PhysiologyModel(
                self.config,
                self.food_items,
                self.step_hours,
            ),
            mobility=MobilityModel(),
            shopping=ShoppingModel(
                self.config,
                self.food_items,
                self.cash_name,
                self.prng,
            ),
            capabilities=ActionCapabilities.from_config(self.config),
        )

    def add_supplemental_policy(self, policy: SupplementalPolicy) -> None:
        """Register one custom policy after existing supplemental policies.

        Args:
            policy: Object implementing synchronous ``decide(context, state)``.
        """
        self.supplemental_policies.append(policy)

    async def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        r"""Orchestrate one household decision.

        Args:
            obs: EconSimulacra observation mapping for the current step.

        Returns:
            Composed, capability-filtered EconSimulacra action mapping.

        .. math::

           O_t\to o_t\to x_t\to
           (a_t^{\rm core},a_t^{\rm supplemental,*})
           \to A_t.
        """
        context = self._context(obs)
        self._initialize_home(obs)
        self.decision_policy.physiology.update_state(self.state, context.time_step)
        return self._compose_action(context)

    @property
    def mode(self) -> str:
        """Return the current activity mode for diagnostics."""
        return self.state.mode

    def _initialize_home(self, obs: dict[str, Any]) -> None:
        """Initialize home exactly once from the environment observation.

        Args:
            obs: Observation containing ``self_init_pos`` on the first call.
        """
        if self.state.home is None:
            self.state.home = tuple(obs["self_init_pos"])

    def _compose_action(self, context: DecisionContext) -> dict[str, Any]:
        """Evaluate, filter, and compose all policy fragments.

        Args:
            context: Normalized current observation.

        Returns:
            One action mapping ready for EconSimulacra.
        """
        action_fragments = [
            self._enabled(self.decision_policy.decide(context, self.state)),
        ]
        action_fragments.extend(self._supplemental_fragments(context))
        return self._compose_fragments(action_fragments)

    def _compose_fragments(
        self, action_fragments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        r"""Merge action fragments without silently overwriting them.

        Args:
            fragments: Partial action mappings in deterministic evaluation order.

        Returns:
            One EconSimulacra action mapping. Sequence-valued actions are
            concatenated in fragment order.

        Raises:
            ValueError: If fragments assign unequal values to the same scalar
                action key.

        Let :math:`a^{(1)},\ldots,a^{(m)}` be the supplied fragments. Their
        composition is :math:`A=a^{(1)}\oplus\cdots\oplus a^{(m)}`. For
        sequence-valued keys, :math:`\oplus` concatenates tuples in registration
        order. For every scalar key :math:`k`, it retains the unique value:

        .. math::

           A(k)=v \quad\Longleftrightarrow\quad
           \exists r:a^{(r)}(k)=v
           \ \land\
           \ \nexists s:a^{(s)}(k)\ne v.

        Conflicting scalar values are undefined and raise ``ValueError``.
        """
        sequence_actions = {
            "consumptions",
            "orders",
            "proposals",
            "reactions",
            "set_prices",
        }
        composed_action: dict[str, Any] = {}
        for fragment in action_fragments:
            for key, value in fragment.items():
                if key in sequence_actions:
                    existing = tuple(composed_action.get(key, ()))
                    composed_action[key] = existing + tuple(value)
                elif key in composed_action and composed_action[key] != value:
                    raise ValueError(f"Conflicting action fragments for '{key}'.")
                else:
                    composed_action[key] = value
        return composed_action

    def _supplemental_fragments(self, context: DecisionContext) -> list[dict[str, Any]]:
        """Evaluate supplemental policies in registration order.

        Args:
            context: Normalized current observation.

        Returns:
            Capability-filtered supplemental action fragments.
        """
        return [
            self._enabled(policy.decide(context, self.state))
            for policy in self.supplemental_policies
        ]

    def _enabled(self, fragment: dict[str, Any]) -> dict[str, Any]:
        """Restrict one fragment to enabled action keys.

        Args:
            fragment: Partial action mapping produced by one policy.

        Returns:
            Filtered action mapping.
        """
        return self.capabilities.filter(fragment)

    def _context(self, obs: dict[str, Any]) -> DecisionContext:
        """Normalize an EconSimulacra observation.

        Args:
            obs: Raw observation mapping.

        Returns:
            Typed context with step, clock hour, position, and numeric inventory.
        """
        time_step = self.get_time_step(obs)
        return DecisionContext(
            obs=obs,
            time_step=time_step,
            hour=(self.start_hour + time_step * self.step_hours) % 24.0,
            current_pos=tuple(obs["self_pos"]),
            inventory={
                key: float(value)
                for key, value in obs.get("self_inventory", {}).items()
            },
        )

    def get_time_step(self, obs: dict[str, Any]) -> int:
        """Extract a numeric step with a monotone fallback.

        Args:
            obs: Raw observation that may contain numeric ``time``.

        Returns:
            Integer simulation step.
        """
        value = obs.get("time", 0)
        if isinstance(value, (int, float)):
            return int(value)
        return (self.state.last_step or 0) + 1
