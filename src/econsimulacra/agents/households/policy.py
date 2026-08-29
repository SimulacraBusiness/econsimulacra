from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any, Awaitable, Optional, Protocol

from .social import SocialDecision, _SocialMediaPolicyRules
from .states import MODE, DecisionContext, HouseholdState
from .stylized_models import (
    MobilityModel,
    PhysiologyModel,
    ProposalReactionModel,
    ShoppingModel,
)
from .tweet_renderer import TweetRenderer


@dataclass(frozen=True)
class ActionCapabilities:
    r"""Represent the action keys enabled for an agent.

    EconSimulacra configurations name unavailable actions in
    ``disabledActions``. Let :math:`\mathcal{A}` be the universe of action keys
    and :math:`\mathcal{D}` the configured disabled set. The feasible set is

    .. math::

       \mathcal{A}^{\mathrm{enabled}}=\mathcal{A}\setminus\mathcal{D}.

    Core policies query this object before mutating state. :meth:`filter` is a
    second boundary check for reactions and third-party supplemental policies.
    Unknown names are retained in :math:`\mathcal{D}` and filtered literally,
    which permits future EconSimulacra action types without changing this class.

    Args:
        disabled: Immutable set of exact disabled action-dictionary keys.
    """

    disabled: frozenset[str] = frozenset()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ActionCapabilities:
        """Construct capabilities from an agent configuration.

        Args:
            config: Agent configuration that may contain ``disabledActions``.

        Returns:
            Immutable action capabilities.

        Raises:
            TypeError: If ``disabledActions`` is a string rather than a sequence.
        """
        disabled = config.get("disabledActions") or ()
        if isinstance(disabled, str):
            raise TypeError("disabledActions must be a sequence, not a string.")
        return cls(frozenset(str(action) for action in disabled))

    def is_enabled(self, action_key: str) -> bool:
        """Return whether an action key may be emitted.

        Args:
            action_key: Exact EconSimulacra action-dictionary key.

        Returns:
            ``True`` when the key is not disabled.
        """
        return action_key not in self.disabled

    def filter(self, action_fragment: dict[str, Any]) -> dict[str, Any]:
        r"""Remove disabled keys from an action fragment.

        Args:
            action_fragment: Partial action mapping to restrict.

        Returns:
            New mapping containing only enabled keys.

        For fragment :math:`a`, the returned mapping is the restriction
        :math:`a|_{\mathcal{A}^{\mathrm{enabled}}}`. The input is not mutated.
        """
        return {
            key: value for key, value in action_fragment.items() if self.is_enabled(key)
        }


@dataclass(frozen=True)
class DecisionSignals:
    """Precomputed feasibility and demand signals for one decision.

    Args:
        can_move: Whether ``move`` is enabled.
        can_sleep: Whether ``sleep_duration`` is enabled.
        can_consume: Whether ``consumptions`` is enabled.
        can_order: Whether ``orders`` is enabled.
        should_sleep: Whether enabled sleep demand has reached onset.
        can_eat: Whether consumption is enabled and food is available.
        should_eat: Whether enabled scheduled-meal demand is active.
    """

    can_move: bool
    can_sleep: bool
    can_consume: bool
    can_order: bool
    should_sleep: bool
    can_eat: bool
    should_eat: bool


class HouseholdDecisionPolicy:
    """Apply physiological and activity priorities to produce one core action.

    This class is the sole arbiter among sleep, food, shopping, and mobility
    demands.  The priority ordering is an explicit design assumption; it is not
    claimed as an empirical utility maximization result.

    Args:
        physiology: Sleep, hunger, and meal policy.
        mobility: Movement action adapter.
        shopping: Replenishment, store-choice, and order policy.
        capabilities: Enabled-action policy. All actions are enabled by default.
    """

    def __init__(
        self,
        physiology: PhysiologyModel,
        mobility: MobilityModel,
        shopping: ShoppingModel,
        capabilities: Optional[ActionCapabilities] = None,
    ) -> None:
        """Wire the policy components used by the arbiter.

        Args:
            physiology: Sleep, hunger, and meal policy.
            mobility: Movement action adapter.
            shopping: Replenishment, store-choice, and order policy.
            capabilities: Enabled-action policy, or all-enabled by default.
        """
        self.physiology = physiology
        self.mobility = mobility
        self.shopping = shopping
        self.capabilities = capabilities or ActionCapabilities()

    def decide(
        self,
        context: DecisionContext,
        state: HouseholdState,
    ) -> dict[str, Any]:
        r"""Return the highest-priority feasible household action.

        The policy evaluates guards in the following strict lexicographic
        order and executes only the first applicable core branch:

        1. Environment reports that the household is already sleeping: emit no
           core action.
        2. Sleep and movement are enabled and sleep is due away from home: move
           home.
        3. Consumption is enabled, hunger is critical, and food exists: eat at
           the current position.
        4. Sleep is enabled and due at home: sleep.
        5. Movement is enabled and a return-home trip is active and incomplete:
           continue it.
        6. Consumption is enabled, a meal is due, and food exists: eat at the
           current position.
        7. Orders are enabled, shopping is due, budget is positive, and no
           destination is active: choose a store. If movement is disabled, only
           co-located stores belong to the choice set.
        8. A store trip is active: order on arrival and depart for home in the
           same action when movement is enabled, move toward the store when
           necessary, or cancel an infeasible trip.
        9. Movement is enabled and the household remains away from home: move
            home.
        10. Otherwise: normalize the current home/away mode and emit no core
            action.

        Formally, let :math:`g_k(o_t,x_t)\in\{0,1\}` denote applicability of
        branch :math:`k` in that list and let :math:`a_k(o_t,x_t)` be its state
        transition and action fragment.  The selected branch is

        .. math::

           k_t^*=\min\{k\in\{1,\ldots,10\}:g_k(o_t,x_t)=1\},\qquad
           a_t^{\mathrm{core}}=a_{k_t^*}(o_t,x_t).

        Branch 10 is unconditional, so the set is nonempty.  Sleep demand,
        meal demand, food feasibility, shopping demand, and budget are defined
        by :meth:`PhysiologyModel.sleep_due`,
        :meth:`PhysiologyModel.meal_due`, :meth:`PhysiologyModel.can_eat`,
        :meth:`ShoppingPolicy.shopping_due`, and :meth:`ShoppingPolicy.budget`.
        Critical hunger means
        :math:`G_t\geq G_{\mathrm{critical}}`.  Home equality is exact tuple
        equality :math:`p_t=h`. If :math:`e_a=\mathbf{1}\{a\notin\mathcal{D}\}`
        is the capability indicator for action key :math:`a`, every branch that
        emits :math:`a` also includes :math:`e_a=1` in its guard. Thus a disabled
        high-priority demand cannot block a lower-priority feasible branch.

        Args:
            context: Normalized current observation.
            state: Mutable household state.

        Returns:
            Exactly one core action fragment, possibly empty.

        Before evaluating the priority branches, any co-located seller offers
        update this household's store-specific price and availability beliefs.
        """
        self.shopping.update_beliefs(context)
        signals = self.get_signals(context, state)
        rules = (
            self._env_sleep_action,
            self._critical_meal_action,
            self._sleep_action,
            self._return_home_and_sleep_action,
            self._continue_return_action,
            self._scheduled_meal_action,
        )
        for rule in rules:
            action = rule(context, state, signals)
            if action is not None:
                return action
        self._select_store(context, state, signals)
        store_action = self._store_trip_action(context, state, signals)
        if store_action is not None:
            return store_action
        away_action = self._away_action(context, state, signals)
        if away_action is not None:
            return away_action
        return self._idle_action(context, state, signals)

    def get_signals(
        self,
        context: DecisionContext,
        state: HouseholdState,
    ) -> DecisionSignals:
        """Calculate action feasibility and physiological demand once.

        Args:

            context: Normalized current observation.
            state: Current household state.

        Returns:
            Immutable signals shared by all priority rules this step.
        """
        can_sleep = self.capabilities.is_enabled("sleep_duration")
        can_consume = self.capabilities.is_enabled("consumptions")
        return DecisionSignals(
            can_move=self.capabilities.is_enabled("move"),
            can_sleep=can_sleep,
            can_consume=can_consume,
            can_order=self.capabilities.is_enabled("orders"),
            should_sleep=can_sleep
            and self.physiology.should_sleep(state, context.hour),
            can_eat=can_consume and self.physiology.can_eat(context.inventory),
            should_eat=can_consume and self.physiology.should_eat(state, context.hour),
        )

    def _sleep_action(
        self,
        context: DecisionContext,
        state: HouseholdState,
        signals: DecisionSignals,
    ) -> Optional[dict[str, Any]]:
        """Start sleep when due at home.

        Args:
            context: Normalized current observation.
            state: Mutable household state.
            signals: Precomputed decision signals.

        Returns:
            Sleep action when applicable, otherwise ``None``.
        """
        if signals.should_sleep and context.current_pos == state.home:
            return self.physiology.generate_sleep_action(state, context.hour)
        return None

    def _env_sleep_action(
        self,
        context: DecisionContext,
        state: HouseholdState,
        signals: DecisionSignals,
    ) -> Optional[dict[str, Any]]:
        """Honor an already-active environment sleep interval.

        Args:
            context: Normalized current observation.
            state: Mutable household state.
            signals: Precomputed decision signals; unused by this rule.

        Returns:
            Empty selected action while sleeping, otherwise ``None``.
        """
        del signals
        if not context.obs.get("self_is_sleeping", False):
            return None
        state.mode = "SLEEPING"
        state.has_been_sleeping = True
        return {}

    def _return_home_and_sleep_action(
        self,
        context: DecisionContext,
        state: HouseholdState,
        signals: DecisionSignals,
    ) -> Optional[dict[str, Any]]:
        """Return home when sleep is due away from home.

        Args:
            context: Normalized current observation.
            state: Mutable household state.
            signals: Precomputed decision signals.

        Returns:
            Homeward move when applicable, otherwise ``None``.
        """
        if (
            signals.should_sleep
            and context.current_pos != state.home
            and signals.can_move
        ):
            return self._return_home(state, "RETURN_HOME_SLEEP")
        return None

    def _return_home(
        self,
        state: HouseholdState,
        mode: MODE,
    ) -> dict[str, Any]:
        """Construct a homeward movement action.

        Args:
            state: Mutable household state with initialized home position.
            mode: Return-home activity label to persist.

        Returns:
            Movement action targeting home.

        Raises:
            ValueError: If home has not been initialized.
        """
        if state.home is None:
            raise ValueError("Household home is not initialized.")
        return self.mobility.generate_move_action(state, state.home, mode)

    def _continue_return_action(
        self,
        context: DecisionContext,
        state: HouseholdState,
        signals: DecisionSignals,
    ) -> Optional[dict[str, Any]]:
        """Continue an active homeward trip.

        Args:
            context: Normalized current observation.
            state: Mutable household state.
            signals: Precomputed decision signals.

        Returns:
            Homeward move when applicable, otherwise ``None``.
        """
        if (
            state.mode.startswith("RETURN_HOME")
            and context.current_pos != state.home
            and signals.can_move
        ):
            return self._return_home(state, state.mode)
        return None

    def _scheduled_meal_action(
        self,
        context: DecisionContext,
        state: HouseholdState,
        signals: DecisionSignals,
    ) -> Optional[dict[str, Any]]:
        """Eat a scheduled meal at the current position.

        Args:
            context: Normalized current observation.
            state: Mutable household state.
            signals: Precomputed decision signals.

        Returns:
            Consumption action when applicable, otherwise ``None``.
        """
        if signals.should_eat and signals.can_eat:
            return self.physiology.generate_eat_action(state, context.inventory)
        return None

    def _critical_meal_action(
        self,
        context: DecisionContext,
        state: HouseholdState,
        signals: DecisionSignals,
    ) -> Optional[dict[str, Any]]:
        """Resolve critical hunger by eating at the current position.

        Args:
            context: Normalized current observation.
            state: Mutable household state.
            signals: Precomputed decision signals.

        Returns:
            Consumption when feasible, otherwise ``None``.
        """
        if state.hunger < self.physiology.critical_hunger or not signals.can_eat:
            return None
        return self.physiology.generate_eat_action(state, context.inventory)

    def _select_store(
        self,
        context: DecisionContext,
        state: HouseholdState,
        signals: DecisionSignals,
    ) -> None:
        """Select and persist a shopping destination when replenishment is due.

        Args:
            context: Normalized current observation.
            state: Mutable household state.
            signals: Precomputed decision signals.
        """
        if (
            not signals.can_order
            or not self.shopping.should_shop(context.inventory)
            or self.shopping.get_budget(context.inventory) <= 1e-9
            or state.destination is not None
        ):
            return
        stores = self.shopping.get_stores(context)
        if not signals.can_move:
            stores = [
                store for store in stores if tuple(store["pos"]) == context.current_pos
            ]
        store = self.shopping.choose_store(
            context.current_pos,
            stores,
            context.inventory,
        )
        if store is not None:
            state.destination = tuple(store["pos"])
            state.mode = "TRAVEL_STORE"

    def _store_trip_action(
        self,
        context: DecisionContext,
        state: HouseholdState,
        signals: DecisionSignals,
    ) -> Optional[dict[str, Any]]:
        """Continue, complete, or cancel an active store trip.

        Args:
            context: Normalized current observation.
            state: Mutable household state.
            signals: Precomputed decision signals.

        Returns:
            Movement, order, or combined order-and-movement action when emitted.
        """
        if state.mode != "TRAVEL_STORE" or state.destination is None:
            return None
        if not signals.can_order:
            state.mode = "RETURN_HOME" if signals.can_move else "AWAY"
            state.destination = state.home if signals.can_move else None
            return None
        if context.current_pos != state.destination:
            if signals.can_move:
                return self.mobility.generate_move_action(
                    state, state.destination, "TRAVEL_STORE"
                )
            state.mode = "HOME" if context.current_pos == state.home else "AWAY"
            state.destination = None
            return None
        action = self.shopping.generate_order_action(context)
        if action:
            state.has_been_sleeping = False
            if signals.can_move and context.current_pos != state.home:
                movement = self._return_home(state, "RETURN_HOME")
                return action | movement
            state.mode = "HOME" if context.current_pos == state.home else "AWAY"
            state.destination = None
            return action
        state.mode = "RETURN_HOME"
        state.destination = state.home
        return None

    def _away_action(
        self,
        context: DecisionContext,
        state: HouseholdState,
        signals: DecisionSignals,
    ) -> Optional[dict[str, Any]]:
        """Return an otherwise idle household home when movement is enabled.

        Args:
            context: Normalized current observation.
            state: Mutable household state.
            signals: Precomputed decision signals.

        Returns:
            Homeward move when away and mobile, otherwise ``None``.
        """
        if context.current_pos != state.home and signals.can_move:
            return self._return_home(state, "RETURN_HOME")
        return None

    def _idle_action(
        self,
        context: DecisionContext,
        state: HouseholdState,
        signals: DecisionSignals,
    ) -> dict[str, Any]:
        """Normalize idle state and emit no core action.

        Args:
            context: Normalized current observation.
            state: Mutable household state.
            signals: Precomputed signals; unused by this rule.

        Returns:
            Empty action mapping.
        """
        del signals
        state.mode = "HOME" if context.current_pos == state.home else "AWAY"
        state.destination = None
        state.has_been_sleeping = False
        return {}


class SupplementalPolicy(Protocol):
    """Protocol for optional actions such as social-network behavior.

    A supplemental policy observes the same context and mutable state as the
    core policy and returns an action fragment. It is evaluated after the core
    household decision, then combined by :class:RuleBasedHousehold._compose_fragments.
    The default list contains :class:`ProposalReactionPolicy`;
    additional policies can add independent behavior such as SNS actions.

    Implementations do not need to inherit this Protocol; providing a
    compatible synchronous or asynchronous :meth:`decide` method is sufficient.

    .. rubric:: Usage

    .. code-block:: python

       class HungerTweetPolicy:
           def decide(self, context, state):
               if state.hunger >= 0.8:
                   return {"tweet": "I am getting hungry."}
               return {}


       class SocialHousehold(RuleBasedHousehold):
           def __init__(self, *args, **kwargs):
               super().__init__(*args, **kwargs)
               self.add_supplemental_policy(HungerTweetPolicy())


       simulator.register_classes([SocialHousehold])

    The corresponding agent configuration uses
    ``"type": "SocialHousehold"``. Disabled action keys returned by a custom
    policy are removed before composition. Custom policies should normally read
    rather than mutate ``state`` so that core behavior remains independent.
    """

    def decide(
        self,
        context: DecisionContext,
        state: HouseholdState,
    ) -> dict[str, Any] | Awaitable[dict[str, Any]]:
        """Return an action fragment to merge with the core decision.

        Args:
            context: Normalized current observation and time.
            state: Mutable household state after the core policy has run.

        Returns:
            Partial EconSimulacra action mapping. Return an empty mapping when
            the supplemental policy has no action for this step.
        """
        raise NotImplementedError


class SocialMediaPolicy(_SocialMediaPolicyRules, SupplementalPolicy):
    """Generate a complete supplemental SNS action fragment.

    Args:
        config: ``socialRule`` household configuration.
        prng: Shared seeded pseudo-random generator.
        capabilities: Household action capabilities used to avoid infeasible
            decisions and unnecessary Tiny LM calls.
        tweet_renderer: Optional asynchronous renderer. It is required only
            when tweet actions are enabled.

    Follow, unfollow, tweet-event occurrence, topic, sentiment, and style are
    selected by rules. The renderer controls wording only. This policy is
    asynchronous solely because text generation may take time; existing
    synchronous supplemental policies remain valid.
    """

    def __init__(
        self,
        config: dict[str, Any],
        prng: Random,
        capabilities: Optional[ActionCapabilities] = None,
        tweet_renderer: Optional[TweetRenderer] = None,
    ) -> None:
        """Initialize SNS rules and optional text realization.

        Args:
            config: ``socialRule`` household configuration.
            prng: Shared seeded pseudo-random generator.
            capabilities: Enabled household actions.
            tweet_renderer: Renderer used only after a Hawkes tweet event.
        """
        super().__init__(config=config, prng=prng)
        self.capabilities = capabilities or ActionCapabilities()
        self.tweet_renderer = tweet_renderer

    async def decide(
        self,
        context: DecisionContext,
        state: HouseholdState,
    ) -> dict[str, Any]:
        """Generate one supplemental social-network action fragment.

        Args:
            context: Normalized current observation.
            state: Household state after the core policy decision.

        Returns:
            Mapping containing zero or more of ``follow``, ``unfollow``, and
            ``tweet``.
        """
        decision: SocialDecision = self.generate_social_decision(
            context=context,
            household_state=state,
            capabilities=self.capabilities,
        )
        fragment: dict[str, Any] = {}
        if decision.follow_agent_id is not None:
            fragment["follow"] = decision.follow_agent_id
        if decision.unfollow_agent_id is not None:
            fragment["unfollow"] = decision.unfollow_agent_id
        if decision.tweet_intent is not None and self.tweet_renderer is not None:
            tweet = await self.tweet_renderer.generate_tweet(
                intent=decision.tweet_intent,
                previous_tweet=context.obs.get("self_tweet"),
            )
            if tweet is not None:
                fragment["tweet"] = tweet
                self.record_generated_tweet(decision.tweet_intent)
        return fragment


class ProposalReactionPolicy(SupplementalPolicy):
    """React to proposals from other agents.

    This policy is the default supplemental policy for all households. It
    observes the same context and mutable state as the core policy and returns
    an action fragment. It is evaluated after the core household decision, then
    combined by :class:RuleBasedHousehold._compose_fragments.
    """

    def __init__(self) -> None:
        """Initialization."""
        self.reaction_model = ProposalReactionModel()

    def decide(
        self,
        context: DecisionContext,
        state: HouseholdState,
    ) -> dict[str, Any]:
        """Return a proposal-reaction action fragment.

        Args:
            context: Normalized current observation and time.
            state: Mutable household state after the core policy has run.

        Returns:
            Partial EconSimulacra action mapping. Return an empty mapping when
            no proposal reactions are applicable.
        """
        return self.reaction_model.generate_reactions(context, state)
