from __future__ import annotations

import math
from random import Random
from typing import Any, Optional

from .states import MODE, DecisionContext, HouseholdState


class PhysiologyModel:
    r"""Update sleep pressure and hunger and create physiological actions.

    This is a stylized discrete-time model inspired by the exponential sleep
    Process S and sinusoidal Process C (Borbély, 1982), plus empirical circadian
    meal timing (de Castro, 1987). It is not a calibrated reproduction.

    .. rubric:: References

    * Borbély, A. A. (1982). A two process model of sleep regulation.
      *Human Neurobiology, 1*\ (3), 195-204.
      https://pubmed.ncbi.nlm.nih.gov/7185792/
    * de Castro, J. M. (1987). Circadian rhythms of the spontaneous meal
      pattern, macronutrient intake, and mood of humans. *Physiology & Behavior,
      40*\ (4), 437-446.
      https://doi.org/10.1016/0031-9384(87)90028-X
    """

    def __init__(
        self,
        config: dict[str, Any],
        food_items: tuple[str, ...],
        step_hours: float,
    ) -> None:
        """Initialization.

        Args:
            config: Household configuration containing physiological rules. Identical to the household configuration.
            It may include:
                - ``sleepRule``: sleep rule parameters.
                - ``mealRule``: meal rule parameters.
            food_items: Ordered food names used for meal allocation.
            step_hours: Simulated hours per environment step.
        """
        self.food_items = food_items
        self.step_hours = step_hours

        sleep_rules: dict[str, float] = config.get("sleepRule", {})
        self.initial_sleep_pressure = float(sleep_rules.get("initialPressure", 0.35))
        self.sleep_lower = float(sleep_rules.get("lowerAsymptote", 0.05))
        self.sleep_upper = float(sleep_rules.get("upperAsymptote", 1.0))
        self.tau_wake = float(sleep_rules.get("tauWakeHours", 18.0))
        self.tau_sleep = float(sleep_rules.get("tauSleepHours", 4.0))
        self.circadian_amplitude = float(sleep_rules.get("circadianAmplitude", 0.12))
        self.circadian_phase = float(sleep_rules.get("circadianPhaseHour", 2.0))
        self.sleep_on = float(sleep_rules.get("onsetThreshold", 0.78))
        self.sleep_off = float(sleep_rules.get("wakeThreshold", 0.28))
        self.max_sleep_steps = int(sleep_rules.get("maxSleepSteps", 10))

        meal_rules: dict[str, Any] = config.get("mealRule", {})
        self.initial_hunger = float(meal_rules.get("initialHunger", 0.25))
        self.hunger_rate_awake = float(meal_rules.get("awakeHungerRate", 0.07))
        self.hunger_rate_sleep = float(meal_rules.get("sleepHungerRate", 0.025))
        self.meal_centers = tuple(
            float(value) for value in meal_rules.get("mealCenters", (8, 13, 19))
        )
        self.meal_width = float(meal_rules.get("mealSignalWidthHours", 1.25))
        self.meal_signal_weight = float(meal_rules.get("mealSignalWeight", 0.35))
        self.meal_threshold = float(meal_rules.get("mealThreshold", 0.72))
        self.critical_hunger = float(meal_rules.get("criticalHunger", 0.92))
        self.min_meal_interval = float(meal_rules.get("minMealIntervalHours", 3.0))
        default_share = 1.0 / len(food_items)
        self.composition: dict[str, float] = {
            item: float(meal_rules.get("composition", {}).get(item, default_share))
            for item in food_items
        }
        self.energy_per_unit: dict[str, float] = {
            item: float(meal_rules.get("energyPerUnit", {}).get(item, 1.0))
            for item in food_items
        }

    def initialize_state(self) -> HouseholdState:
        r"""Create the initial household state.

        Returns:
            State with configured initial sleep pressure and hunger, and an
            infinite elapsed meal interval.

        .. math::

           H_0=H_{\mathrm{init}},\quad G_0=G_{\mathrm{init}},\quad M_0=\infty.
        """
        return HouseholdState(
            sleep_pressure=self.initial_sleep_pressure,
            hunger=self.initial_hunger,
            last_meal_elapsed=math.inf,
        )

    def update_state(self, state: HouseholdState, step: int) -> None:
        r"""Advance physiological stocks to ``step``.

        Args:
            state: Mutable household state to update in place.
            step: Current nonnegative simulation step.

        For elapsed hours :math:`\delta_t`, lower and upper asymptotes
        :math:`L<U`, and time constants :math:`\theta_s,\theta_w>0`,

        .. math::

           H_t=\begin{cases}
             L+(H_{t-1}-L)e^{-\delta_t/\theta_s},&\sigma_{t-1}=1,\\
             U-(U-H_{t-1})e^{-\delta_t/\theta_w},&\sigma_{t-1}=0.
           \end{cases}

        Hunger evolves as

        .. math::

           G_t=\min\{1,G_{t-1}+\rho_t\delta_t\},\qquad
           M_t=M_{t-1}+\delta_t,

        using the configured awake or asleep rate :math:`\rho_t`.
        """
        if state.last_step is None:
            state.last_step = step
            return
        elapsed = max(0, step - state.last_step) * self.step_hours
        if state.has_been_sleeping:
            state.sleep_pressure = self.sleep_lower + (
                state.sleep_pressure - self.sleep_lower
            ) * math.exp(-elapsed / self.tau_sleep)
            hunger_rate = self.hunger_rate_sleep
        else:
            state.sleep_pressure = self.sleep_upper - (
                self.sleep_upper - state.sleep_pressure
            ) * math.exp(-elapsed / self.tau_wake)
            hunger_rate = self.hunger_rate_awake
        state.hunger = min(1.0, state.hunger + hunger_rate * elapsed)
        state.last_meal_elapsed += elapsed
        state.last_step = step

    def should_sleep(self, state: HouseholdState, current_hour: float) -> bool:
        r"""Test the sleep-onset condition.

        Args:
            state: Household state containing current sleep pressure.
            current_hour: Current local clock hour in ``[0, 24)``.

        Returns:
            ``True`` when homeostatic plus circadian pressure reaches onset.

        .. math::

           C(\tau)=A\cos\left(\frac{2\pi(\tau-\phi)}{24}\right),\qquad
           D_t^{\mathrm{sleep}}=\mathbf{1}\{H_t+C(\tau_t)\geq\eta_{\rm on}\}.
        """
        return state.sleep_pressure + self._circadian(current_hour) >= self.sleep_on

    def generate_sleep_action(
        self, state: HouseholdState, current_hour: float
    ) -> dict[str, Any]:
        r"""Enter sleep mode and construct a sleep action.

        Args:
            state: Mutable household state to mark as sleeping.
            current_hour: Current local clock hour used to predict wake time.

        Returns:
            Action containing the predicted integer ``sleep_duration``.

        For maximum duration :math:`N`, the chosen duration is the first
        :math:`n\in\{1,\ldots,N\}` at which projected pressure reaches the wake
        threshold, or :math:`N` if no such step exists.
        """
        state.mode = "SLEEPING"
        state.destination = None
        state.has_been_sleeping = True
        return {"sleep_duration": self._sleep_duration(state, current_hour)}

    def should_eat(self, state: HouseholdState, current_hour: float) -> bool:
        r"""Test the scheduled-meal condition.

        Args:
            state: Household state containing hunger and elapsed meal time.
            current_hour: Current local clock hour in ``[0, 24)``.

        Returns:
            ``True`` when minimum spacing and weighted meal demand both hold.

        .. math::

           S(\tau)=\max_{\mu\in\mathcal K}
           \exp\left[-\frac{d_{24}(\tau,\mu)^2}{2w^2}\right],

        .. math::

           D_t^{\mathrm{meal}}=\mathbf{1}\{M_t\geq m_{\min}\ \land\
           G_t+\omega S(\tau_t)\geq\eta_m\}.
        """
        signal = max(
            math.exp(
                -(self._clock_distance(current_hour, center) ** 2)
                / (2 * self.meal_width**2)
            )
            for center in self.meal_centers
        )
        return (
            state.last_meal_elapsed >= self.min_meal_interval
            and state.hunger + self.meal_signal_weight * signal >= self.meal_threshold
        )

    def can_eat(self, inventory: dict[str, float]) -> bool:
        """Test whether at least one configured food is available.

        Args:
            inventory: Current self-inventory keyed by item name.

        Returns:
            ``True`` when any configured food has a positive amount.
        """
        return any(inventory.get(item, 0.0) > 0 for item in self.food_items)

    def generate_eat_action(
        self,
        state: HouseholdState,
        inventory: dict[str, float],
    ) -> dict[str, Any]:
        r"""Allocate one meal and update hunger.

        Args:
            state: Mutable household state to update after consumption.
            inventory: Current self-inventory keyed by item name.

        Returns:
            ``consumptions`` records for positive food quantities, or an empty
            mapping when no positive quantity can be consumed.

        For food share :math:`s_i` and energy per unit :math:`e_i`,

        .. math::

           c_{i,t}=\min\left\{I_t(i),
           \frac{\max(0,G_t)s_i}{\max(e_i,10^{-12})}\right\}.
        """
        target_energy = max(0.0, state.hunger)
        consumptions: list[dict[str, Any]] = []
        consumed_energy = 0.0
        for item in self.food_items:
            desired_energy = target_energy * self.composition[item]
            amount = min(
                inventory.get(item, 0.0),
                desired_energy / max(self.energy_per_unit[item], 1e-12),
            )
            if amount <= 0:
                continue
            consumptions.append({"item_name": item, "item_amount": amount})
            consumed_energy += amount * self.energy_per_unit[item]
        if not consumptions:
            return {}
        state.hunger = max(0.0, state.hunger - consumed_energy)
        state.last_meal_elapsed = 0.0
        state.mode = "HOME"
        state.destination = None
        state.has_been_sleeping = False
        return {"consumptions": tuple(consumptions)}

    def _circadian(self, hour: float) -> float:
        """Calculate the circadian sleep contribution for one clock hour.

        Args:
            hour: Local clock hour.

        Returns:
            Sinusoidal circadian contribution.
        """
        return self.circadian_amplitude * math.cos(
            2.0 * math.pi * (hour - self.circadian_phase) / 24.0
        )

    def _sleep_duration(self, state: HouseholdState, hour: float) -> int:
        """Predict the first sleep step satisfying the wake threshold.

        Args:
            state: Household state at sleep onset.
            hour: Local clock hour at sleep onset.

        Returns:
            Integer sleep duration capped by ``max_sleep_steps``.
        """
        for steps in range(1, self.max_sleep_steps + 1):
            elapsed = steps * self.step_hours
            pressure = self.sleep_lower + (
                state.sleep_pressure - self.sleep_lower
            ) * math.exp(-elapsed / self.tau_sleep)
            future_hour = (hour + elapsed) % 24.0
            if pressure + self._circadian(future_hour) <= self.sleep_off:
                return steps
        return self.max_sleep_steps

    @staticmethod
    def _clock_distance(first: float, second: float) -> float:
        """Calculate shortest circular distance between two clock hours.

        Args:
            first: First clock hour.
            second: Second clock hour.

        Returns:
            Distance in hours in the closed interval ``[0, 12]``.
        """
        delta = abs(first - second) % 24.0
        return min(delta, 24.0 - delta)


class ShoppingModel:
    r"""Manage replenishment, store choice, budgeting, and order settlement.

    The replenishment trigger is an :math:`(s,S)`-style rule inspired by the
    classical inventory policy (Scarf, 1960). Here it is a household heuristic,
    not the solution of Scarf's dynamic program: reorder points and target
    stocks are fixed configuration values and the basket is cash constrained.

    Store choice uses a multinomial logit over observed stores, following the
    random-utility form associated with conditional logit (McFadden, 1974).
    Travel is embedded in an activity sequence (home, shopping, home), a
    minimal stylization of activity-schedule demand models (Bowman & Ben-Akiva,
    2001).

    All stochastic choice uses the injected ``random.Random`` instance.  A
    fixed seed, observation order, configuration, and initial state therefore
    reproduce the same choice sequence.

    .. rubric:: References

    * Bowman, J. L., & Ben-Akiva, M. E. (2001). Activity-based disaggregate
      travel demand model system with activity schedules. *Transportation
      Research Part A: Policy and Practice, 35*\ (1), 1-28.
      https://doi.org/10.1016/S0965-8564(99)00043-9
    * McFadden, D. (1974). Conditional logit analysis of qualitative choice
      behavior. In P. Zarembka (Ed.), *Frontiers in econometrics* (pp. 105-142).
      Academic Press.
      https://eml.berkeley.edu/reprints/mcfadden/zarembka.pdf
    * Scarf, H. E. (1960). The optimality of (S, s) policies in the dynamic
      inventory problem. In K. J. Arrow, S. Karlin, & P. Suppes (Eds.),
      *Mathematical methods in the social sciences, 1959* (pp. 196-202).
      Stanford University Press.
      https://cir.nii.ac.jp/crid/1572824499232680448
    """

    def __init__(
        self,
        config: dict[str, Any],
        food_items: tuple[str, ...],
        cash_name: str,
        prng: Random,
    ) -> None:
        """Initialization.

        Args:
            config: Household shopping configuration. Identical to the household configuration.
            It may include:
                - ``sSinventoryRule``: reorder points and target stocks.
                - ``budgetRule``: cash reserve and maximum basket share.
                - ``pricePriors``: prior prices for each food item.
                - ``storeChoice``: store choice parameters.
            food_items: Ordered configured food names.
            cash_name: Inventory key used as currency.
            prng: Pseudo-random generator used by multinomial logit sampling.
        """
        self.food_items = food_items
        self.cash_name = cash_name
        self.prng = prng

        ss_inventory_rule: dict[str, dict[str, float]] = config.get(
            "sSinventoryRule", {}
        )
        self.reorder_points: dict[str, float] = {
            item: float(ss_inventory_rule.get("reorderPoints", {}).get(item, 1.0))
            for item in food_items
        }
        self.target_stocks: dict[str, float] = {
            item: float(ss_inventory_rule.get("targetStocks", {}).get(item, 4.0))
            for item in food_items
        }
        budget_rule: dict[str, float] = config.get("budgetRule", {})
        self.cash_reserve: float = float(budget_rule.get("cashReserve", 0.0))
        self.max_basket_share: float = float(budget_rule.get("maxBasketShare", 0.35))
        self.price_priors: dict[str, float] = {
            item: float(config.get("pricePriors", {}).get(item, 1.0))
            for item in food_items
        }

        store: dict[str, Any] = config.get("storeChoice", {})
        self.beta_cost: float = float(store.get("betaCost", 0.04))
        self.beta_distance: float = float(store.get("betaDistance", 0.65))

    def should_shop(self, inventory: dict[str, float]) -> bool:
        r"""Return whether any stock is at or below its reorder point.

        Args:
            inventory: Current self-inventory keyed by item name.

        Returns:
            Whether at least one food has reached its reorder point.

        For foods :math:`\mathcal{F}` and configured reorder point :math:`s_i`,

        .. math::

           D_t^{\mathrm{shop}}=
           \mathbf{1}\{\exists i\in\mathcal{F}:I_t(i)\leq s_i\}.
        """
        return any(
            inventory.get(item, 0.0) <= self.reorder_points[item]
            for item in self.food_items
        )

    def get_budget(self, inventory: dict[str, float]) -> float:
        r"""Return cash available after reserve and basket-share constraints.

        Args:
            inventory: Current self-inventory containing the configured cash.

        Returns:
            Nonnegative maximum basket expenditure.

        Let :math:`Y_t=I_t(c)` be holdings of configured cash item :math:`c`,
        :math:`R\geq0` the cash reserve, and :math:`\alpha\geq0` the maximum
        basket share.  The shopping budget is

        .. math::

           B_t=\max\{0,\min(Y_t-R,\alpha Y_t)\}.
        """
        cash = inventory.get(self.cash_name, 0.0)
        return max(
            0.0,
            min(
                cash - self.cash_reserve,
                cash * self.max_basket_share,
            ),
        )

    def get_stores(self, context: DecisionContext) -> list[dict[str, Any]]:
        """Return visible sellers matching configured name prefixes.

        Args:
            context: Observation context containing optional ``others_pos``.

        Returns:
            Visible sellers in observation order.
            Example:

            .. code-block:: python
                [
                    {
                        "agent_id": 10,
                        "agent_name": "MarketEast",
                        "pos": (2, 3),
                    },
                    {
                        "agent_id": 12,
                        "agent_name": "MarketWest",
                        "pos": (8, 4),
                    },
                ]

        Seller order is preserved from ``others_pos`` because it is also the
        deterministic tie order used by store choice.
        """
        return [store for store in context.obs.get("others_pos", [])]

    def choose_store(
        self,
        current_pos: tuple[int, ...],
        stores: list[dict[str, Any]],
        inventory: dict[str, float],
    ) -> Optional[dict[str, Any]]:
        r"""Choose a store with a cost-distance multinomial logit.

        Args:
            current_pos: Household's current grid position.
            stores: Ordered candidate-store observation records.
            inventory: Current household inventory.

        Returns:
            Sampled store record, or ``None`` for an empty choice set.

        Let :math:`\mathcal{J}_t` be the ordered visible-store list,
        :math:`T_i` target stock, :math:`\bar p_i` the price prior, and
        :math:`p_t,d_j` the household and store positions.  The desired basket
        cost, Chebyshev grid distance, and deterministic utility are

        .. math::

           K_t=\sum_{i\in\mathcal{F}}
             \max\{0,T_i-I_t(i)\}\bar p_i,

        .. math::

           D_{tj}=\max_k|p_{t,k}-d_{j,k}|,\qquad
           V_{tj}=-\beta_K K_t-\beta_D D_{tj}.

        A store is sampled with

        .. math::

           P(J_t=j\mid o_t,x_t)=
           \frac{\exp(V_{tj})}{\sum_{r\in\mathcal{J}_t}\exp(V_{tr})}.

        Numerically, ``max(V)`` is subtracted before exponentiation.  Because
        current price priors are not store-specific, :math:`K_t` is common to
        every alternative and cancels from the probabilities; store choice is
        therefore distance-driven in the current implementation.  An empty
        choice set returns ``None``.
        """
        if not stores:
            return None
        scored: list[tuple[dict[str, Any], float]] = []
        for store in stores:
            basket_cost = sum(
                max(
                    0.0,
                    self.target_stocks[item] - inventory.get(item, 0.0),
                )
                * self.price_priors[item]
                for item in self.food_items
            )
            destination = tuple(store["pos"])
            distance = max(
                abs(current - target)
                for current, target in zip(current_pos, destination)
            )
            utility = -self.beta_cost * basket_cost - self.beta_distance * distance
            scored.append((store, utility))
        max_utility = max(utility for _, utility in scored)
        weights = [math.exp(utility - max_utility) for _, utility in scored]
        draw = self.prng.random() * sum(weights)
        cumulative = 0.0
        for (store, _), weight in zip(scored, weights):
            cumulative += weight
            if draw <= cumulative:
                return store
        return scored[-1][0]

    def generate_order_action(
        self,
        context: DecisionContext,
    ) -> dict[str, Any]:
        r"""Build a budget-feasible basket from a co-located seller.

        Args:
            context: Observation containing current inventory and co-located
                seller inventories.
        Returns:
            Order records, or an empty mapping when no order is feasible.

        The first matching seller in ``others_inventory`` is used.  Foods are
        processed in configured order :math:`i=1,\ldots,F`.  Let :math:`p_i>0`
        be observed price (zero or negative price removes the affordability
        bound), :math:`A_i` observed seller stock when numeric and
        :math:`+\infty` otherwise, and :math:`B_1=B_t`.  Then

        .. math::

           q_i^*=\min\{\max(0,T_i-I_t(i)),A_i\},

        .. math::

           q_i=\begin{cases}
             \min\{q_i^*,B_i/p_i\},&p_i>0,\\
             q_i^*,&p_i\leq0,
           \end{cases}
           \qquad B_{i+1}=B_i-p_iq_i.

        Positive :math:`q_i` values become EconSimulacra orders with TTL 2.
        EconSimulacra processes ``orders`` before ``move`` in the same action
        dictionary and exposes resulting inventory in the next observation;
        therefore no settlement-wait state is retained. An empty seller set or
        basket produces no action.
        """
        sellers = [seller for seller in context.obs.get("others_inventory", ())]
        if not sellers:
            return {}
        seller = sellers[0]
        remaining_budget = self.get_budget(context.inventory)
        orders: list[dict[str, Any]] = []
        for item in self.food_items:
            offer = seller.get(item)
            if not isinstance(offer, dict):
                continue
            price = float(offer["price"])
            desired = max(
                0.0,
                self.target_stocks[item] - context.inventory.get(item, 0.0),
            )
            available = offer.get("amount")
            if isinstance(available, (int, float)):
                desired = min(desired, float(available))
            affordable = remaining_budget / price if price > 0 else desired
            amount = min(desired, affordable)
            if amount <= 0:
                continue
            orders.append(
                {
                    "counterparty_id": int(seller["agent_id"]),
                    "item_name": item,
                    "item_amount": amount,
                    "ttl": 2,
                }
            )
            remaining_budget -= amount * price
        if not orders:
            return {}
        return {"orders": tuple(orders)}


class MobilityModel:
    """Translate an activity destination into EconSimulacra movement.

    Destination choice belongs to the activity policy; this component only
    adapts a chosen position to the environment action schema.  Reissuing the
    destination also makes movement intent explicit if environment-side motion
    persistence changes.
    """

    def generate_move_action(
        self,
        state: HouseholdState,
        destination: tuple[int, ...],
        mode: MODE,
    ) -> dict[str, Any]:
        r"""Persist intent and reissue the destination for this step.

        Args:
            state: Mutable household state in which to persist movement intent.
            destination: Target grid position.
            mode: Activity-mode label associated with the trip.

        Returns:
            ``move`` action targeting ``destination``.

        For selected destination :math:`d` and activity label :math:`z`, the
        state transition and action are

        .. math::

           (d_t,z_t,\sigma_t)\leftarrow(d,z,0),\qquad
           a_t=\{\mathtt{move}:d\}.
        """
        state.destination = destination
        state.mode = mode
        state.has_been_sleeping = False
        return {"move": destination}


class ProposalReactionModel:
    """Default model for unsolicited proposals.

    This is a boundary policy rather than a behavioral model: households do
    not initiate swaps and deterministically reject every observed proposal.
    """

    def generate_reactions(
        self,
        context: DecisionContext,
        state: HouseholdState,
    ) -> dict[str, Any]:
        """Reject every incoming swap proposal.

        Args:
            context: Observation containing optional ``incoming_proposals``.
            state: Household state; unused because rejection is stateless.

        Returns:
            Proposal-rejection reactions, or an empty mapping.
        """
        del state
        reactions = tuple(
            {
                "kind": "proposal",
                "id": proposal["proposal_id"],
                "accept": False,
            }
            for proposal in context.obs.get("incoming_proposals", ())
        )
        return {"reactions": reactions} if reactions else {}
