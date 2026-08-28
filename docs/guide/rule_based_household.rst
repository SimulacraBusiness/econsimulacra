Rule-Based Household
====================

The :class:`~econsimulacra.agents.households.RuleBasedHousehold` is a
deterministic-policy alternative to an LLM household. It combines stylized
models of sleep, hunger, meals, inventory replenishment, store choice, and
spatial movement. Random store choice uses the household's injected
``random.Random`` instance, so a fixed seed and fixed observation order are
reproducible.

This model is intended as a transparent simulation heuristic. Its parameters
are not estimated automatically and its :math:`(s,S)` rule is not the solution
of a household dynamic program.

Configuration overview
----------------------

A typical configuration has the following structure:

.. code-block:: text

   "Household": {
       "type": "RuleBasedHousehold",
       "isHousehold": true,
       "foodItems": ["Rice", "Chocolate", "Sushi"],
       "cashName": "Yen",
       "startHour": 0.0,
       "stepHours": 1.0,
       "sleepRule": { ... },
       "mealRule": { ... },
       "sSinventoryRule": { ... },
       "budgetRule": { ... },
       "pricePriors": { ... },
       "itemImportance": { ... },
       "storeChoice": { ... }
   }

The agent should request the observations required by its policy. In
particular, ``others_pos`` supplies store locations and
``others_inventory`` supplies offers observed after co-location. Using
``"requestObs": ["all"]`` includes both built-in observations.

Decision priority
-----------------

At each step, the household updates store beliefs from co-located offers and
then applies a strict priority order. Active environment sleep is honored
first; critical hunger, sleep, return travel, scheduled meals, shopping, and
ordinary return-home behavior follow. Only the first applicable core branch
emits an action. Actions listed in ``disabledActions`` are removed from the
feasible policy.

Sleep and hunger
----------------

``sleepRule`` configures an exponential homeostatic sleep stock plus a
sinusoidal circadian contribution. ``mealRule`` configures hunger accumulation,
meal-time signals, and minimum spacing between meals. See
:class:`~econsimulacra.agents.households.PhysiologyModel` for the complete
state equations and default values.

Meal composition
----------------

``mealRule.composition`` is the fraction of current hunger energy assigned to
each food in one meal. For hunger :math:`G_t`, configured share :math:`s_i`,
energy per unit :math:`e_i`, and inventory :math:`I_t(i)`, consumption is

.. math::

   c_{i,t}=\min\left\{I_t(i),
   \frac{\max(0,G_t)s_i}{\max(e_i,10^{-12})}\right\}.

For example:

.. code-block:: json

   "mealRule": {
       "composition": {
           "Rice": 0.7,
           "Eggs": 0.2,
           "Chocolate": 0.1
       },
       "energyPerUnit": {
           "Rice": 1.0,
           "Eggs": 1.0,
           "Chocolate": 1.0
       }
   }

If ``composition`` omits an item, its default share is :math:`1/F`, where
:math:`F` is the number of configured ``foodItems``. Consequently, explicitly
assigning ``0.043478260869565216`` to each of 23 foods is simply equal sharing:

.. math::

   0.043478260869565216 = \frac{1}{23}.

The implementation deliberately remains simple:

* shares are not normalized or validated to sum to one;
* a meal may consume every configured food simultaneously;
* unavailable food is not replaced by another food; and
* unallocated hunger remains after the meal.

These properties make equal shares useful as a neutral placeholder, but they
are usually not a realistic diet model. Simulations that distinguish staples,
prepared meals, and preference goods should configure meaningful shares or
provide a custom physiology model.

Inventory replenishment
-----------------------

The shopping trigger is an :math:`(s,S)`-style rule. For reorder point
:math:`s_i`, target stock :math:`T_i`, and current inventory :math:`I_t(i)`, a
shopping trip is due when

.. math::

   \exists i:\ I_t(i)\leq s_i,

and desired replenishment is

.. math::

   q_i=\max(0,T_i-I_t(i)).

These parameters are configured per item:

.. code-block:: json

   "sSinventoryRule": {
       "reorderPoints": {"Rice": 1.0, "Chocolate": 0.5},
       "targetStocks": {"Rice": 4.0, "Chocolate": 1.0}
   }

Shopping budget
---------------

For cash :math:`Y_t`, reserve :math:`R`, and maximum basket share
:math:`\alpha`, the available shopping budget is

.. math::

   B_t=\max\{0,\min(Y_t-R,\alpha Y_t)\}.

.. code-block:: json

   "budgetRule": {
       "cashReserve": 20000.0,
       "maxBasketShare": 0.1
   }

The budget affects both store choice and the quantities that can actually be
ordered.

Item importance
---------------

``itemImportance`` gives each food a nonnegative replenishment weight. A
larger weight makes availability for that shortage more important in store
choice. When the budget is tight, orders are also processed in descending
importance, with ``foodItems`` order as the stable tie break.

.. code-block:: json

   "itemImportance": {
       "Rice": 3.0,
       "Chocolate": 0.5,
       "Sushi": 0.25
   }

An omitted item defaults to ``1.0`` and a negative configured value is clamped
to zero. Reorder points and target stocks remain independent parameters, so
``itemImportance`` does not by itself change when the shopping trigger fires.

Stores and household beliefs
----------------------------

Every visible agent whose observation has ``is_household: false`` is treated
as a store. Agent names and name prefixes do not affect this classification.
Households are excluded from store choice, belief updates, and order targets.

Each household has private, store-specific mappings:

.. code-block:: python

   expected_price[store_id][item]
   expected_availability[store_id][item]

For an unvisited store, expected price starts at the item's ``pricePriors``
value and expected availability starts at ``initialAvailability``. A
co-located offer updates the beliefs by exponential smoothing:

.. math::

   \hat p_{ij}\leftarrow
   (1-\rho_p)\hat p_{ij}+\rho_p p_{ij},

.. math::

   \hat a_{ij}\leftarrow
   (1-\rho_a)\hat a_{ij}+\rho_a a_{ij}.

An offered item with a positive or masked amount has observed availability
:math:`a_{ij}=1`; a missing item has :math:`a_{ij}=0`. Price is updated only
for offered items. ``beliefLearningRate`` supplies the default for both
learning rates, while ``priceLearningRate`` and
``availabilityLearningRate`` can override it separately.

Store-choice utility
--------------------

For item importance :math:`w_i`, desired replenishment :math:`q_i`, expected
availability :math:`\hat a_{ij}`, expected price :math:`\hat p_{ij}`, and
price prior :math:`\bar p_i`, define weighted need

.. math::

   W_t=\sum_i w_iq_i.

Expected coverage and relative price savings are

.. math::

   C_{tj}=\frac{\sum_iw_iq_i\hat a_{ij}}{W_t},

.. math::

   R_{tj}=\frac{\sum_iw_iq_i\hat a_{ij}
   (\bar p_i-\hat p_{ij})/\bar p_i}{W_t}.

The price term measures a store's expected price relative to the household's
prior for the *same item*. An expensive luxury item is therefore not penalized
merely for costing more than rice. Absolute affordability is represented by
expected expenditure and budget pressure:

.. math::

   E_{tj}=\sum_iq_i\hat a_{ij}\max(0,\hat p_{ij}),
   \qquad L_{tj}=E_{tj}/B_t.

The distance :math:`D_{tj}` is Chebyshev grid distance. Deterministic utility
is

.. math::

   V_{tj}=\beta_P R_{tj}+\beta_A C_{tj}
          -\beta_B L_{tj}-\beta_DD_{tj}.

A store is sampled with multinomial-logit probability

.. math::

   P(J_t=j)=\frac{\exp(V_{tj})}{\sum_r\exp(V_{tr})}.

The budget-pressure term means that the same expected premium basket deters a
cash-constrained household more strongly than a wealthy household. Coverage
is retained as a separate positive term so that a store cannot appear
attractive merely because carrying fewer desired goods makes its expected
expenditure small.

``storeChoice`` parameters
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Key
     - Default
     - Meaning
   * - ``betaPrice``
     - ``1.0``
     - Weight on expected relative price savings :math:`R_{tj}`.
   * - ``betaAvailability``
     - ``2.0``
     - Weight on importance-adjusted expected coverage :math:`C_{tj}`.
   * - ``betaBudgetPressure``
     - ``1.0``
     - Penalty weight on expected expenditure divided by shopping budget.
   * - ``betaDistance``
     - ``0.1``
     - Penalty per unit of Chebyshev grid distance.
   * - ``initialAvailability``
     - ``0.5``
     - Initial availability belief, clamped to ``[0, 1]``.
   * - ``beliefLearningRate``
     - ``0.5``
     - Default smoothing rate for both beliefs, clamped to ``[0, 1]``.
   * - ``priceLearningRate``
     - ``beliefLearningRate``
     - Optional price-specific smoothing rate.
   * - ``availabilityLearningRate``
     - ``beliefLearningRate``
     - Optional availability-specific smoothing rate.

With identical initial beliefs, price, availability, and budget-pressure terms
are initially common across stores; early choices are therefore mainly
distance-driven. Store differentiation emerges as households observe offers.

Ordering at the selected store
------------------------------

After arrival, the household uses the seller's observed price and stock rather
than its beliefs. Items are considered by descending importance. For positive
price :math:`p_i`, available stock :math:`A_i`, remaining budget :math:`B_i`,
and desired amount :math:`q_i^*`, the affordable amount is bounded by

.. math::

   \min\{q_i^*,A_i,B_i/p_i\}.

The current implementation truncates the emitted order amount to an integer.
Unspent budget can then be used by the next item in importance order.

Extending the policy
--------------------

The adapter, decision arbiter, and stylized models are separate classes.
Custom behavior can replace the default
:class:`~econsimulacra.agents.households.HouseholdDecisionPolicy` or add a
supplemental policy without modifying the environment action protocol. See the
API reference for
:mod:`econsimulacra.agents.households.household`,
:mod:`econsimulacra.agents.households.policy`, and
:mod:`econsimulacra.agents.households.stylized_models`.
