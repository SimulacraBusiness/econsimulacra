Architecture Overview
=====================

EconSimulacra is built around four primary abstractions:
:class:`~econsimulacra.Simulator`,
:class:`~econsimulacra.envs.Environment`,
:class:`~econsimulacra.agents.Agent`, and
:class:`~econsimulacra.events.Event`.

.. code-block:: text

   ┌─────────────────────────────────────────────────────────┐
   │                        Simulator                        │
   │  ┌───────────────────────────────────────────────────┐  │
   │  │                    Environment                    │  │
   │  │  ┌─────────┐  ┌────────────────┐  ┌───────────┐  │  │
   │  │  │GridSpace│  │ SocialNetwork  │  │   Items   │  │  │
   │  │  └─────────┘  └────────────────┘  └───────────┘  │  │
   │  │  ┌──────────────────────────────────────────────┐ │  │
   │  │  │                   Agents                     │ │  │
   │  │  │  (Household / Retailer / LLMAgent / …)       │ │  │
   │  │  └──────────────────────────────────────────────┘ │  │
   │  │  ┌──────────────────────────────────────────────┐ │  │
   │  │  │               EventManager                   │ │  │
   │  │  └──────────────────────────────────────────────┘ │  │
   │  └───────────────────────────────────────────────────┘  │
   └─────────────────────────────────────────────────────────┘

Simulation Loop
---------------

The :class:`~econsimulacra.Simulator` drives the following loop (simplified):

.. code-block:: python

   env.reset(seed=seed)
   for step in range(num_steps):
       obs_per_agent  = {id: env.get_observations(id) for id in env.agent_ids}
       actions        = {id: await agent.act(obs_per_agent[id]) for ...}
       env.step(actions)

Concurrency is achieved by batching agent ``act`` calls with
:func:`asyncio.gather`, enabling multiple LLM inference requests to proceed
in parallel while the environment step itself remains synchronous.

Spatial Layer — GridSpace
--------------------------

Agents occupy a cell
:math:`(x, y) \in \{0,\ldots,W-1\} \times \{0,\ldots,H-1\}` of a discrete
2-D grid.  At each step an agent can move to an adjacent cell or
stay in place.

Social Layer — SocialNetwork
-----------------------------

The social network is a directed graph :math:`G = (V, E)` where each vertex
:math:`v \in V` corresponds to an agent and each directed edge
:math:`(u, v) \in E` indicates that agent :math:`u` follows agent :math:`v`.

Each agent is subject to an optional follow cap :math:`c`: the out-degree of
every vertex is bounded by :math:`c`.

A :class:`~econsimulacra.social_networks.RecommenderSystem` can suggest new
edges at each step.  The built-in ``TwoHopRecommenderSystem`` considers the
two-hop neighbourhood of each agent and scores candidates using a softmax
over edge weights:

.. math::

   P(\text{follow } v \mid u) = \frac{e^{w_{uv}/\tau}}{\sum_{v'} e^{w_{uv'}/\tau}}

where :math:`w_{uv}` is a relevance score and :math:`\tau > 0` is the
temperature parameter.

Market Layer — Items and Orders
--------------------------------

Each tradable good is represented by an :class:`~econsimulacra.items.Item`
that carries a current price :math:`p_t`.  An :class:`~econsimulacra.envs.Order`
is a bilateral trade request:

.. math::

   \text{Order} = (\text{agent}_i,\; \text{counterparty}_j,\; \text{item},\; q,\; p)

where :math:`q` is the quantity and :math:`p` is the agreed price.  Orders
expire after a configurable time-to-live (TTL).

Agent Layer
-----------

:class:`~econsimulacra.agents.Agent` is an abstract base class.  All concrete
agents must implement:

.. code-block:: python

   async def act(self, obs: ObsT) -> dict[str, Any]: ...

The built-in :class:`~econsimulacra.agents.LLMAgent` delegates action
generation to a language model via the following services:

* **LLMClient** — wraps the underlying model and manages inference.
* **PromptBuilder** — constructs the text prompt from the agent's
  observation.
* **PersonaBuilder** — assigns a role-playing persona at reset time.
* **MemoryHandler** — stores past experiences and retrieves relevant
  context.

Event Layer
-----------

:class:`~econsimulacra.events.Event` subclasses implement exogenous shocks or
policy interventions.  An :class:`~econsimulacra.events.EventTrigger`
specifies *when* the event fires using four independent conditions that are
combined with logical AND:

* ``at`` — specific step indices.
* ``every`` — periodic cadence :math:`k`.
* ``between`` — a step-range guard.
* ``with`` — reaction to a particular log type.
* ``probability`` — probabilistic gate :math:`p`.

Logging
-------

Every state change is recorded as a typed :class:`~econsimulacra.logs.Log`
dataclass (e.g., :class:`~econsimulacra.logs.OrderLog`,
:class:`~econsimulacra.logs.ConsumptionLog`).  A
:class:`~econsimulacra.logs.Logger` collects these records and persists them
at the end of the simulation.
