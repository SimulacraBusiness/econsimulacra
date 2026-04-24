Configuration Reference
=======================

All simulation settings are stored in a single JSON (or Python ``dict``)
that is passed to :class:`~econsimulacra.Simulator`.  This page documents
every top-level key and their required / optional sub-keys.

Top-Level Structure
-------------------

.. code-block:: json

   {
       "simulation": { ... },
       "environment": { ... },
       "<spaceName>": { ... },
       "<socialNetworkName>": { ... },
       "<agentName>": { ... },
       "<itemName>": { ... },
       "<serviceName>": { ... },
       "<eventName>": { ... }
   }

``simulation``
--------------

Controls the overall simulation run.

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Key
     - Required
     - Description
   * - ``numSteps``
     - ✓
     - Total number of environment steps to execute.
   * - ``parallelBatchSize``
     -
     - Number of agents that call :meth:`~econsimulacra.agents.Agent.act`
       concurrently per step.  Defaults to ``1`` (sequential).
   * - ``events``
     -
     - List of event configuration keys.  Each key must appear as a
       top-level entry in the config.

``environment``
---------------

Declares the components that make up the simulation world.

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Key
     - Required
     - Description
   * - ``space``
     - ✓
     - Key of the grid-space configuration block.
   * - ``socialNetwork``
     - ✓
     - Key of the social-network configuration block.
   * - ``cashName``
     - ✓
     - Name of the item that acts as the numéraire (e.g., ``"Yen"``).
       Must appear in ``items``.
   * - ``agents``
     - ✓
     - List of agent configuration keys.
   * - ``items``
     - ✓
     - List of item configuration keys.  Must include ``cashName``.
   * - ``service``
     -
     - List of service configuration keys (e.g., ``"llmClient"``).

Grid Space (``GridSpace``)
--------------------------

.. code-block:: json

   "gridSpace": {
       "type": "GridSpace",
       "gridSize": [50, 50]
   }

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Key
     - Required
     - Description
   * - ``type``
     - ✓
     - Class name — use ``"GridSpace"`` for the built-in implementation.
   * - ``gridSize``
     - ✓
     - ``[width, height]`` of the 2-D discrete grid.

Social Network (``SocialNetwork``)
-----------------------------------

.. code-block:: json

   "socialNetwork": {
       "type": "SocialNetwork",
       "followCap": 3,
       "recSys": {
           "type": "TwoHopRecommenderSystem",
           "maxRecommendations": 2,
           "isRandomized": true,
           "temperature": 0.4
       }
   }

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Key
     - Required
     - Description
   * - ``type``
     - ✓
     - Class name — use ``"SocialNetwork"`` for the built-in implementation.
   * - ``followCap``
     -
     - Maximum number of agents a single agent may follow.
       Omit for no limit.
   * - ``recSys``
     -
     - Recommender-system configuration block (see below).

``recSys`` sub-keys:

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Key
     - Required
     - Description
   * - ``type``
     - ✓
     - Recommender-system class name (e.g., ``"TwoHopRecommenderSystem"``).
   * - ``maxRecommendations``
     -
     - Maximum number of follow suggestions per step.
   * - ``isRandomized``
     -
     - Whether suggestions are sampled probabilistically.
   * - ``temperature``
     -
     - Softmax temperature :math:`\tau` used when ``isRandomized`` is
       ``true``.  Higher :math:`\tau` → more uniform sampling.

Agent Configuration
-------------------

Each agent key maps to a block that configures all agents of that type:

.. code-block:: json

   "Household": {
       "type": "LLMAgent",
       "isHousehold": true,
       "numAgents": 118,
       "inventory": {
           "Yen": [100000, 200000],
           "Rice": [3, 10]
       }
   }

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Key
     - Required
     - Description
   * - ``type``
     - ✓
     - Agent class name registered with the simulator.
   * - ``isHousehold``
     -
     - If ``true``, the agent can move on the grid.  If ``false``
       (firm / retailer), the agent can set item prices.  Defaults to
       ``false``.
   * - ``numAgents``
     -
     - Number of instances to create.  Defaults to ``1``.
   * - ``inventory``
     -
     - Initial inventory.  Each value is either a fixed number or a
       ``[min, max]`` range drawn uniformly at reset.
   * - ``requestObs``
     -
     - List of observation keys the agent requests from the environment.
       Use ``["all"]`` to request everything available.
   * - ``provideInfo4AllAgents``
     -
     - List of information keys this agent broadcasts to every other agent
       (e.g., ``["self_pos"]``).
   * - ``provideInfo4CoLocatedAgents``
     -
     - List of information keys shared only with co-located agents
       (e.g., ``["inventory"]``).
   * - ``isRichInfoAllowed``
     -
     - If ``true``, the agent receives market-price information.
       Defaults to ``false``.

Item Configuration
------------------

.. code-block:: json

   "Rice": {
       "type": "Item",
       "initialPrice": 1000.0
   }

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Key
     - Required
     - Description
   * - ``type``
     -
     - Item class name.  Defaults to the item key name.
   * - ``initialPrice``
     -
     - Initial market price :math:`p_0`.  Defaults to ``0.0``.

Event Configuration
-------------------

.. code-block:: json

   "provideSubsidy": {
       "type": "SubsidyEvent",
       "trigger": {
           "every": 30,
           "probability": 1.0
       },
       "subsidyAmount": 50000
   }

``trigger`` sub-keys:

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Key
     - Required
     - Description
   * - ``at``
     -
     - Tuple of step indices at which to fire the event.
   * - ``every``
     -
     - Period :math:`k`; the event fires at steps
       :math:`k, 2k, 3k, \ldots`
   * - ``between``
     -
     - ``[start, end]`` — the event is only eligible within this step range.
   * - ``with``
     -
     - List of :class:`~econsimulacra.logs.Log` class names; the event
       fires whenever one of these log entries is produced.
   * - ``probability``
     -
     - Probability :math:`p \in [0, 1]` with which the event fires on
       each eligible step.  Defaults to ``1.0``.
