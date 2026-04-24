Quick Start
===========

This page shows how to run the bundled OpenAI example, and then how to write
a minimal custom simulation from scratch.

Running the OpenAI Example
--------------------------

1. **Set your API key**

   .. code-block:: bash

      export OPENAI_API_KEY="sk-..."

2. **Set the log output path** (optional)

   .. code-block:: bash

      export LOG_TXT_PATH="simulation.log"

3. **Run the example**

   .. code-block:: bash

      cd examples/openai
      python main.py

   The script loads ``config.json``, registers the custom ``SubsidyEvent``,
   resets the environment, runs the simulation loop, and writes structured logs
   to the configured path.

Minimal Custom Simulation
--------------------------

The following snippet shows the core objects and the simulation loop that
:class:`~econsimulacra.Simulator` manages internally:

.. code-block:: python

   import asyncio
   import pathlib

   from econsimulacra import Simulator, SimulationSummarizer
   from econsimulacra.envs import Environment
   from econsimulacra.logs import DictLogger

   config_path = pathlib.Path("examples/openai/config.json")
   logger = DictLogger()

   simulator = Simulator(
       config=config_path,
       env_class=Environment,
       logger=logger,
       summarizer_class=SimulationSummarizer,
   )

   asyncio.run(simulator.simulate(seed=42))

Registering Custom Classes
--------------------------

Any custom :class:`~econsimulacra.agents.Agent`,
:class:`~econsimulacra.events.Event`, or
:class:`~econsimulacra.items.Item` subclass must be registered before the
simulation starts so that the environment can instantiate them by name:

.. code-block:: python

   from my_project import MyCustomAgent, MyCustomEvent

   simulator.register_classes([MyCustomAgent, MyCustomEvent])
   asyncio.run(simulator.simulate(seed=0))
