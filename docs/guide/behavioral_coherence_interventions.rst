Behavioral-Coherence Interventions
==================================

The ``examples/exp_behavioral_coherence`` experiment provides six intervention
conditions derived from one common baseline. The ``*_llm.json`` variants keep
the baseline population and environment settings and use
``openai/gpt-oss-120b`` for the single LLM household.

Conditions
----------

.. list-table::
   :header-rows: 1
   :widths: 12 30 58

   * - Condition
     - Configuration
     - Intervention
   * - E1
     - ``config_e1_price_increase_llm.json``
     - ``DiscountRetailer`` raises Daily Mart staple prices by 10 percent from
       step 144. It uses the same time lookup and ``discounts`` schema as the
       discount-restaurant example; a negative discount is a markup.
   * - E2
     - ``config_e2_supply_reduction_llm.json``
     - Daily Mart is removed from ``ConstantSupply`` and moved to
       ``DynamicSupply``. Its replenishment ratio changes from ``0.5`` to
       ``0.25`` at step 144.
   * - M1
     - ``config_m1_keep_out_llm.json``
     - ``KeepOut`` prevents entry into the four common-space cells from step
       144 through the end of the run.
   * - M2
     - ``config_m2_fast_car_llm.json``
     - A same-price ``HighPerformanceCar`` with velocity ``4`` becomes
       available at step 144; the baseline ``GasolineCar`` has velocity ``3``.
   * - S1
     - ``config_s1_positive_info_llm.json``
     - Rule-based households may replace an ordinary tweet with positive
       restaurant information from ``tweet_candidates``.
   * - S2
     - ``config_s2_misinformation_llm.json``
     - Rule-based households may replace an ordinary tweet with false Daily
       Mart stockout information from ``tweet_candidates``.

``DiscountRetailer`` is experiment-specific and lives in
``intervention_agents.py``. ``DynamicSupply`` and ``KeepOut`` are reusable
built-in events documented in :doc:`configuration`.

Running a condition
-------------------

Run commands from the repository root. For example, run E1 once with seed 42:

.. code-block:: console

   .venv/bin/python examples/exp_behavioral_coherence/main.py \
       --config examples/exp_behavioral_coherence/config_e1_price_increase_llm.json \
       --condition e1_price_increase \
       --model-label gpt-oss-120b \
       --seed-start 42 \
       --num-seeds 1

``--config`` determines simulation behavior. ``--condition`` and
``--model-label`` only determine the output directory; omitting
``--condition`` uses ``baseline`` and can overwrite a baseline log with the
same seed. Results are written to:

.. code-block:: text

   examples/exp_behavioral_coherence/logs/<condition>/<model-label>/<seed>.txt

The no-argument default deliberately runs the non-LLM ``config_baseline.json``
so a routine local check cannot start a vLLM server accidentally. Add
``--summarize`` to print the simulation configuration summary. The LLM files
configure GPU IDs 0 and 1 and allow ``VLLMClient`` to start its server as
usual.

Candidate paths
---------------

Candidate-file paths in JSON are relative to the configuration file. The
experiment runner resolves them to absolute paths before creating the
simulator, so invocation from another working directory does not change which
file is loaded.
