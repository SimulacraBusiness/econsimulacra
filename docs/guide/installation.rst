Installation
============

EconSimulacra requires **Python ≥ 3.10, < 3.13**.

PyPI
----

.. code-block:: bash

   pip install econsimulacra

From Source
-----------

.. code-block:: bash

   git clone https://github.com/SimulacraBusiness/econsimulacra.git
   cd econsimulacra
   pip install poetry
   poetry install

Optional: vLLM Backend
-----------------------

The vLLM backend requires a separate virtual environment because some of its
dependencies conflict with the core package:

.. code-block:: bash

   python3.10 -m venv .venv-vllm
   source .venv-vllm/bin/activate
   pip install vllm
