# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make the package importable without a full installation
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -------------------------------------------------------
project = "EconSimulacra"
copyright = "2026, Simulacra Inc"
author = "Ryuji Hashimoto"
release = "0.11.5"

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

autosummary_generate = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

# Heavy runtime dependencies that are not installed in the docs environment
autodoc_mock_imports = [
    "torch",
    "numpy",
    "openai",
    "outlines",
    "rich",
    "accelerate",
    "datasets",
    "matplotlib",
    "transformers",
    "statsmodels",
    "sentencepiece",
    "protobuf",
    "tqdm",
    "manim",
    "typing_extensions",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Suppress warnings that are acceptable (e.g. network unavailable in CI,
# pre-existing docstring formatting issues in source files).
suppress_warnings = [
    "intersphinx.fetch",  # network not always available in CI
]

# -- Options for HTML output ---------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
}
