"""Documentation configuration."""

import os
import sys
from importlib import metadata

project = "AIFS on Modal"
author = "Martí Bosch"

release = metadata.version("aifs_modal")
version = ".".join(release.split(".")[:2])


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "nbsphinx",
    "sphinxcontrib.mermaid",
]

exclude_patterns = ["_build", "**/.ipynb_checkpoints", "**/.ipynb_checkpoints/*"]
nbsphinx_execute = "never"

autodoc_typehints = "description"
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/martibosch/aifs-modal",
}

# add module to path
sys.path.insert(0, os.path.abspath(".."))
