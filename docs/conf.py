"""Documentation configuration."""

import dataclasses
import os
import sys
from importlib import metadata

from sphinx.builders.latex import transforms
from sphinxcontrib.bibtex import plugin as sphinxcontrib_bibtex_plugin
from sphinxcontrib.bibtex.style.referencing import BracketStyle
from sphinxcontrib.bibtex.style.referencing.author_year import AuthorYearReferenceStyle

project = "AIFS on Modal"
author = "Martí Bosch"

release = metadata.version("aifs_modal")
version = ".".join(release.split(".")[:2])


extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.mermaid",
]

exclude_patterns = [
    "_build",
    "jupyter_execute",
    "jupyter_execute/**",
    "**/.ipynb_checkpoints",
    "**/.ipynb_checkpoints/*",
]
nb_execution_mode = "off"
nb_render_markdown_format = "myst"

autodoc_typehints = "description"
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/martibosch/aifs-modal",
}

# add module to path
sys.path.insert(0, os.path.abspath(".."))


# citation styles
def bracket_style() -> BracketStyle:
    """Bracket style."""
    return BracketStyle(
        left="(",
        right=")",
    )


@dataclasses.dataclass
class MyReferenceStyle(AuthorYearReferenceStyle):
    """Custom reference style."""

    bracket_parenthetical: BracketStyle = dataclasses.field(
        default_factory=bracket_style
    )
    bracket_textual: BracketStyle = dataclasses.field(default_factory=bracket_style)
    bracket_author: BracketStyle = dataclasses.field(default_factory=bracket_style)
    bracket_label: BracketStyle = dataclasses.field(default_factory=bracket_style)
    bracket_year: BracketStyle = dataclasses.field(default_factory=bracket_style)


sphinxcontrib_bibtex_plugin.register_plugin(
    "sphinxcontrib.bibtex.style.referencing", "author_year_round", MyReferenceStyle
)


# work-around to get LaTeX references at the same place as HTML
# see https://github.com/mcmtroffaes/sphinxcontrib-bibtex/issues/156
class DummyTransform(transforms.BibliographyTransform):
    """Dummy transform."""

    def run(self, **kwargs):
        """Run."""
        pass


transforms.BibliographyTransform = DummyTransform

# bibliography
bibtex_bibfiles = ["user-guide/references.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year_round"
