[![PyPI version fury.io](https://badge.fury.io/py/aifs-modal.svg)](https://pypi.python.org/pypi/aifs-modal)
[![Documentation Status](https://readthedocs.org/projects/aifs-modal/badge/?version=latest)](https://aifs-modal.readthedocs.io/en/latest/?badge=latest)
[![CI/CD](https://github.com/martibosch/aifs-modal/actions/workflows/tests.yml/badge.svg)](https://github.com/martibosch/aifs-modal/blob/main/.github/workflows/tests.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/martibosch/aifs-modal/main.svg)](https://results.pre-commit.ci/latest/github/martibosch/aifs-modal/main)
[![codecov](https://codecov.io/gh/martibosch/aifs-modal/branch/main/graph/badge.svg?token=hKoSSRn58a)](https://codecov.io/gh/martibosch/aifs-modal)
[![GitHub license](https://img.shields.io/github/license/martibosch/aifs-modal.svg)](https://github.com/martibosch/aifs-modal/blob/main/LICENSE)

# AIFS on Modal

Serverless AIFS setup on Modal

## Installation

```bash
pip install aifs-modal
```

## Development

Install [pixi](https://pixi.sh/latest) and use the generated environments and tasks:

```bash
pixi install -e dev
pixi run -e test test
pixi run -e doc build-doc
```

Commit the generated `pixi.lock` file once you have resolved the environments for your project, and update it whenever dependencies change.

## Acknowledgements

- This package was created with the [martibosch/cookiecutter-pypixi-package](https://github.com/martibosch/cookiecutter-pypixi-package) project template.
