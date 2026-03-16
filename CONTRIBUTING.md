# Contributing

Contributions are always greatly appreciated and credit will always be given.

## Types of contributions

### Report bugs

Report bugs at https://github.com/martibosch/aifs-modal/issues.

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

### Implement features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

## Pull request guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
1. If the pull request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring, and add the feature to the list in README.md.
1. Install [pixi](https://pixi.sh/latest/) and use it to work in the project environments, for example `pixi install -e dev`, `pixi run -e test test`, and `pixi run -e doc build-doc`.
1. Commit `pixi.lock` after resolving dependencies, and include updates to it in pull requests whenever dependency changes require a new solve.
1. The pull request should work for Python 3.10, 3.11, 3.12, and 3.13. Check https://github.com/martibosch/aifs-modal/actions and make sure that the tests pass for all supported Python versions.
