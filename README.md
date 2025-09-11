# buildstock-fetch

[![Release](https://img.shields.io/github/v/release/switchbox-data/buildstock-fetch)](https://img.shields.io/github/v/release/switchbox-data/buildstock-fetch)
[![Build status](https://img.shields.io/github/actions/workflow/status/switchbox-data/buildstock-fetch/main.yml?branch=main)](https://github.com/switchbox-data/buildstock-fetch/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/switchbox-data/buildstock-fetch)](https://img.shields.io/github/commit-activity/m/switchbox-data/buildstock-fetch)
[![License](https://img.shields.io/github/license/switchbox-data/buildstock-fetch)](https://img.shields.io/github/license/switchbox-data/buildstock-fetch)

This library simplifies downloading building characteristics and load curve data from NREL's ResStock and ComStock projects.

- **Github repository**: <https://github.com/switchbox-data/buildstock-fetch/>
- **Documentation**: <https://switchbox-data.github.io/buildstock-fetch/>
- **PyPI page***: <https://pypi.org/project/buildstock-fetch/>

## Installing the library

`buildstock-fetch` is available on [PyPI](https://pypi.org/project/buildstock-fetch/) and can be installed via:

```bash
pip install buildstock-fetch
```

or

```bash
uv add buildstock-fetch
```

## Using the library

To use the CLI tool in `buildstock-fetch` run:

```bash
bsf
```

to activate the interactive mode. Alternatively, provide the inputs directly. For example,

```bash
bsf --product resstock --release_year 2022 --weather_file tmy3 --release_version 1 --states CA --file_type "hpxml metadata" --upgrade_id "0 1 2" --output_directory ./CA_data
```

If the above options don't work, try:

```bash
python -m buildstock_fetch
```

For more details about the usage, see [Usage](https://switchbox-data.github.io/buildstock-fetch/usage/)

## Developing the library

To simplify development, this library uses [devcontainers](https://containers.dev/) for reproducible development environments and the [just](https://github.com/casey/just) command runner for all major tasks.

Available commands are defined in [Justfile](Justile). To view available them:

```bash
just --list
```

### Setting up development environment

This library uses [uv](https://docs.astral.sh/uv/) for managing python versions, virtual environments, and packages.

However, given the presence of system dependencies, the easiest way to set up the library's develop environment is to use devcontainers. To do so, open up the repo in VSCode, or a VSCode fork like Cursor or Positron.

The editor will auto-detect the presence of the repo's devcontainer (configured in [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json)). Click "Reopen in Container" to launch the devcontainer.

If you prefer not to use a devcontainer, you can install uv, install the pre-commit hooks, launch the virtualenv, and download the packages (pinned in [uv.lock](uv.lock)) by running:

```bash
./devcontainer/postCreateCommand.sh
just install
```

Using a system package manager (like `brew` or `apt`), you'll also need to manually install `just` and `quarto`.

### Adding a package
To add a python package to the project:

```bash
uv add <package-name>
```

This replaces `pip install <package-name>`, and has the effect of adding the package to [pyproject.toml](pyproject.toml), as well as pinning the package version in [uv.lock](uv.lock).

To add a package that will only be used as a development tool:

```bash
uv add --dev <package-name>
```

This will update the dev `dependency-groups` in [pyproject.toml](pyproject.toml), and ensure that the package isn't declared as a run-time dependency of the library itself.

### Running code quality checks

We use [ruff](https://github.com/astral-sh/ruff) for linting and code formatting, [mypy](https://mypy.readthedocs.io/en/stable/running_mypy.html) for type checking, and a series of [post-commit hooks](pre-commit-config.yaml) for validating YAML, JSON, whitespaces, and so on.

To run code quality checks:

```bash
just check
```

The checks will also be run automatically by Github Actions when opening PRs, merging to main, or creating a new release.

### Running tests

Our test are written using `pytest` and live in [tests/](tests/). They are checked against multiple python versions using `tox`.

To run the tests:

```bash
just test
```

### Rendering docs

We use `mkdocs` and [mkdocs-material](https://squidfunk.github.io/mkdocs-material/) for writing and rendering docs. The docs are written in markdown and live in [docs/](docs/).

To dynamically render the docs as you develop them:

```bash
just docs
```

To statically render the docs into HTML:

```bash
just docs-test
```

The docs are served by Github Pages out of the `gh-pages` branch. We do not publish them manually: they are automatically rendered and published by the Github Actions workflow when merging to `main`.

### Releasing the library

To build the package into a wheel file, type:

```bash
just build
```

To publish a release to PyPI using [twine](https://pypi.org/project/twine/):

```bash
just publish
```

To do both in one go:

```bash
just build-and-publish
```

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
