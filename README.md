# buildstock-fetch

[![Release](https://img.shields.io/github/v/release/switchbox-data/buildstock-fetch)](https://img.shields.io/github/v/release/switchbox-data/buildstock-fetch)
[![Build status](https://img.shields.io/github/actions/workflow/status/switchbox-data/buildstock-fetch/main.yml?branch=main)](https://github.com/switchbox-data/buildstock-fetch/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/switchbox-data/buildstock-fetch)](https://img.shields.io/github/commit-activity/m/switchbox-data/buildstock-fetch)
[![License](https://img.shields.io/github/license/switchbox-data/buildstock-fetch)](https://img.shields.io/github/license/switchbox-data/buildstock-fetch)

This library simplifies downloading building characteristics and load curve data from NREL's ResStock and ComStock projects.

- **Github repository**: <https://github.com/switchbox-data/buildstock-fetch/>
- **Documentation** <https://switchbox-data.github.io/buildstock-fetch/>

## Getting start with the project

### 1. Set Up Your Development Environment

The easiest way to set up the library's dev environment is to use devcontainers. To do so, open up the repo in VSCode or a VSCode fork like Cursor or Positron. The editor will auto-detect the presence of the repo's devcontainer (configured in `.devcontainer/devcontainer.json`). Click "Reopen in Container" to launch the devcontainer.

Alternatively, you can install the environment and the pre-commit hooks on your laptop with

```bash
make install
```

You are now ready to start development on the library!
The github action CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

### 2. Set up PyPI publishing

To finalize the set-up for publishing to PyPI, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/publishing/#set-up-for-pypi).

### 3. Activate automatic documentation
For activating the automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github).

## Releasing a new version

- Create an API Token on [PyPI](https://pypi.org/).
- Add the API Token to your projects secrets with the name `PYPI_TOKEN` by visiting [this page](https://github.com/switchbox-data/buildstock-fetch/settings/secrets/actions/new).
- Create a [new release](https://github.com/switchbox-data/buildstock-fetch/releases/new) on Github.
- Create a new tag in the form `*.*.*`.

For more details, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/cicd/#how-to-trigger-a-release).

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
