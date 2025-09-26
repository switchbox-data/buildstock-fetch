# buildstock-fetch

[![Release](https://img.shields.io/github/v/release/switchbox-data/buildstock-fetch)](https://img.shields.io/github/v/release/switchbox-data/buildstock-fetch)
[![Build status](https://img.shields.io/github/actions/workflow/status/switchbox-data/buildstock-fetch/main.yml?branch=main)](https://github.com/switchbox-data/buildstock-fetch/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/switchbox-data/buildstock-fetch)](https://img.shields.io/github/commit-activity/m/switchbox-data/buildstock-fetch)
[![License](https://img.shields.io/github/license/switchbox-data/buildstock-fetch)](https://img.shields.io/github/license/switchbox-data/buildstock-fetch)

A CLI tool to simplify downloading building characteristics and load curve data from NREL's ResStock and ComStock projects. Also available as a Python library.

## Installing bsf

To install the buildstock-fetch CLI tool, we recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) or [pipx](https://pipx.pypa.io/stable/installation/):

```bash
uv install buildstock-fetch
```
or

```bash
pipx install buildstock-fetch
```

You'll then be able to access the `bsf` command system-wide:

```bash
bsf --help
```

## Using bsf

To make it easy to download what you want from NREL's S3 bucket, `bsf` has an interactive mode. Activate it with:

```bash
bsf
```

Alternatively, you can tell `bsf` exactly what to download via CLI args:

```bash
bsf --product resstock --release_year 2022 --weather_file tmy3 --release_version 1 --states CA --file_type "hpxml metadata" --upgrade_id "0 1 2" --output_directory ./CA_data
```

For more details about the usage, see [Usage](https://switchbox-data.github.io/buildstock-fetch/usage/)


## Installing the Python library

buildstock-fetch is implemented in Python, and we expose our internal functions via a library.

If you're using Python, and you want to install the CLI only for a particular project (rather than system-wide), or you want to user the underlying library, install our [PyPI](https://pypi.org/project/buildstock-fetch/) package via:

```bash
pip install buildstock-fetch
```

or

```bash
uv add buildstock-fetch
```
