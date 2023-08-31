# Geec

[![Release](https://img.shields.io/github/v/release/julienpaul/geec)](https://img.shields.io/github/v/release/julienpaul/geec)
[![Build status](https://img.shields.io/github/actions/workflow/status/julienpaul/geec/main.yml?branch=main)](https://github.com/julienpaul/geec/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/julienpaul/geec/branch/main/graph/badge.svg)](https://codecov.io/gh/julienpaul/geec)
[![License](https://img.shields.io/github/license/julienpaul/geec)](https://img.shields.io/github/license/julienpaul/geec)
<!-- [![Commit activity](https://img.shields.io/github/commit-activity/m/julienpaul/geec)](https://img.shields.io/github/commit-activity/m/julienpaul/geec) -->

[![Issues](https://img.shields.io/github/issues-raw/julienpaul/geec)](https://github.com/julienpaul/geec/issues)
[![Bugs](https://img.shields.io/github/issues/julienpaul/geec/bug?color=red&label=known%20bugs)](https://github.com/julienpaul/geec/issues?q=is%3Aissue+is%3Aopen+label%3Abug)
[![Pull Requests](https://img.shields.io/github/issues-pr/BjerknesClimateDataCentre/QuinCe)](https://github.com/julienpaul/geec/pulls)

[![cookiecutter](https://img.shields.io/badge/built%20with-Cookiecutter%20Poetry-ff69b4.svg?logo=cookiecutter)](https://github.com/fpgmaas/cookiecutter-poetry)

Program to calculate gravity fields [mGal] and gravity gradients fields [E] due to irregularly shaped bodies.

---
<p align="center">
  <a href="https://julienpaul.github.io/GEEC/">Documentation</a> - <a href="https://github.com/julienpaul/geec/">Github repository</a>
  <!--
   - <a href="https://pypi.org/project/geec/">PyPi</a>
  -->
</p>

---
### To get help/usage on 'package' from terminal
```
$ python -m geec --help
```

### To compute gravity fields [mGal] from a mass body at some observation points.
```
$ python3 -m geec run <ouptut>
```

### To compute gravity fields [mGal] and gradient gravity fields [E] from a mass body at some observation points.
```
$ python3 -m geec run <ouptut> --gradient
```

<!--
## Installation

### Installation using Poetry

- Install [Poetry](https://python-poetry.org/docs/#installation)

- clone git repo:

  ```shell
  git clone https://github.com/julienpaul/geec.git
  cd geec
  poetry install
  ```

## Getting started with your project

First, create a repository on GitHub with the same name as this project, and then run the following commands:

``` bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:julienpaul/geec.git
git push -u origin main
```

Finally, install the environment and the pre-commit hooks with

```bash
make install
```

You are now ready to start development on your project! The CI/CD
pipeline will be triggered when you open a pull request, merge to main,
or when you create a new release.

To finalize the set-up for publishing to PyPi or Artifactory, see
[here](https://fpgmaas.github.io/cookiecutter-poetry/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see
[here](https://fpgmaas.github.io/cookiecutter-poetry/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-poetry/features/codecov/).

## Releasing a new version

- Create an API Token on [Pypi](https://pypi.org/).
- Add the API Token to your projects secrets with the name `PYPI_TOKEN` by visiting
[this page](https://github.com/julienpaul/geec/settings/secrets/actions/new).
- Create a [new release](https://github.com/julienpaul/geec/releases/new) on Github.
Create a new tag in the form ``*.*.*``.

For more details, see [here](https://fpgmaas.github.io/cookiecutter-poetry/features/cicd/#how-to-trigger-a-release).

---

Repository initiated with [fpgmaas/cookiecutter-poetry](https://github.com/fpgmaas/cookiecutter-poetry).
-->
