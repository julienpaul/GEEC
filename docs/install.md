
# Install for user

Set up/update package library

<!-- 
> **Note:** Recommended python release >= 3.11
> 
> Using pyenv, activate Python 3.11.4 for the current project
> ```
> $ pyenv local 3.11.4
> ```
-->

## Install one of github hosted repo's specific tag

### Using pip
```
$ pip(3) install -e git+https://github.com/{ username }/{ repo name }.git@{ tag name }#egg={ desired egg name }
```

> for 'egg name', use the contents of project-name.egg-info/top_level.txt

### Exemple, for tag 0.0.3
```
$ pip install git+https://github.com/julienpaul/GEEC.git@0.0.3#egg=geec
```

## Install package
```
$ python3 -m pip install path/to/package/dist/icp2edd.SOME_RELEASE.tar.gz
```

## Install package in development mode
```
$ pip(3) install -e path/to/package
```

---

# Install for developper

## Install with Poetry

- Install [Poetry](https://python-poetry.org/docs/#installation)

- Clone git repository and install libraries with Poetry:

```shell
$ git clone https://github.com/<your_github_username>/geec.git
$ cd geec
$ poetry install
```

## Install with Conda

- Install [Conda](https://docs.conda.io/en/latest/miniconda.html)

- Clone git repository, install virtual environment **requirements.txt** and libraries:

```
$ git clone https://github.com/<your_github_username>/geec.git
$ cd geec
$ conda env create -f environment.yml
```
- activate virtual environment **geec-env**:
```shell
$ conda activate geec-env
```
> **Note:**
> - deactivate virtual environment **geec-env**:
> ```shell
> $ conda deactivate
> ```
>
> - delete virtual environment **geec-env**
> ```bash
> $ conda env remove --name geec-env
> ```

## Install with pip and virtualenv

```shell
$ git clone https://github.com/<your_github_username>/geec.git
$ cd geec
$ python -m venv .env
```

On Unix or MacOS, run:
```
$ source .env/bin/activate
```
On Windows, run:
```
$ .env\Scripts\activate.bat
```

```
$ python -m pip install --upgrade pip
$ python -m pip install -r requirements.txt
```

> **Note:**
> - deactivate virtual environment **.env**:
> ```shell
> $ deactivate
> ```