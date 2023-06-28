
# Install for user

Set up/update package library

## Install package
```
$ python3 -m pip install path/to/package/dist/icp2edd.SOME_RELEASE.tar.gz
```

## Install package in development mode
```
$ pip(3) install -e path/to/package
```

## Install one of github hosted repo's specific tag

### Using pip
```
$ pip(3) install -e git+https://github.com/{ username }/{ repo name }.git@{ tag name }#egg={ desired egg name }
```

> for 'egg name', use the contents of project-name.egg-info/top_level.txt

### Exemple, for tag 0.0.2
```
$ pip install git+https://github.com/julienpaul/GEEC.git@0.0.2#egg=geec
```


# Install for developper

## Install with Conda

- Install [Conda](https://docs.conda.io/en/latest/miniconda.html)

- Clone git repository, install virtual environment **requirements.txt** and libraries:

  ```shell
  git clone https://github.com/<your_github_username>/geec.git
  cd geec
  conda create --name <env_name> --file requirements.txt
  ```
- activer l'environement virutel **<env_name>**:
  ```shell
  conda activate <env_name>
  ```
> **Note:**
> - desactiver l'environement virutel **<env_name>**:
>  ```shell
>  conda deactivate
>  ```
>
> - supprimer l'environement virutel **<env_name>**
> ```bash
> conda env remove --name <env_name>
> ```

## Install with Poetry

- Install [Poetry](https://python-poetry.org/docs/#installation)

- Clone git repository and install libraries with Poetry:

  ```shell
  git clone https://github.com/<your_github_username>/geec.git
  cd geec
  poetry install
  ```

## Install with pip and virtualenv

```shell
pip install -r requirements.txt
python -m venv .env
.env\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
