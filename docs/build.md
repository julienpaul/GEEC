
## Build and publish new release (developper)

### Bump the version of the project
```
$ poetry version [major, minor, patch]  
```

### Create tag
```
$ git tag <version>  
```

### Export requirements.txt
```
$ poetry export --format=requirements.txt > requirements.txt
```
<!-- $ poetry export --without-hashes --format=requirements.txt > requirements.txt -->
### Export environment.yml
```
$ poetry2conda pyproject.toml environment.yml
```

> **_Note:_**  
> Until a new release of poetry2conda, add **channels** to environment.yml  
> ```
> channels:  
>   - conda-forge  
>   - defaults
> ```


### Make online documentation
You need to create a new release on github from the latest tag.  
Github Actions will automatically generate documentation

### Builds the source and wheels archives (optional)
```
$ poetry build 
```

### Publish on Pypi (optional)
```
$ poetry publish
```