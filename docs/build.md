
# Bump the version of the project
```
$ poetry version [major, minor, patch]  
```

# Create tag
```
$ pyhton -m geec run --version
> <version>
$ git tag <version>  
```

# Builds the source and wheels archives
```
$ poetry build 
```

# Publish on Pypi (optional)
```
$ poetry publish
```