
## Get help/usage on 'package' from terminal
```
$ python -m geec --help
```
<!--
or (using poetry)
```
$ poetry run geec --help
```
-->

### Get version
```
$ pyhton -m geec --version
```

## Compute gravity fields [mGal] from a mass body at some observation points.

```
$ python -m geec run <output>
```

> **_Note:_**  
> Get help/usage message  
> ```
> $ python -m geec run --help
> ```
  
## Create a template of the configuration file

```
$ python -m geec config
```
  
> **_Note:_**  
> Get help/usage message  
> ```
> $ python -m geec config --help
> ```
  
### Configuration file
This file contains configuration parameters

There is a default configuration file (see below).
You can create your own default configuration file by putting it in `~/.config/geec/config.yaml`.
Finally configuration file can be added as arguments.

> **_NOTE:_**  
> Parameters come first from the argument configuration file, 
> secondly from your own default configuration file, 
> and finally from the default configuration file.

#### Default configuration is:
~~~yaml
{% include "../geec/cfg/config_default.yaml" %}
~~~