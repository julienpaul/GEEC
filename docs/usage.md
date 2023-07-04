
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
$ pyhton -m geec run --version
```

## Compute gravity fields [mGal] from a mass body at some observation points.

```
$ python3 -m geec run <output>
```

> **_Note:_**  
> Get help/usage message  
> ```
> $ python3 -m geec run --help
> ```
  
## Create a template of the configuration file

```
$ python3 -m geec config
```
  
> **_Note:_**  
> Get help/usage message  
> ```
> $ python3 -m geec run --help
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
```python
# Mass Bodies
mass:
  # choose one between [points, file_path]
  points: # list of points [x,y,z]
  file_path: "./geec/data/cube.csv" # full path to file with mass body points
  density: 1000. # density of the mass body [kg m-3]
  gravity_constant: 6.67408e-11 # gravity constant [m3 kg−1 s−2]

# Observation points
obs:
  # choose one between [points, file_path, grid]
  points: [] # list of points [(x,y,z),(),..]
  input: "" # path to file with observation points
  grid:
    xstart_xend_xstep: [-1.05,1.06,0.1]
    ystart_yend_ystep: [-1.05,1.06,0.1]
    zstart_zend_zstep: [0,1,1]
```