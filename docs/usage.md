
# To run 'package' from terminal
```
$ python -m geec --help
```
or
```
$ poetry run geec --help
```

> **_Note:_**
> To get help/usage message
> ```
> $ python3 -m geec --help
> ```

# To compute gravity fields [mGal] from a mass body at some observation points.

```
$ python3 -m geec run
```

> **_Note:_**
> To get help/usage message
> ```
> $ python3 -m geec run --help
> ```
  
# To create a template of the configuration file

```
$ python3 -m geec config
```
  
> **_Note:_**
> To get help/usage message
> ```
> $ python3 -m geec run --help
> ```
  
## Configuration file
This file contains configuration parameters

There is a default configuration file (see below).
You can create your own default configuration file by putting it in `~/.config/geec/config.yaml`.
Finally configuration file can be added as arguments.

Parameters come first from the argument configuration file, 
secondly from your own default configuration file, 
and finally from the default configuration file.

> **NOTE:** arguments overwrite value in configuration file.

> **_NOTE:_** Default configuration is:
> ```python
> # Mass Bodies
> mass:
>   # choose one between [points, file_path]
>   points: # list of points [x,y,z]
>   file_path: "./geec/data/cube.csv" # full path to file with mass body points
>   density: 1000. # density of the mass body [kg m-3]
>   gravity_constant: 6.67408e-11 # gravity constant [m3 kg−1 s−2]
> 
> # Observation points
> obs:
>   # choose one between [points, file_path, grid]
>   points: [] # list of points [(x,y,z),(),..]
>   input: "" # path to file with observation points
>   grid:
>     xstart_xend_xstep: [-1.05,1.06,0.1]
>     ystart_yend_ystep: [-1.05,1.06,0.1]
>     zstart_zend_zstep: [0,1,1]
> ```
  
  
  