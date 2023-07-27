"""
Read the configuration file, and check parameters values.
"""

# --- import -----------------------------------
# import from standard lib
# import from other lib
import confuse  # Initialize config with your app
from loguru import logger

# import from my project
import geec


def show(cfg):
    logger.info(f"Version: {geec.__version__}")
    logger.info("Configuration:")
    logger.info("Mass Bodies")
    for mass in cfg["masses"]:
        logger.info("  ---")
        logger.info(f"  density: {mass['density']} kg m-3")
        logger.info(f"  gravity_constant: {mass['gravity_constant']} m3 kg-1 s-2")
        logger.info("  coordinates:")
        logger.info(f"    points  : {mass['points']}")
        logger.info(f"    file_path: {mass['file_path']}")
        logger.info("  crs:")
        logger.info(f"    ref: {mass['crs']['ref']}")
        logger.info("    ellipsoid:")
        logger.info(f"      name: {mass['crs']['name']}")
        logger.info(f"      semimajor_axis: {mass['crs']['semimajor_axis']}")
        logger.info(f"      semiminor_axis: {mass['crs']['semiminor_axis']}")
        logger.info("    local ENU:")
        logger.info(f"      longitude_origin: {mass['crs']['longitude_origin']}")
        logger.info(f"      latitude_origin: {mass['crs']['latitude_origin']}")
        logger.info(f"      altitude_origin: {mass['crs']['altitude_origin']}")
        logger.info("    Vertical Datum:")
        logger.info(f"      vdatum: {mass['crs']['vdatum']}")
        logger.info("  ---")

    logger.info("Observers")
    obs = cfg["observers"]
    # choose one between [points, file_path, grid]
    logger.info("  coordinates:")
    logger.info(f"    points: {obs['points']}")
    logger.info(f"    file_path: {obs['file_path']}")
    logger.info("    grid:")
    grid = obs["grid"]
    logger.info(f"      xstart_xend_xstep: {grid['xstart_xend_xstep']}")
    logger.info(f"      ystart_yend_ystep: {grid['ystart_yend_ystep']}")
    logger.info(f"      zstart_zend_zstep: {grid['zstart_zend_zstep']}")
    logger.info("  crs:")
    logger.info(f"    ref: {obs['crs']['ref']}")
    logger.info("    ellipsoid:")
    logger.info(f"       name: {obs['crs']['name']}")
    logger.info(f"       semimajor_axis: {obs['crs']['semimajor_axis']}")
    logger.info(f"       semiminor_axis: {obs['crs']['semiminor_axis']}")
    logger.info("    local ENU:")
    logger.info(f"      longitude_origin: {obs['crs']['longitude_origin']}")
    logger.info(f"      latitude_origin: {obs['crs']['latitude_origin']}")
    logger.info(f"      altitude_origin: {obs['crs']['altitude_origin']}\n")


def _default_setup():
    """set up from configuration file(s)

    read parameters from
    ~/.config/geec/config.yaml
    otherwise from
    /path/to/package/cfg/config_default.yaml
    """
    # set up configuration file
    try:
        # Read configuration file
        config = confuse.LazyConfig(
            "geec", modname=geec.__pkg_cfg__
        )  # Get a value from your YAML file

        # TODO check use of templates,
        #  cf examples in
        # https://github.com/beetbox/confuse/tree/c244db70c6c2e92b001ce02951cf60e1c8793f75

        # read configuration file path
        # First load default configuration file (lowest priority):
        #  geec/cfg/config_default.yaml
        # Second (higher priority) load local user configuration file:
        #  ~/.config/geec/config.yaml
        config.read()

    except Exception:
        logger.error("Something goes wrong when loading config file.")
        raise  # Throw exception again so calling code knows it happened
    else:
        return config


def setup(user_cfg=None):
    """set up from configuration file(s)

    read parameters from
    ~/.config/geec/config.yaml
    otherwise from
    /path/to/package/cfg/config_default.yaml
    otherwise from
    user_cfg configuration file (given in argument)
    """
    # set up configuration file
    config = _default_setup()
    if user_cfg:
        # Load user configuration given as arguments (highest priority)
        config.set_file(user_cfg)

    return config


def main(user_cfg=None):
    """set up from configuration file(s)

    read parameters from
    ~/.config/geec/config.yaml
    otherwise from
    /path/to/package/cfg/config_default.yaml
    otherwise from
    user_cfg configuration file (given in argument)
    """
    # set up configuration file
    config = setup(user_cfg)
    show(config)


if __name__ == "__main__":
    main()
