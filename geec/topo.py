"""
"""

# --- import -----------------------------------
# import from standard lib
# import from other lib
from confuse import Subview

# import from my project


def get_topo(config: Subview) -> tuple[bool, float, float]:
    """Create a Topo object from 'topo' subview (from configuration file)"""
    is_topo = config["is_topo"].get(bool)
    extension = config["extension"].get(float)
    water_density = config["water_density"].get(float)

    return (is_topo, extension, water_density)
