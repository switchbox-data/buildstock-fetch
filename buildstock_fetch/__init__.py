"""
BuildStock Fetch - A Python library for downloading building energy simulation data from NREL's ResStock and ComStock projects.
"""

__version__ = "0.0.1"
__author__ = "Switchbox"
__email__ = "bryan@switch.box"

# Import main functions for easier access
from .main import fetch_bldg_ids, fetch_bldg_data

__all__ = ["fetch_bldg_ids", "fetch_bldg_data"]
