"""Geospatial extensions for mlcroissant.

This module provides functionality for working with geospatial datasets,
including converters for STAC catalogs to GeoCroissant format.
"""

from .nasa_umm_g_converter import umm_g_to_geocroissant
from .stac_converters import stac_to_geocroissant

__all__ = ["stac_to_geocroissant", "umm_g_to_geocroissant"]
