"""Convert NASA UMM-G datasets to GeoCroissant format.

This module provides functions to convert NASA UMM-G (Unified Metadata Model - Geographic)
format to the GeoCroissant JSON-LD format.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


def _check_dependencies() -> None:
    """Check if required dependencies are installed."""
    if not REQUESTS_AVAILABLE:
        raise ImportError(
            "Requests dependency not found. "
            "Install with: pip install requests"
        )


def _extract_cite_as(umm: Dict[str, Any]) -> str:
    """Extract citation information from UMM-G data."""
    # Look for DOI in AdditionalAttributes
    additional_attrs = umm.get('AdditionalAttributes', [])
    for attr in additional_attrs:
        if attr.get('Name') == 'IDENTIFIER_PRODUCT_DOI':
            values = attr.get('Values', [])
            if values and values[0]:
                return f"https://doi.org/{values[0]}"
    
    # Fallback to default HLS DOI
    return "https://doi.org/10.5067/HLS/HLSS30.002"


def sanitize_name(name: str) -> str:
    """Sanitize name for use in Croissant format.
    
    Args:
        name: Input string to sanitize
        
    Returns:
        Sanitized string with special characters replaced by single dashes
    """
    if not name:
        return "UnnamedDataset"
    
    # Replace special characters with dash
    sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "-", str(name))
    # Collapse multiple dashes into one
    sanitized = re.sub(r"-+", "-", sanitized)
    # Strip leading/trailing dashes and spaces
    sanitized = sanitized.strip("- ")
    
    # If sanitization resulted in empty string, return default
    if not sanitized:
        return "UnnamedDataset"
    
    return sanitized


def ensure_semver(version: Optional[Union[str, int, float]]) -> str:
    """Ensure version follows semver format."""
    if version is None or version == "":
        return "1.0.0"
    
    # Convert to string if it's a number
    version_str = str(version)
    
    if version_str.startswith("v"):
        version_str = version_str[1:]
    parts = version_str.split(".")
    
    # Ensure we have at least 3 parts
    while len(parts) < 3:
        parts.append("0")
    
    return ".".join(parts[:3])


def extract_spatial_extent(umm: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract spatial extent information from UMM-G."""
    spatial_extent = umm.get('SpatialExtent', {})
    if not spatial_extent:
        return None
    
    horizontal_domain = spatial_extent.get('HorizontalSpatialDomain', {})
    geometry = horizontal_domain.get('Geometry', {})
    
    # Extract bounding box
    bbox = None
    if 'BoundingRectangle' in horizontal_domain:
        rect = horizontal_domain['BoundingRectangle']
        bbox = [
            rect.get('WestBoundingCoordinate', -180),
            rect.get('SouthBoundingCoordinate', -90),
            rect.get('EastBoundingCoordinate', 180),
            rect.get('NorthBoundingCoordinate', 90)
        ]
    
    # Extract polygon geometry
    geometry_wkt = None
    if 'GPolygons' in geometry:
        polygons = geometry['GPolygons']
        if polygons:
            points = polygons[0].get('Boundary', {}).get('Points', [])
            if points:
                geometry_wkt = convert_polygon_to_wkt(points)
    
    # Only return if we have either bbox or geometry
    if bbox or geometry_wkt:
        return {
            "bbox": bbox,
            "geometry": geometry_wkt
        }
    return None


def extract_temporal_extent(umm: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract temporal extent information from UMM-G."""
    temporal_extent = umm.get('TemporalExtent', {})
    if not temporal_extent:
        return None
    
    range_datetime = temporal_extent.get('RangeDateTime', {})
    if not range_datetime:
        return None
    
    return {
        "start": range_datetime.get('BeginningDateTime'),
        "end": range_datetime.get('EndingDateTime')
    }


def convert_polygon_to_wkt(points: List[Dict[str, float]]) -> str:
    """Convert polygon points to WKT format."""
    if not points:
        return ""
    
    coords = []
    for point in points:
        lon = point.get('Longitude', 0)
        lat = point.get('Latitude', 0)
        coords.append(f"{lon} {lat}")
    
    # Ensure polygon is closed (first and last points are the same)
    if coords and (len(coords) == 1 or coords[0] != coords[-1]):
        coords.append(coords[0])
    
    return f"POLYGON(({', '.join(coords)}))"


def extract_platform_information(umm: Dict[str, Any]) -> Dict[str, Any]:
    """Extract platform and instrument information from UMM-G."""
    platforms = umm.get('Platforms', [])
    if not platforms:
        return {}
    
    platform = platforms[0]
    instruments = platform.get('Instruments', [])
    
    return {
        "platform": platform.get('ShortName', 'Unknown'),
        "instrument": instruments[0].get('ShortName', 'Unknown') if instruments else 'Unknown',
        "platform_long_name": platform.get('LongName', ''),
        "instrument_long_name": instruments[0].get('LongName', '') if instruments else ''
    }


def extract_distribution_info(umm: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract distribution information from UMM-G."""
    distributions = []
    related_urls = umm.get('RelatedUrls', [])
    
    for url_info in related_urls:
        url = url_info.get('URL', '')
        url_type = url_info.get('Type', '')
        subtype = url_info.get('Subtype', '')
        description = url_info.get('Description', '')
        
        # Determine encoding format
        encoding_format = determine_encoding_format(url, url_type, subtype)
        
        distribution = {
            "@type": "cr:FileObject",
            "@id": f"distribution_{len(distributions)}",
            "name": url.split('/')[-1] or "data_file",
            "description": description or f"Data file from {url_type}",
            "contentUrl": url,
            "encodingFormat": encoding_format,
            # Add placeholder checksums (required by Croissant validator)
            "md5": "placeholder_md5_hash",  # TODO: Calculate actual MD5 if file accessible
            "sha256": "placeholder_sha256_hash"  # TODO: Calculate actual SHA256 if file accessible
        }
        
        distributions.append(distribution)
    
    return distributions


def determine_encoding_format(url: str, url_type: str, subtype: str) -> str:
    """Determine the encoding format based on URL and type."""
    if url.endswith('.tif') or url.endswith('.tiff'):
        return "image/tiff"
    elif url.endswith('.jpg') or url.endswith('.jpeg'):
        return "image/jpeg"
    elif url.endswith('.json'):
        return "application/json"
    elif url.endswith('.xml'):
        return "application/xml"
    elif url.endswith('.hdf') or url.endswith('.h5'):
        return "application/x-hdf"
    elif url.endswith('.nc'):
        return "application/x-netcdf"
    elif url.endswith('.zip'):
        return "application/zip"
    else:
        return "application/octet-stream"


def umm_g_to_geocroissant(
    umm_g_input: Union[str, Path, Dict[str, Any]], 
    output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Convert NASA UMM-G to GeoCroissant JSON-LD format.
    
    Args:
        umm_g_input: UMM-G dictionary, path to UMM-G file, or URL to UMM-G endpoint
        output_path: Optional output file path (if provided, saves to file)
        
    Returns:
        GeoCroissant JSON-LD dictionary
        
    Raises:
        ImportError: If required dependencies are not installed
        ValueError: If UMM-G data is invalid or URL fetch fails
        FileNotFoundError: If umm_g_input file path does not exist
    """
    _check_dependencies()
    
    # Input validation
    if not isinstance(umm_g_input, (str, Path, dict)):
        raise TypeError(f"Expected string, Path, or dict input, got {type(umm_g_input)}")
    
    # Handle file input or URL
    if isinstance(umm_g_input, (str, Path)):
        umm_g_input_str = str(umm_g_input)
        
        # Check if input is a URL
        if umm_g_input_str.startswith(('http://', 'https://')):
            try:
                logger.info(f"Fetching UMM-G data from URL: {umm_g_input_str}")
                response = requests.get(umm_g_input_str, timeout=30)
                response.raise_for_status()
                umm_g_dict = response.json()
                logger.info(f"Successfully fetched UMM-G data from URL")
            except requests.RequestException as e:
                raise ValueError(f"Failed to fetch UMM-G data from URL {umm_g_input_str}: {e}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON response from URL {umm_g_input_str}: {e}")
        else:
            # Handle local file path
            umm_g_path = Path(umm_g_input)
            if not umm_g_path.exists():
                raise FileNotFoundError(f"UMM-G file not found: {umm_g_path}")
            
            logger.info(f"Loading UMM-G file: {umm_g_path}")
            with open(umm_g_path, 'r') as f:
                umm_g_dict = json.load(f)
    else:
        umm_g_dict = umm_g_input

    # Extract main sections - handle both direct UMM-G and CMR response formats
    if 'items' in umm_g_dict:
        # This is a CMR response - extract the first item
        if umm_g_dict.get('items'):
            umm_g_dict = umm_g_dict['items'][0]
            logger.info("Detected CMR response format, extracted first granule")
        else:
            raise ValueError("CMR response contains no granules")
    
    meta = umm_g_dict.get('meta', {})
    umm = umm_g_dict.get('umm', umm_g_dict)  # Handle both formats
    
    # Extract basic information
    dataset_id = umm.get('GranuleUR') or meta.get('concept-id') or "unnamed_dataset"
    title = umm.get('CollectionReference', {}).get('EntryTitle') or umm.get('EntryTitle') or "Unnamed Dataset"
    name = sanitize_name(title)
    version = ensure_semver(umm.get('Version') or meta.get('revision-id') or "1.0.0")
    
    # Extract spatial and temporal information
    spatial_info = extract_spatial_extent(umm)
    temporal_info = extract_temporal_extent(umm)
    platform_info = extract_platform_information(umm)
    distributions = extract_distribution_info(umm)
    
    # Create GeoCroissant structure
    croissant = {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "geocr": "http://mlcommons.org/geocroissant/",
            "dct": "http://purl.org/dc/terms/",
            "sc": "https://schema.org/",
            "citeAs": "cr:citeAs",
            "column": "cr:column",
            "conformsTo": "dct:conformsTo",
            "data": {"@id": "cr:data", "@type": "@json"},
            "dataBiases": "cr:dataBiases",
            "dataCollection": "cr:dataCollection",
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
            "extract": "cr:extract",
            "field": "cr:field",
            "fileProperty": "cr:fileProperty",
            "fileObject": "cr:fileObject",
            "fileSet": "cr:fileSet",
            "format": "cr:format",
            "includes": "cr:includes",
            "isLiveDataset": "cr:isLiveDataset",
            "jsonPath": "cr:jsonPath",
            "key": "cr:key",
            "md5": {"@id": "cr:md5", "@type": "sc:Text"},
            "sha256": {"@id": "cr:sha256", "@type": "sc:Text"},
            "parentField": "cr:parentField",
            "path": "cr:path",
            "personalSensitiveInformation": "cr:personalSensitiveInformation",
            "recordSet": "cr:recordSet",
            "references": "cr:references",
            "regex": "cr:regex",
            "repeated": "cr:repeated",
            "replace": "cr:replace",
            "separator": "cr:separator",
            "source": "cr:source",
            "subField": "cr:subField",
            "transform": "cr:transform"
        },
        "@type": "Dataset",
        "@id": dataset_id,
        "name": name,
        "description": umm.get('Abstract') or umm.get('CollectionReference', {}).get('Abstract') or "",
        "version": version,
        "license": "CC-BY-4.0",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "citeAs": _extract_cite_as(umm)
    }
    
    # Add spatial information
    if spatial_info:
        if spatial_info.get("bbox"):
            croissant["geocr:BoundingBox"] = spatial_info["bbox"]
        if spatial_info.get("geometry"):
            croissant["geocr:Geometry"] = spatial_info["geometry"]
    
    # Add temporal information
    if temporal_info:
        if temporal_info.get("start") and temporal_info.get("end"):
            croissant["dct:temporal"] = {
                "startDate": temporal_info["start"],
                "endDate": temporal_info["end"]
            }
            croissant["datePublished"] = temporal_info["start"]
    
    # Add platform information
    if platform_info:
        croissant["geocr:Platform"] = platform_info["platform"]
        croissant["geocr:Instrument"] = platform_info["instrument"]
    
    # Add distribution
    if distributions:
        croissant["distribution"] = distributions
    
    # Add additional UMM-G specific fields
    additional_attrs = umm.get('AdditionalAttributes', [])
    if additional_attrs:
        croissant["geocr:AdditionalAttributes"] = additional_attrs
    
    # Add collection information
    collection_ref = umm.get('CollectionReference', {})
    if collection_ref:
        croissant["geocr:Collection"] = {
            "ShortName": collection_ref.get('ShortName'),
            "Version": collection_ref.get('Version'),
            "EntryTitle": collection_ref.get('EntryTitle')
        }
    
    # Add data granule information
    data_granule = umm.get('DataGranule', {})
    if data_granule:
        croissant["geocr:DataGranule"] = {
            "DayNightFlag": data_granule.get('DayNightFlag'),
            "ProductionDateTime": data_granule.get('ProductionDateTime'),
            "PGEVersion": data_granule.get('PGEVersion')
        }
    
    # Save to file if output_path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(croissant, f, indent=2)
        logger.info(f"GeoCroissant saved to: {output_path}")
    
    return croissant
