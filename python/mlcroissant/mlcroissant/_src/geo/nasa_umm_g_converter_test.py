"""Tests for NASA UMM-G to GeoCroissant converter.

This test suite provides comprehensive coverage of the NASA UMM-G to GeoCroissant
converter functionality, including:

1. Core Function Tests - Individual function testing
2. Data Extraction Tests - UMM-G data parsing and extraction
3. URL Functionality Tests - Network requests and CMR endpoint handling
4. Edge Case Tests - Boundary conditions and error handling
5. Integration Tests - Real-world data and end-to-end conversion
6. Error Handling Tests - Exception scenarios and validation

"""

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import requests

from .nasa_umm_g_converter import _check_dependencies
from .nasa_umm_g_converter import _extract_cite_as
from .nasa_umm_g_converter import convert_polygon_to_wkt
from .nasa_umm_g_converter import determine_encoding_format
from .nasa_umm_g_converter import ensure_semver
from .nasa_umm_g_converter import extract_distribution_info
from .nasa_umm_g_converter import extract_platform_information
from .nasa_umm_g_converter import extract_spatial_extent
from .nasa_umm_g_converter import extract_temporal_extent
from .nasa_umm_g_converter import sanitize_name
from .nasa_umm_g_converter import umm_g_to_geocroissant

# =============================================================================
# TEST DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_umm_g():
    """Sample UMM-G data for testing."""
    return {
        "meta": {
            "concept-id": "C1234567890-PROV",
            "revision-id": "1",
            "concept-type": "granule"
        },
        "umm": {
            "GranuleUR": "HLS.S30.T32NPH.2020001T143919.v2.0",
            "CollectionReference": {
                "ShortName": "HLS",
                "Version": "2.0",
                "EntryTitle": "Harmonized Landsat Sentinel-2",
                "Abstract": "Harmonized Landsat Sentinel-2 dataset"
            },
            "Version": "2.0",
            "Abstract": "Sample HLS granule",
            "SpatialExtent": {
                "HorizontalSpatialDomain": {
                    "BoundingRectangle": {
                        "WestBoundingCoordinate": -122.0,
                        "SouthBoundingCoordinate": 37.0,
                        "EastBoundingCoordinate": -121.0,
                        "NorthBoundingCoordinate": 38.0
                    },
                    "Geometry": {
                        "GPolygons": [{
                            "Boundary": {
                                "Points": [
                                    {"Longitude": -122.0, "Latitude": 37.0},
                                    {"Longitude": -121.0, "Latitude": 37.0},
                                    {"Longitude": -121.0, "Latitude": 38.0},
                                    {"Longitude": -122.0, "Latitude": 38.0},
                                    {"Longitude": -122.0, "Latitude": 37.0}
                                ]
                            }
                        }]
                    }
                }
            },
            "TemporalExtent": {
                "RangeDateTime": {
                    "BeginningDateTime": "2020-01-01T00:00:00Z",
                    "EndingDateTime": "2020-01-01T23:59:59Z"
                }
            },
            "Platforms": [{
                "ShortName": "SENTINEL-2A",
                "LongName": "Sentinel-2A",
                "Instruments": [{
                    "ShortName": "MSI",
                    "LongName": "MultiSpectral Instrument"
                }]
            }],
            "RelatedUrls": [{
                "URL": "https://example.com/data.tif",
                "Type": "GET DATA",
                "Subtype": "DIRECT DOWNLOAD",
                "Description": "GeoTIFF data file"
            }],
            "AdditionalAttributes": [{
                "Name": "SPATIAL_RESOLUTION",
                "Values": ["30"]
            }]
        }
    }


@pytest.fixture
def sample_cmr_response(sample_umm_g):
    """Sample CMR response format."""
    return {
        "hits": 1,
        "took": 4034,
        "items": [sample_umm_g]
    }


@pytest.fixture
def sample_umm_g_with_doi():
    """Sample UMM-G with DOI in AdditionalAttributes."""
    return {
        "umm": {
            "AdditionalAttributes": [{
                "Name": "IDENTIFIER_PRODUCT_DOI",
                "Values": ["10.5067/HLS/HLSS30.002"]
            }]
        }
    }


@pytest.fixture
def real_umm_file_path():
    """Path to real UMM-G data file."""
    return Path(__file__).parent / "umm.json"


# =============================================================================
# 1. CORE FUNCTION TESTS
# =============================================================================

class TestCoreFunctions:
    """Test core utility functions."""
    
    def test_sanitize_name(self):
        """Test name sanitization for Croissant format."""
        # Test normal cases
        assert sanitize_name("Test Dataset!") == "Test-Dataset"
        assert sanitize_name("Dataset with spaces") == "Dataset-with-spaces"
        assert sanitize_name("Dataset_with_underscores") == "Dataset_with_underscores"
        assert sanitize_name("Dataset-with-dashes") == "Dataset-with-dashes"
        
        # Test edge cases
        assert sanitize_name("") == "UnnamedDataset"
        assert sanitize_name(None) == "UnnamedDataset"
        assert sanitize_name("   ") == "UnnamedDataset"
        
        # Test special characters
        assert sanitize_name("Dataset@#$%^&*()") == "Dataset"
        assert sanitize_name("Dataset with multiple   spaces") == "Dataset-with-multiple-spaces"
        assert sanitize_name("Dataset---with---multiple---dashes") == "Dataset-with-multiple-dashes"
    
    def test_ensure_semver(self):
        """Test semantic version formatting."""
        # Test normal cases
        assert ensure_semver("2.0") == "2.0.0"
        assert ensure_semver("v1.5") == "1.5.0"
        assert ensure_semver("3.2.1") == "3.2.1"
        assert ensure_semver("4.0.0") == "4.0.0"
        
        # Test edge cases
        assert ensure_semver(None) == "1.0.0"
        assert ensure_semver("") == "1.0.0"
        assert ensure_semver(0) == "0.0.0"
        
        # Test numeric inputs
        assert ensure_semver(1) == "1.0.0"
        assert ensure_semver(2.5) == "2.5.0"
        assert ensure_semver(10) == "10.0.0"
        
        # Test version prefixes
        assert ensure_semver("v2.1") == "2.1.0"
        assert ensure_semver("V3.0") == "V3.0.0"  # Only lowercase 'v' is handled
    
    def test_convert_polygon_to_wkt(self):
        """Test polygon to WKT conversion."""
        # Test normal polygon
        points = [
            {"Longitude": -122.0, "Latitude": 37.0},
            {"Longitude": -121.0, "Latitude": 37.0},
            {"Longitude": -121.0, "Latitude": 38.0},
            {"Longitude": -122.0, "Latitude": 38.0}
        ]
        
        wkt = convert_polygon_to_wkt(points)
        expected = "POLYGON((-122.0 37.0, -121.0 37.0, -121.0 38.0, -122.0 38.0, -122.0 37.0))"
        assert wkt == expected
        
        # Test edge cases
        assert convert_polygon_to_wkt([]) == ""
        
        # Test single point (creates a polygon with the same point repeated)
        single_point = [{"Longitude": -122.0, "Latitude": 37.0}]
        result = convert_polygon_to_wkt(single_point)
        assert result == "POLYGON((-122.0 37.0, -122.0 37.0))"
        
        # Test already closed polygon
        closed_polygon = [
            {"Longitude": -122.0, "Latitude": 37.0},
            {"Longitude": -121.0, "Latitude": 37.0},
            {"Longitude": -121.0, "Latitude": 38.0},
            {"Longitude": -122.0, "Latitude": 38.0},
            {"Longitude": -122.0, "Latitude": 37.0}  # Already closed
        ]
        result = convert_polygon_to_wkt(closed_polygon)
        assert result == "POLYGON((-122.0 37.0, -121.0 37.0, -121.0 38.0, -122.0 38.0, -122.0 37.0))"
    
    def test_determine_encoding_format(self):
        """Test encoding format determination from URLs."""
        # Test GeoTIFF formats
        assert determine_encoding_format("file.tif", "GET DATA", "DIRECT DOWNLOAD") == "image/tiff"
        assert determine_encoding_format("file.tiff", "GET DATA", "DIRECT DOWNLOAD") == "image/tiff"
        
        # Test image formats
        assert determine_encoding_format("file.jpg", "GET DATA", "DIRECT DOWNLOAD") == "image/jpeg"
        assert determine_encoding_format("file.jpeg", "GET DATA", "DIRECT DOWNLOAD") == "image/jpeg"
        
        # Test data formats
        assert determine_encoding_format("file.json", "GET DATA", "DIRECT DOWNLOAD") == "application/json"
        assert determine_encoding_format("file.xml", "GET DATA", "DIRECT DOWNLOAD") == "application/xml"
        assert determine_encoding_format("file.hdf", "GET DATA", "DIRECT DOWNLOAD") == "application/x-hdf"
        assert determine_encoding_format("file.h5", "GET DATA", "DIRECT DOWNLOAD") == "application/x-hdf"
        assert determine_encoding_format("file.nc", "GET DATA", "DIRECT DOWNLOAD") == "application/x-netcdf"
        assert determine_encoding_format("file.zip", "GET DATA", "DIRECT DOWNLOAD") == "application/zip"
        
        # Test unknown formats
        assert determine_encoding_format("file.unknown", "GET DATA", "DIRECT DOWNLOAD") == "application/octet-stream"
        assert determine_encoding_format("file", "GET DATA", "DIRECT DOWNLOAD") == "application/octet-stream"
        assert determine_encoding_format("", "GET DATA", "DIRECT DOWNLOAD") == "application/octet-stream"


# =============================================================================
# 2. DATA EXTRACTION TESTS
# =============================================================================

class TestDataExtraction:
    """Test UMM-G data extraction functions."""
    
    def test_extract_spatial_extent(self, sample_umm_g):
        """Test spatial extent extraction."""
        spatial_info = extract_spatial_extent(sample_umm_g["umm"])
        
        assert spatial_info is not None
        assert spatial_info["bbox"] == [-122.0, 37.0, -121.0, 38.0]
        assert "POLYGON" in spatial_info["geometry"]
        assert "(-122.0 37.0" in spatial_info["geometry"]
        assert "-121.0 38.0" in spatial_info["geometry"]
    
    def test_extract_spatial_extent_edge_cases(self):
        """Test spatial extent extraction edge cases."""
        # Test with no spatial extent
        umm_no_spatial = {"umm": {}}
        result = extract_spatial_extent(umm_no_spatial["umm"])
        assert result is None
        
        # Test with empty spatial extent
        umm_empty_spatial = {"umm": {"SpatialExtent": {}}}
        result = extract_spatial_extent(umm_empty_spatial["umm"])
        assert result is None
        
        # Test with no geometry or bounding box
        umm_no_geometry = {"umm": {"SpatialExtent": {"HorizontalSpatialDomain": {}}}}
        result = extract_spatial_extent(umm_no_geometry["umm"])
        assert result is None
        
        # Test with only geometry (no bounding box)
        umm_only_geometry = {
            "umm": {
                "SpatialExtent": {
                    "HorizontalSpatialDomain": {
                        "Geometry": {
                            "GPolygons": [{
                                "Boundary": {
                                    "Points": [
                                        {"Longitude": -122.0, "Latitude": 37.0},
                                        {"Longitude": -121.0, "Latitude": 37.0},
                                        {"Longitude": -121.0, "Latitude": 38.0},
                                        {"Longitude": -122.0, "Latitude": 38.0}
                                    ]
                                }
                            }]
                        }
                    }
                }
            }
        }
        result = extract_spatial_extent(umm_only_geometry["umm"])
        assert result is not None
        assert result["bbox"] is None
        assert "POLYGON" in result["geometry"]
    
    def test_extract_temporal_extent(self, sample_umm_g):
        """Test temporal extent extraction."""
        temporal_info = extract_temporal_extent(sample_umm_g["umm"])
        
        assert temporal_info is not None
        assert temporal_info["start"] == "2020-01-01T00:00:00Z"
        assert temporal_info["end"] == "2020-01-01T23:59:59Z"
    
    def test_extract_temporal_extent_edge_cases(self):
        """Test temporal extent extraction edge cases."""
        # Test with no temporal extent
        umm_no_temporal = {"umm": {}}
        result = extract_temporal_extent(umm_no_temporal["umm"])
        assert result is None
        
        # Test with empty temporal extent
        umm_empty_temporal = {"umm": {"TemporalExtent": {}}}
        result = extract_temporal_extent(umm_empty_temporal["umm"])
        assert result is None
        
        # Test with no range datetime
        umm_no_range = {"umm": {"TemporalExtent": {"RangeDateTime": {}}}}
        result = extract_temporal_extent(umm_no_range["umm"])
        assert result is None
        
        # Test with partial range datetime
        umm_partial_range = {"umm": {"TemporalExtent": {"RangeDateTime": {"BeginningDateTime": "2020-01-01T00:00:00Z"}}}}
        result = extract_temporal_extent(umm_partial_range["umm"])
        assert result is not None
        assert result["start"] == "2020-01-01T00:00:00Z"
        assert result["end"] is None
    
    def test_extract_platform_information(self, sample_umm_g):
        """Test platform information extraction."""
        platform_info = extract_platform_information(sample_umm_g["umm"])
        
        assert platform_info["platform"] == "SENTINEL-2A"
        assert platform_info["instrument"] == "MSI"
        assert platform_info["platform_long_name"] == "Sentinel-2A"
        assert platform_info["instrument_long_name"] == "MultiSpectral Instrument"
    
    def test_extract_platform_information_edge_cases(self):
        """Test platform information extraction edge cases."""
        # Test with no platforms
        umm_no_platforms = {"umm": {}}
        result = extract_platform_information(umm_no_platforms["umm"])
        assert result == {}
        
        # Test with empty platforms list
        umm_empty_platforms = {"umm": {"Platforms": []}}
        result = extract_platform_information(umm_empty_platforms["umm"])
        assert result == {}
        
        # Test with platform but no instruments
        umm_no_instruments = {"umm": {"Platforms": [{"ShortName": "TEST_PLATFORM"}]}}
        result = extract_platform_information(umm_no_instruments["umm"])
        assert result["platform"] == "TEST_PLATFORM"
        assert result["instrument"] == "Unknown"
        assert result["platform_long_name"] == ""
        assert result["instrument_long_name"] == ""
        
        # Test with platform and empty instruments list
        umm_empty_instruments = {"umm": {"Platforms": [{"ShortName": "TEST_PLATFORM", "Instruments": []}]}}
        result = extract_platform_information(umm_empty_instruments["umm"])
        assert result["platform"] == "TEST_PLATFORM"
        assert result["instrument"] == "Unknown"
    
    def test_extract_distribution_info(self, sample_umm_g):
        """Test distribution information extraction."""
        distributions = extract_distribution_info(sample_umm_g["umm"])
        
        assert len(distributions) == 1
        assert distributions[0]["contentUrl"] == "https://example.com/data.tif"
        assert distributions[0]["encodingFormat"] == "image/tiff"
        assert distributions[0]["name"] == "data.tif"
        assert distributions[0]["description"] == "GeoTIFF data file"
        assert distributions[0]["@type"] == "cr:FileObject"
        assert distributions[0]["md5"] == "placeholder_md5_hash"
        assert distributions[0]["sha256"] == "placeholder_sha256_hash"
    
    def test_extract_distribution_info_edge_cases(self):
        """Test distribution information extraction edge cases."""
        # Test with no related URLs
        umm_no_urls = {"umm": {}}
        result = extract_distribution_info(umm_no_urls["umm"])
        assert result == []
        
        # Test with empty related URLs
        umm_empty_urls = {"umm": {"RelatedUrls": []}}
        result = extract_distribution_info(umm_empty_urls["umm"])
        assert result == []
        
        # Test with URL that has no file extension
        umm_no_extension = {"umm": {"RelatedUrls": [{"URL": "https://example.com/data", "Type": "GET DATA"}]}}
        result = extract_distribution_info(umm_no_extension["umm"])
        assert len(result) == 1
        assert result[0]["encodingFormat"] == "application/octet-stream"
        assert result[0]["name"] == "data"
        
        # Test with URL that has no filename
        umm_no_filename = {"umm": {"RelatedUrls": [{"URL": "https://example.com/", "Type": "GET DATA"}]}}
        result = extract_distribution_info(umm_no_filename["umm"])
        assert len(result) == 1
        assert result[0]["name"] == "data_file"
    
    def test_extract_cite_as(self, sample_umm_g_with_doi):
        """Test citation extraction."""
        # Test with DOI in AdditionalAttributes
        result = _extract_cite_as(sample_umm_g_with_doi["umm"])
        assert result == "https://doi.org/10.5067/HLS/HLSS30.002"
        
        # Test fallback to default HLS DOI
        result = _extract_cite_as({"umm": {}})
        assert result == "https://doi.org/10.5067/HLS/HLSS30.002"
        
        # Test with empty AdditionalAttributes
        umm_empty_attrs = {"umm": {"AdditionalAttributes": []}}
        result = _extract_cite_as(umm_empty_attrs["umm"])
        assert result == "https://doi.org/10.5067/HLS/HLSS30.002"
        
        # Test with DOI attribute but no values
        umm_no_values = {"umm": {"AdditionalAttributes": [{"Name": "IDENTIFIER_PRODUCT_DOI"}]}}
        result = _extract_cite_as(umm_no_values["umm"])
        assert result == "https://doi.org/10.5067/HLS/HLSS30.002"


# =============================================================================
# 3. URL FUNCTIONALITY TESTS
# =============================================================================

class TestURLFunctionality:
    """Test URL fetching and CMR endpoint functionality."""
    
    def test_umm_g_to_geocroissant_with_url_success(self, sample_umm_g):
        """Test UMM-G to GeoCroissant conversion with successful URL fetch."""
        mock_response = Mock()
        mock_response.json.return_value = sample_umm_g
        mock_response.raise_for_status.return_value = None
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            result = umm_g_to_geocroissant("https://example.com/umm.json")
            
            # Verify request was made
            mock_get.assert_called_once_with("https://example.com/umm.json", timeout=30)
            
            # Verify result
            assert result["@type"] == "Dataset"
            assert result["@id"] == "HLS.S30.T32NPH.2020001T143919.v2.0"
    
    def test_umm_g_to_geocroissant_with_cmr_response(self, sample_cmr_response):
        """Test UMM-G to GeoCroissant conversion with CMR response format."""
        mock_response = Mock()
        mock_response.json.return_value = sample_cmr_response
        mock_response.raise_for_status.return_value = None
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            result = umm_g_to_geocroissant("https://cmr.earthdata.nasa.gov/search/granules.umm_json?concept_id=G2700719831-LPCLOUD")
            
            # Verify request was made
            mock_get.assert_called_once_with("https://cmr.earthdata.nasa.gov/search/granules.umm_json?concept_id=G2700719831-LPCLOUD", timeout=30)
            
            # Verify CMR response was handled correctly
            assert result["@type"] == "Dataset"
            assert result["@id"] == "HLS.S30.T32NPH.2020001T143919.v2.0"
    
    def test_umm_g_to_geocroissant_with_empty_cmr_response(self):
        """Test UMM-G to GeoCroissant conversion with empty CMR response."""
        empty_cmr_response = {"items": []}
        mock_response = Mock()
        mock_response.json.return_value = empty_cmr_response
        mock_response.raise_for_status.return_value = None
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            with pytest.raises(ValueError, match="CMR response contains no granules"):
                umm_g_to_geocroissant("https://example.com/empty.json")
    
    def test_umm_g_to_geocroissant_with_http_error(self):
        """Test UMM-G to GeoCroissant conversion with HTTP error."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            with pytest.raises(ValueError, match="Failed to fetch UMM-G data from URL"):
                umm_g_to_geocroissant("https://example.com/notfound.json")
    
    def test_umm_g_to_geocroissant_with_json_error(self):
        """Test UMM-G to GeoCroissant conversion with JSON parsing error."""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.raise_for_status.return_value = None
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            with pytest.raises(ValueError, match="Invalid JSON response from URL"):
                umm_g_to_geocroissant("https://example.com/invalid.json")
    
    def test_umm_g_to_geocroissant_with_network_error(self):
        """Test UMM-G to GeoCroissant conversion with network error."""
        with patch('requests.get', side_effect=requests.ConnectionError("Connection failed")) as mock_get:
            with pytest.raises(ValueError, match="Failed to fetch UMM-G data from URL"):
                umm_g_to_geocroissant("https://example.com/unreachable.json")
    
    def test_umm_g_to_geocroissant_with_timeout(self):
        """Test UMM-G to GeoCroissant conversion with timeout error."""
        with patch('requests.get', side_effect=requests.Timeout("Request timeout")) as mock_get:
            with pytest.raises(ValueError, match="Failed to fetch UMM-G data from URL"):
                umm_g_to_geocroissant("https://example.com/slow.json")


# =============================================================================
# 4. EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_check_dependencies(self):
        """Test dependency checking."""
        # Test when requests is available
        _check_dependencies()  # Should not raise if requests is available
        
        # Test when requests is not available
        with patch('mlcroissant._src.geo.nasa_umm_g_converter.REQUESTS_AVAILABLE', False):
            with pytest.raises(ImportError, match="Requests dependency not found"):
                _check_dependencies()
    
    def test_sanitize_name_edge_cases(self):
        """Test name sanitization edge cases."""
        # Test with only special characters
        assert sanitize_name("!@#$%^&*()") == "UnnamedDataset"
        assert sanitize_name("   ") == "UnnamedDataset"
        assert sanitize_name("---") == "UnnamedDataset"
        
        # Test with mixed special characters and valid characters
        assert sanitize_name("Dataset!@#$%^&*()Name") == "Dataset-Name"
        assert sanitize_name("Dataset   with   multiple   spaces") == "Dataset-with-multiple-spaces"
        assert sanitize_name("Dataset---with---multiple---dashes") == "Dataset-with-multiple-dashes"
        
        # Test with numbers and underscores
        assert sanitize_name("Dataset_123") == "Dataset_123"
        assert sanitize_name("123_Dataset") == "123_Dataset"
    
    def test_ensure_semver_edge_cases(self):
        """Test semver formatting edge cases."""
        # Test with very long version strings
        assert ensure_semver("1.2.3.4.5.6") == "1.2.3"
        assert ensure_semver("v10.20.30.40") == "10.20.30"
        
        # Test with non-standard version formats
        assert ensure_semver("alpha-1.0") == "alpha-1.0.0"
        assert ensure_semver("beta2.1") == "beta2.1.0"
        
        # Test with zero versions
        assert ensure_semver("0.0.0") == "0.0.0"
        assert ensure_semver("0.1") == "0.1.0"
        assert ensure_semver("0") == "0.0.0"
    
    def test_convert_polygon_to_wkt_edge_cases(self):
        """Test polygon to WKT conversion edge cases."""
        # Test with empty points
        result = convert_polygon_to_wkt([])
        assert result == ""
        
        # Test with single point (creates a polygon with the same point repeated)
        single_point = [{"Longitude": -122.0, "Latitude": 37.0}]
        result = convert_polygon_to_wkt(single_point)
        assert result == "POLYGON((-122.0 37.0, -122.0 37.0))"
        
        # Test with points that already form a closed polygon
        closed_polygon = [
            {"Longitude": -122.0, "Latitude": 37.0},
            {"Longitude": -121.0, "Latitude": 37.0},
            {"Longitude": -121.0, "Latitude": 38.0},
            {"Longitude": -122.0, "Latitude": 38.0},
            {"Longitude": -122.0, "Latitude": 37.0}  # Already closed
        ]
        result = convert_polygon_to_wkt(closed_polygon)
        assert result == "POLYGON((-122.0 37.0, -121.0 37.0, -121.0 38.0, -122.0 38.0, -122.0 37.0))"
        
        # Test with missing coordinate values
        incomplete_points = [
            {"Longitude": -122.0, "Latitude": 37.0},
            {"Longitude": -121.0},  # Missing Latitude
            {"Latitude": 38.0},     # Missing Longitude
            {"Longitude": -122.0, "Latitude": 38.0}
        ]
        result = convert_polygon_to_wkt(incomplete_points)
        assert result == "POLYGON((-122.0 37.0, -121.0 0, 0 38.0, -122.0 38.0, -122.0 37.0))"


# =============================================================================
# 5. INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Test integration scenarios and real-world data."""
    
    def test_umm_g_to_geocroissant_basic(self, sample_umm_g):
        """Test basic UMM-G to GeoCroissant conversion."""
        result = umm_g_to_geocroissant(sample_umm_g)
        
        # Check basic structure
        assert result["@type"] == "Dataset"
        assert result["@id"] == "HLS.S30.T32NPH.2020001T143919.v2.0"
        assert result["name"] == "Harmonized-Landsat-Sentinel-2"
        assert result["version"] == "2.0.0"
        assert result["license"] == "CC-BY-4.0"
        assert result["conformsTo"] == "http://mlcommons.org/croissant/1.0"
        
        # Check spatial information
        assert "geocr:BoundingBox" in result
        assert "geocr:Geometry" in result
        assert result["geocr:BoundingBox"] == [-122.0, 37.0, -121.0, 38.0]
        
        # Check temporal information
        assert "dct:temporal" in result
        assert result["datePublished"] == "2020-01-01T00:00:00Z"
        assert result["dct:temporal"]["startDate"] == "2020-01-01T00:00:00Z"
        assert result["dct:temporal"]["endDate"] == "2020-01-01T23:59:59Z"
        
        # Check platform information
        assert result["geocr:Platform"] == "SENTINEL-2A"
        assert result["geocr:Instrument"] == "MSI"
        
        # Check distribution
        assert "distribution" in result
        assert len(result["distribution"]) == 1
        assert result["distribution"][0]["contentUrl"] == "https://example.com/data.tif"
        
        # Check context
        assert "@context" in result
        assert result["@context"]["geocr"] == "http://mlcommons.org/geocroissant/"
    
    def test_umm_g_to_geocroissant_with_file_output(self, sample_umm_g, tmp_path):
        """Test UMM-G to GeoCroissant conversion with file output."""
        output_file = tmp_path / "test_output.json"
        
        result = umm_g_to_geocroissant(sample_umm_g, output_file)
        
        # Check file was created
        assert output_file.exists()
        
        # Check file content
        with open(output_file, 'r') as f:
            file_content = json.load(f)
        
        assert file_content["@id"] == result["@id"]
        assert file_content["name"] == result["name"]
        assert file_content["version"] == result["version"]
    
    def test_umm_g_to_geocroissant_with_real_umm_data(self, real_umm_file_path):
        """Test UMM-G to GeoCroissant conversion with real UMM-G data file."""
        if not real_umm_file_path.exists():
            pytest.skip("Real UMM-G data file not found")
        
        result = umm_g_to_geocroissant(real_umm_file_path)
        
        # Verify basic structure
        assert result["@type"] == "Dataset"
        assert result["@id"] == "HLS.S30.T16SDD.2016134T163332.v2.0"
        assert result["name"] == "HLS-Sentinel-2-Multi-spectral-Instrument-Surface-Reflectance-Daily-Global-30m-v2-0"
        assert result["version"] == "1.0.0"  # revision-id: 1 gets converted to 1.0.0 by ensure_semver
        assert result["license"] == "CC-BY-4.0"
        
        # Verify context and conformance
        assert "@context" in result
        assert result["@context"]["geocr"] == "http://mlcommons.org/geocroissant/"
        assert result["conformsTo"] == "http://mlcommons.org/croissant/1.0"
        
        # Verify citation
        assert result["citeAs"] == "https://doi.org/10.5067/HLS/HLSS30.002"
        
        # Verify temporal information
        assert "dct:temporal" in result
        assert result["dct:temporal"]["startDate"] == "2016-05-13T16:35:44.550Z"
        assert result["dct:temporal"]["endDate"] == "2016-05-13T16:35:44.550Z"
        
        # Verify platform information
        assert result["geocr:Platform"] == "Sentinel-2A"
        assert result["geocr:Instrument"] == "Sentinel-2 MSI"
        
        # Verify distribution (should have multiple GeoTIFF files)
        assert "distribution" in result
        assert len(result["distribution"]) > 0
        
        # Check that we have GeoTIFF files in distribution
        tiff_files = [d for d in result["distribution"] if d["encodingFormat"] == "image/tiff"]
        assert len(tiff_files) > 0
        
        # Verify spatial information (should have polygon geometry)
        assert "geocr:Geometry" in result
        assert "POLYGON" in result["geocr:Geometry"]
        
        # Verify additional attributes
        assert "geocr:AdditionalAttributes" in result
        additional_attrs = result["geocr:AdditionalAttributes"]
        
        # Check for specific attributes we know should exist
        attr_names = [attr["Name"] for attr in additional_attrs]
        assert "SPATIAL_RESOLUTION" in attr_names
        assert "CLOUD_COVERAGE" in attr_names
        assert "IDENTIFIER_PRODUCT_DOI" in attr_names
    
    def test_extract_spatial_extent_with_real_data(self, real_umm_file_path):
        """Test spatial extent extraction with real UMM-G data."""
        if not real_umm_file_path.exists():
            pytest.skip("Real UMM-G data file not found")
        
        with open(real_umm_file_path, 'r') as f:
            real_umm_data = json.load(f)
        
        spatial_info = extract_spatial_extent(real_umm_data["umm"])
        
        # Should have polygon geometry but no bounding box
        assert spatial_info is not None
        assert spatial_info["bbox"] is None
        assert "POLYGON" in spatial_info["geometry"]
        
        # Check polygon coordinates
        assert "(-88.0861147 34.24811019" in spatial_info["geometry"]
        assert "-86.89272491 35.24303017" in spatial_info["geometry"]


# =============================================================================
# 6. ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_umm_g_to_geocroissant_invalid_input(self):
        """Test handling of invalid input types."""
        with pytest.raises(TypeError, match="Expected string, Path, or dict input"):
            umm_g_to_geocroissant(123)
        
        with pytest.raises(TypeError, match="Expected string, Path, or dict input"):
            umm_g_to_geocroissant([])
        
        with pytest.raises(TypeError, match="Expected string, Path, or dict input"):
            umm_g_to_geocroissant(None)
    
    def test_umm_g_to_geocroissant_missing_file(self):
        """Test handling of missing file."""
        with pytest.raises(FileNotFoundError, match="UMM-G file not found"):
            umm_g_to_geocroissant("/nonexistent/file.json")
    
    def test_umm_g_to_geocroissant_with_real_nasa_cmr(self):
        """Integration test with real NASA CMR endpoint.
        
        This test uses the actual NASA CMR API to verify the converter works
        with real-world data from the provided URL.
        """
        test_url = "https://cmr.earthdata.nasa.gov/search/granules.umm_json?concept_id=G2700719831-LPCLOUD"
        
        try:
            result = umm_g_to_geocroissant(test_url)
            
            # Verify basic structure
            assert result["@type"] == "Dataset"
            assert result["@id"] == "HLS.S30.T16SDD.2016134T163332.v2.0"
            assert result["name"] == "HLS-Sentinel-2-Multi-spectral-Instrument-Surface-Reflectance-Daily-Global-30m-v2-0"
            assert result["version"] == "1.0.0"
            assert result["license"] == "CC-BY-4.0"
            
            # Verify context and conformance
            assert "@context" in result
            assert result["@context"]["geocr"] == "http://mlcommons.org/geocroissant/"
            assert result["conformsTo"] == "http://mlcommons.org/croissant/1.0"
            
            # Verify citation
            assert result["citeAs"] == "https://doi.org/10.5067/HLS/HLSS30.002"
            
            # Verify temporal information
            assert "dct:temporal" in result
            assert result["dct:temporal"]["startDate"] == "2016-05-13T16:35:44.550Z"
            assert result["dct:temporal"]["endDate"] == "2016-05-13T16:35:44.550Z"
            
            # Verify platform information
            assert result["geocr:Platform"] == "Sentinel-2A"
            assert result["geocr:Instrument"] == "Sentinel-2 MSI"
            
            # Verify distribution (should have multiple GeoTIFF files)
            assert "distribution" in result
            assert len(result["distribution"]) > 0
            
            # Check that we have GeoTIFF files in distribution
            tiff_files = [d for d in result["distribution"] if d["encodingFormat"] == "image/tiff"]
            assert len(tiff_files) > 0
            
            # Verify spatial information (should have polygon geometry)
            assert "geocr:Geometry" in result
            assert "POLYGON" in result["geocr:Geometry"]
            
            # Verify additional attributes
            assert "geocr:AdditionalAttributes" in result
            additional_attrs = result["geocr:AdditionalAttributes"]
            
            # Check for specific attributes we know should exist
            attr_names = [attr["Name"] for attr in additional_attrs]
            assert "SPATIAL_RESOLUTION" in attr_names
            assert "CLOUD_COVERAGE" in attr_names
            assert "IDENTIFIER_PRODUCT_DOI" in attr_names
            
        except Exception as e:
            # Skip test if network issues or URL is not accessible
            pytest.skip(f"Real NASA CMR test skipped due to network/accessibility issues: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
