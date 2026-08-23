"""GeoCroissant extension modules."""

from rdflib.namespace import SDO

from mlcroissant._src.core import constants
from mlcroissant._src.core import dataclasses as mlc_dataclasses
from mlcroissant._src.structure_graph.base_node import Node


@mlc_dataclasses.dataclass
class QuantitativeValue(Node):
    """Represents schema.org/QuantitativeValue."""

    JSONLD_TYPE = constants.SCHEMA_ORG_QUANTITATIVE_VALUE

    value: float | int | str | None = mlc_dataclasses.jsonld_field(
        default=None,
        description="The value of the quantitative value.",
        input_types=[SDO.Number, SDO.Integer, SDO.Text],
        url=constants.SCHEMA_ORG_VALUE,
    )
    unitText: str | None = mlc_dataclasses.jsonld_field(
        default=None,
        description="A string or text indicating the unit of measurement.",
        input_types=[SDO.Text],
        url=constants.SCHEMA_ORG_UNIT_TEXT,
    )


@mlc_dataclasses.dataclass
class BandConfiguration(Node):
    """Represents geocr:BandConfiguration."""

    JSONLD_TYPE = constants.GEO.BandConfiguration

    total_bands: int | float | None = mlc_dataclasses.jsonld_field(
        default=None,
        description="Total number of bands.",
        input_types=[SDO.Integer, SDO.Number],
        url=constants.ML_COMMONS_GEO_TOTAL_BANDS,
    )
    band_names_list: list[str] | None = mlc_dataclasses.jsonld_field(
        cardinality="MANY",
        default=None,
        description="Ordered list of band names.",
        input_types=[SDO.Text],
        url=constants.ML_COMMONS_GEO_BAND_NAMES_LIST,
    )


@mlc_dataclasses.dataclass
class SpectralBand(Node):
    """Represents geocr:SpectralBand."""

    JSONLD_TYPE = constants.GEO.SpectralBand

    name: str | dict[str, str] | None = mlc_dataclasses.jsonld_field(
        cardinality="LANGUAGE-TAGGED",
        default=None,
        description="The name of the band.",
        input_types=[SDO.Text],
        url=constants.SCHEMA_ORG_NAME,
    )
    center_wavelength: QuantitativeValue | None = mlc_dataclasses.jsonld_field(
        default=None,
        description="Center wavelength for a spectral band entry.",
        input_types=[QuantitativeValue],
        url=constants.ML_COMMONS_GEO_CENTER_WAVELENGTH,
    )
    bandwidth: QuantitativeValue | None = mlc_dataclasses.jsonld_field(
        default=None,
        description="Spectral bandwidth for a spectral band entry.",
        input_types=[QuantitativeValue],
        url=constants.ML_COMMONS_GEO_BANDWIDTH,
    )


@mlc_dataclasses.dataclass
class MultiWavelengthConfiguration(Node):
    """Represents geocr:MultiWavelengthConfiguration."""

    JSONLD_TYPE = constants.GEO.MultiWavelengthConfiguration

    channel_list: list[str] | None = mlc_dataclasses.jsonld_field(
        cardinality="MANY",
        default=None,
        description="List of wavelength channels.",
        input_types=[SDO.Text],
        url=constants.ML_COMMONS_GEO_CHANNEL_LIST,
    )


@mlc_dataclasses.dataclass
class SolarInstrumentCharacteristics(Node):
    """Represents geocr:SolarInstrumentCharacteristics."""

    JSONLD_TYPE = constants.GEO.SolarInstrumentCharacteristics

    observatory: str | None = mlc_dataclasses.jsonld_field(
        default=None,
        description="Observatory/platform identifier.",
        input_types=[SDO.Text],
        url=constants.ML_COMMONS_GEO_OBSERVATORY,
    )
    instrument: str | None = mlc_dataclasses.jsonld_field(
        default=None,
        description="Instrument identifier.",
        input_types=[SDO.Text],
        url=constants.ML_COMMONS_GEO_INSTRUMENT,
    )
