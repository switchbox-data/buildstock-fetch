class InvalidProductError(ValueError):
    """Raised when an invalid product is provided."""

    pass


class InvalidReleaseNameError(ValueError):
    """Raised when an invalid release name is provided."""

    pass


class NoBuildingDataError(ValueError):
    """Raised when no building data is available for a given release."""

    pass


class NoMetadataError(ValueError):
    """Raised when no metadata is available for a given release."""

    pass


class No15minLoadCurveError(ValueError):
    """Raised when no 15 min load profile timeseries is available for a given release."""

    pass


class NoAnnualLoadCurveError(ValueError):
    """Raised when annual load curve is not available for a release."""

    pass


class NoAggregateLoadCurveError(ValueError):
    """Raised when no monthly load curve is available for a given release."""

    pass


class UnknownAggregationFunctionError(ValueError):
    """Raised when an unknown aggregation function is provided."""

    pass


class NoWeatherFileError(ValueError):
    """Raised when weather file is not available for a release."""

    pass
