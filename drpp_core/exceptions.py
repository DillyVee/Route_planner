"""
Custom exceptions for the DRPP Route Planner.

Provides a clear exception hierarchy for better error handling and debugging.
All exceptions inherit from DRPPError for easy catching of all library errors.
"""


class DRPPError(Exception):
    """Base exception for all DRPP-related errors."""

    pass


# ==============================================================================
# Input/Parsing Errors
# ==============================================================================


class ParseError(DRPPError):
    """Raised when parsing KML or input data fails."""

    pass


class KMLParseError(ParseError):
    """Raised specifically for KML parsing failures."""

    def __init__(self, filepath: str, reason: str):
        self.filepath = filepath
        self.reason = reason
        super().__init__(f"Failed to parse KML file '{filepath}': {reason}")


class ValidationError(DRPPError):
    """Raised when input validation fails."""

    pass


# ==============================================================================
# Graph Construction Errors
# ==============================================================================


class GraphError(DRPPError):
    """Base class for graph-related errors."""

    pass


class GraphBuildError(GraphError):
    """Raised when graph construction fails."""

    def __init__(self, reason: str, num_nodes: int = 0, num_edges: int = 0):
        self.reason = reason
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        msg = f"Graph construction failed: {reason}"
        if num_nodes or num_edges:
            msg += f" (nodes: {num_nodes}, edges: {num_edges})"
        super().__init__(msg)


class DisconnectedGraphError(GraphError):
    """Raised when graph is unexpectedly disconnected."""

    def __init__(self, num_components: int, unreachable_nodes: int = 0):
        self.num_components = num_components
        self.unreachable_nodes = unreachable_nodes
        super().__init__(
            f"Graph is disconnected: {num_components} components, "
            f"{unreachable_nodes} unreachable nodes"
        )


# ==============================================================================
# Routing Errors
# ==============================================================================


class RoutingError(DRPPError):
    """Base class for routing-related errors."""

    pass


class NoPathError(RoutingError):
    """Raised when no path exists between required segments."""

    def __init__(self, from_node, to_node):
        self.from_node = from_node
        self.to_node = to_node
        super().__init__(f"No path exists from {from_node} to {to_node}")


class UnreachableSegmentError(RoutingError):
    """Raised when required segments cannot be reached."""

    def __init__(self, segment_indices: list, reason: str = ""):
        self.segment_indices = segment_indices
        self.reason = reason
        msg = f"Cannot reach {len(segment_indices)} required segment(s)"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class OptimizationError(RoutingError):
    """Raised when route optimization fails."""

    pass


# ==============================================================================
# Clustering Errors
# ==============================================================================


class ClusteringError(DRPPError):
    """Raised when clustering fails."""

    def __init__(self, method: str, reason: str):
        self.method = method
        self.reason = reason
        super().__init__(f"Clustering with method '{method}' failed: {reason}")


# ==============================================================================
# OSM Integration Errors
# ==============================================================================


class OSMError(DRPPError):
    """Base class for OSM integration errors."""

    pass


class OverpassAPIError(OSMError):
    """Raised when Overpass API requests fail."""

    def __init__(self, reason: str, retry_after: int = None):
        self.reason = reason
        self.retry_after = retry_after
        msg = f"Overpass API error: {reason}"
        if retry_after:
            msg += f" (retry after {retry_after}s)"
        super().__init__(msg)


class OSMMatchingError(OSMError):
    """Raised when OSM road matching fails."""

    pass


# ==============================================================================
# Visualization Errors
# ==============================================================================


class VisualizationError(DRPPError):
    """Raised when visualization generation fails."""

    def __init__(self, format_type: str, reason: str):
        self.format_type = format_type
        self.reason = reason
        super().__init__(f"Failed to generate {format_type} visualization: {reason}")


# ==============================================================================
# Resource Errors
# ==============================================================================


class ResourceError(DRPPError):
    """Base class for resource-related errors."""

    pass


class MemoryError(ResourceError):  # noqa: A001
    """Raised when memory limits are exceeded."""

    def __init__(self, required_mb: float, available_mb: float):
        self.required_mb = required_mb
        self.available_mb = available_mb
        super().__init__(
            f"Insufficient memory: need {required_mb:.1f} MB, "
            f"have {available_mb:.1f} MB"
        )


class TimeoutError(ResourceError):  # noqa: A001
    """Raised when operations exceed time limits."""

    def __init__(self, operation: str, timeout_seconds: float):
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Operation '{operation}' exceeded timeout of {timeout_seconds}s"
        )


# ==============================================================================
# Configuration Errors
# ==============================================================================


class ConfigurationError(DRPPError):
    """Raised when configuration is invalid."""

    pass


# ==============================================================================
# Convenience Functions
# ==============================================================================


def handle_parse_error(filepath: str, original_error: Exception) -> None:
    """
    Convert generic parsing errors to specific KMLParseError.

    Args:
        filepath: Path to the file being parsed
        original_error: The original exception that was raised

    Raises:
        KMLParseError: Always raises with context from original error
    """
    import xml.etree.ElementTree as ET

    if isinstance(original_error, ET.ParseError):
        raise KMLParseError(filepath, f"XML syntax error: {original_error}") from original_error
    elif isinstance(original_error, (FileNotFoundError, PermissionError)):
        raise KMLParseError(filepath, str(original_error)) from original_error
    else:
        raise KMLParseError(filepath, f"Unexpected error: {type(original_error).__name__}: {original_error}") from original_error
