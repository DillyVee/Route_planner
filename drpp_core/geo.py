"""
Geographic utilities for the DRPP solver.

Provides distance calculations and coordinate transformations optimized
for route planning applications.
"""

from math import asin, cos, radians, sin, sqrt
from typing import Tuple

# Type alias for clarity
Coordinate = Tuple[float, float]  # (latitude, longitude)


def haversine(coord1: Coordinate, coord2: Coordinate) -> float:
    """
    Calculate great-circle distance between two points using Haversine formula.

    The Haversine formula determines the great-circle distance between two points
    on a sphere given their longitudes and latitudes. This is accurate for most
    routing applications (error < 0.5% for distances up to 500km).

    Args:
        coord1: First coordinate as (latitude, longitude) in decimal degrees
        coord2: Second coordinate as (latitude, longitude) in decimal degrees

    Returns:
        Distance in meters (float)

    Example:
        >>> san_francisco = (37.7749, -122.4194)
        >>> los_angeles = (34.0522, -118.2437)
        >>> distance = haversine(san_francisco, los_angeles)
        >>> print(f"{distance / 1000:.1f} km")
        559.1 km

    Note:
        - Earth radius is approximated as 6,371 km
        - For sub-meter accuracy or distances > 500km, consider using Vincenty formula
        - Coordinates must be in (lat, lon) format, not (lon, lat)
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    # Earth radius in meters
    earth_radius_m = 6371000
    return earth_radius_m * c


def snap_coordinate(lat: float, lon: float, precision: int = 6) -> Coordinate:
    """
    Snap coordinates to fixed precision to eliminate near-duplicates.

    This is useful for eliminating floating-point precision errors and
    consolidating nearly-identical coordinates.

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        precision: Number of decimal places (default: 6)

    Returns:
        Snapped coordinate as (lat, lon) tuple

    Precision guide:
        - precision=4: ~11m accuracy (suitable for city-level routing)
        - precision=5: ~1.1m accuracy (suitable for street-level routing)
        - precision=6: ~0.11m accuracy (recommended, sub-meter)
        - precision=7: ~0.01m accuracy (overkill for most applications)

    Example:
        >>> coord = snap_coordinate(37.774929999, -122.419415001, precision=6)
        >>> print(coord)
        (37.774930, -122.419415)
    """
    return (round(lat, precision), round(lon, precision))


def calculate_bearing(coord1: Coordinate, coord2: Coordinate) -> float:
    """
    Calculate initial bearing (forward azimuth) between two points.

    Args:
        coord1: Start coordinate as (latitude, longitude)
        coord2: End coordinate as (latitude, longitude)

    Returns:
        Bearing in degrees (0-360), where 0° is North, 90° is East, etc.

    Example:
        >>> start = (37.7749, -122.4194)
        >>> end = (34.0522, -118.2437)
        >>> bearing = calculate_bearing(start, end)
        >>> print(f"{bearing:.1f}°")
        126.5°
    """
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)

    dlon = lon2 - lon1

    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)

    import math

    bearing_rad = math.atan2(x, y)
    bearing_deg = (math.degrees(bearing_rad) + 360) % 360

    return bearing_deg
