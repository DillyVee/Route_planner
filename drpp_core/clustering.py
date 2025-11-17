"""
Geographic-aware clustering for road segments.

Supports DBSCAN, K-means, and grid-based clustering with automatic
method selection based on data characteristics.
"""

from typing import List, Dict, Tuple, cast
from enum import Enum
from math import radians, cos, sin, asin, sqrt
from collections import defaultdict

from .types import Coordinate, ClusterID, SegmentIndex, ClusterResult
from .logging_config import get_logger, LogTimer

logger = get_logger(__name__)

# Try to import scikit-learn
try:
    from sklearn.cluster import DBSCAN, KMeans  # type: ignore[import-untyped]
    import numpy as np

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.info("scikit-learn not available - only grid clustering will be available")


class ClusteringMethod(Enum):
    """Available clustering methods."""

    DBSCAN = "dbscan"  # Density-based clustering (best for irregular shapes)
    KMEANS = "kmeans"  # Centroid-based clustering (best when K is known)
    GRID = "grid"  # Simple grid-based clustering (fallback, always available)


def haversine(coord1: Coordinate, coord2: Coordinate) -> float:
    """Calculate great-circle distance between two points in meters.

    Uses the haversine formula for spherical distance calculation.

    Args:
        coord1: First coordinate (latitude, longitude) in decimal degrees
        coord2: Second coordinate (latitude, longitude) in decimal degrees

    Returns:
        Distance in meters

    Example:
        >>> dist = haversine((40.7128, -74.0060), (34.0522, -118.2437))
        >>> print(f"NYC to LA: {dist / 1000:.0f} km")
        NYC to LA: 3944 km
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
    earth_radius = 6371000
    return earth_radius * c


def _analyze_geographic_span(centroids: List[Coordinate]) -> Tuple[float, float, float]:
    """Analyze geographic extent of points.

    Args:
        centroids: List of (lat, lon) coordinates

    Returns:
        Tuple of (lat_span, lon_span, avg_lat) in degrees
    """
    lats = [coord[0] for coord in centroids]
    lons = [coord[1] for coord in centroids]

    lat_span = max(lats) - min(lats)
    lon_span = max(lons) - min(lons)
    avg_lat = sum(lats) / len(lats)

    return lat_span, lon_span, avg_lat


def _dbscan_haversine(centroids: List[Coordinate], eps_km: float, min_samples: int) -> List[int]:
    """DBSCAN with true haversine metric.

    Best for: Large geographic areas, high latitudes, global datasets

    Args:
        centroids: List of (lat, lon) centroids
        eps_km: Maximum distance in kilometers
        min_samples: Minimum points to form a cluster

    Returns:
        List of cluster labels (one per centroid)
    """
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn required for DBSCAN")

    logger.debug(f"Running haversine DBSCAN: eps={eps_km}km, min_samples={min_samples}")

    # Convert to radians for haversine metric
    X_radians = np.radians(centroids)
    eps_radians = eps_km / 6371.0  # Convert km to radians

    db = DBSCAN(eps=eps_radians, min_samples=min_samples, metric="haversine")
    labels = db.fit_predict(X_radians)

    return cast(List[int], labels.tolist())


def _dbscan_mercator(centroids: List[Coordinate], eps_km: float, min_samples: int) -> List[int]:
    """DBSCAN with Mercator projection.

    Best for: Medium geographic areas at mid-latitudes

    Args:
        centroids: List of (lat, lon) centroids
        eps_km: Maximum distance in kilometers
        min_samples: Minimum points to form a cluster

    Returns:
        List of cluster labels
    """
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn required for DBSCAN")

    logger.debug(f"Running Mercator DBSCAN: eps={eps_km}km, min_samples={min_samples}")

    # Project to Mercator (meters)
    earth_radius = 6378137.0  # WGS84 equatorial radius
    centroids_mercator = []

    for lat, lon in centroids:
        x = earth_radius * radians(lon)
        # Mercator y-coordinate
        y = earth_radius * np.log(np.tan(np.pi / 4 + radians(lat) / 2))
        centroids_mercator.append([x, y])

    X = np.array(centroids_mercator)
    eps_meters = eps_km * 1000.0

    db = DBSCAN(eps=eps_meters, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(X)

    return cast(List[int], labels.tolist())


def _dbscan_adjusted_eps(
    centroids: List[Coordinate], eps_km: float, min_samples: int, avg_lat: float
) -> List[int]:
    """DBSCAN with latitude-adjusted epsilon.

    Best for: Small geographic areas with consistent latitude

    Args:
        centroids: List of (lat, lon) centroids
        eps_km: Maximum distance in kilometers
        min_samples: Minimum points to form a cluster
        avg_lat: Average latitude for adjustment

    Returns:
        List of cluster labels
    """
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn required for DBSCAN")

    logger.debug(
        f"Running adjusted-eps DBSCAN: eps={eps_km}km, "
        f"min_samples={min_samples}, avg_lat={avg_lat:.2f}"
    )

    # Approximate conversion at average latitude
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * cos(radians(avg_lat))
    eps_deg = eps_km / ((km_per_deg_lat + km_per_deg_lon) / 2.0)

    X = np.array(centroids)
    db = DBSCAN(eps=eps_deg, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(X)

    return cast(List[int], labels.tolist())


def _select_dbscan_method(
    centroids: List[Coordinate], eps_km: float, min_samples: int
) -> List[int]:
    """Automatically select best DBSCAN method based on data.

    Selection criteria:
    - Very large area (>10°): Grid clustering (DBSCAN not suitable)
    - Large area or high latitude: Haversine metric
    - Medium area: Mercator projection
    - Small area: Adjusted epsilon

    Args:
        centroids: List of (lat, lon) centroids
        eps_km: Maximum distance in kilometers
        min_samples: Minimum points to form a cluster

    Returns:
        List of cluster labels
    """
    lat_span, lon_span, avg_lat = _analyze_geographic_span(centroids)

    logger.info(f"Geographic span: {lat_span:.2f}° × {lon_span:.2f}° " f"(avg lat: {avg_lat:.2f}°)")

    # Very large area - use grid instead
    if lat_span > 10.0 or lon_span > 10.0:
        logger.warning(
            f"Area too large for DBSCAN ({lat_span:.1f}° × {lon_span:.1f}°), "
            "falling back to grid clustering"
        )
        # Convert centroids back to segments format for grid clustering
        # This is a fallback case
        raise ValueError("Area too large for DBSCAN - use grid clustering")

    # Large area or high latitude - use haversine
    elif lat_span > 1.0 or lon_span > 1.0 or abs(avg_lat) > 60:
        logger.info("Using haversine DBSCAN (large area or high latitude)")
        return _dbscan_haversine(centroids, eps_km, min_samples)

    # Medium area - use Mercator
    elif lat_span > 0.1 or lon_span > 0.1:
        logger.info("Using Mercator DBSCAN (medium area)")
        return _dbscan_mercator(centroids, eps_km, min_samples)

    # Small area - use adjusted eps
    else:
        logger.info("Using adjusted-eps DBSCAN (small area)")
        return _dbscan_adjusted_eps(centroids, eps_km, min_samples, avg_lat)


def _process_labels(
    labels: List[int], handle_noise: bool = True
) -> Dict[ClusterID, List[SegmentIndex]]:
    """Convert sklearn labels to cluster dict.

    Args:
        labels: List of cluster labels from sklearn
        handle_noise: If True, noise points (-1) get their own cluster

    Returns:
        Dict mapping cluster ID to list of segment indices
    """
    clusters: Dict[int, List[int]] = defaultdict(list)

    for seg_idx, label in enumerate(labels):
        clusters[label].append(seg_idx)

    # Handle noise cluster
    noise_count = 0
    if -1 in clusters:
        noise_count = len(clusters[-1])
        if handle_noise:
            # Move noise to new cluster ID
            max_label = max(label for label in clusters.keys() if label != -1)
            new_cluster_id = max_label + 1
            clusters[new_cluster_id] = clusters.pop(-1)
            logger.info(f"Moved {noise_count} noise points to cluster {new_cluster_id}")
        else:
            # Remove noise
            clusters.pop(-1)
            logger.warning(f"Discarded {noise_count} noise points")

    return dict(clusters)


def cluster_segments_dbscan(
    segments: List[Dict], eps_km: float = 5.0, min_samples: int = 3, handle_noise: bool = True
) -> ClusterResult:
    """Cluster segments using DBSCAN with automatic method selection.

    Args:
        segments: List of segment dicts with 'start' and 'end' coordinates
        eps_km: Maximum distance in kilometers for neighborhood
        min_samples: Minimum points to form a cluster
        handle_noise: If True, noise points get their own cluster

    Returns:
        ClusterResult with clusters and statistics

    Raises:
        RuntimeError: If scikit-learn is not available

    Example:
        >>> segments = [
        ...     {'start': (40.7, -74.0), 'end': (40.8, -74.1), 'coords': [...]},
        ...     # ... more segments
        ... ]
        >>> result = cluster_segments_dbscan(segments, eps_km=5.0)
        >>> print(f"Created {len(result.clusters)} clusters")
    """
    if not SKLEARN_AVAILABLE:
        raise RuntimeError(
            "scikit-learn required for DBSCAN clustering. " "Install with: pip install scikit-learn"
        )

    logger.info(
        f"Clustering {len(segments)} segments with DBSCAN "
        f"(eps={eps_km}km, min_samples={min_samples})"
    )

    with LogTimer(logger, "DBSCAN clustering"):
        # Compute centroids
        centroids = [
            ((seg["start"][0] + seg["end"][0]) / 2.0, (seg["start"][1] + seg["end"][1]) / 2.0)
            for seg in segments
        ]

        # Select and run appropriate DBSCAN variant
        labels = _select_dbscan_method(centroids, eps_km, min_samples)

        # Process labels into clusters
        clusters = _process_labels(labels, handle_noise)

        # Count noise
        noise_count = sum(1 for label in labels if label == -1)

    logger.info(f"DBSCAN complete: {len(clusters)} clusters, " f"{noise_count} noise points")

    return ClusterResult(clusters=clusters, noise_count=noise_count, method_used="dbscan_auto")


def cluster_segments_kmeans(segments: List[Dict], k_clusters: int = 40) -> ClusterResult:
    """Cluster segments using K-means.

    Args:
        segments: List of segment dicts
        k_clusters: Number of clusters to create

    Returns:
        ClusterResult with clusters and statistics

    Example:
        >>> result = cluster_segments_kmeans(segments, k_clusters=20)
    """
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn required for K-means")

    # Ensure k is valid
    k_clusters = min(k_clusters, max(1, len(segments)))

    logger.info(f"Clustering {len(segments)} segments with K-means (k={k_clusters})")

    with LogTimer(logger, "K-means clustering"):
        # Compute centroids
        centroids = [
            ((seg["start"][0] + seg["end"][0]) / 2.0, (seg["start"][1] + seg["end"][1]) / 2.0)
            for seg in segments
        ]

        X = np.array(centroids)
        km = KMeans(n_clusters=k_clusters, random_state=0, n_init=10)
        labels = km.fit_predict(X)

        clusters = _process_labels(labels.tolist(), handle_noise=False)

    logger.info(f"K-means complete: {len(clusters)} clusters")

    return ClusterResult(clusters=clusters, noise_count=0, method_used="kmeans")


def cluster_segments_grid(
    segments: List[Dict], grid_x: int = 10, grid_y: int = 10
) -> ClusterResult:
    """Cluster segments using simple grid-based method.

    This is a fast, deterministic fallback that doesn't require scikit-learn.

    Args:
        segments: List of segment dicts
        grid_x: Number of grid cells in x-direction
        grid_y: Number of grid cells in y-direction

    Returns:
        ClusterResult with clusters and statistics

    Example:
        >>> result = cluster_segments_grid(segments, grid_x=8, grid_y=8)
    """
    logger.info(f"Clustering {len(segments)} segments with grid " f"({grid_x}×{grid_y} cells)")

    # Handle empty segments
    if not segments:
        logger.warning("No segments to cluster")
        return ClusterResult(clusters={}, noise_count=0, method_used="grid")

    with LogTimer(logger, "Grid clustering"):
        # Compute centroids
        lats = [(seg["start"][0] + seg["end"][0]) / 2.0 for seg in segments]
        lons = [(seg["start"][1] + seg["end"][1]) / 2.0 for seg in segments]

        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        # Compute grid cell sizes
        lat_step = (max_lat - min_lat) / grid_y if max_lat > min_lat else 1.0
        lon_step = (max_lon - min_lon) / grid_x if max_lon > min_lon else 1.0

        # Assign segments to grid cells
        clusters: Dict[int, List[int]] = {}
        for i, seg in enumerate(segments):
            clat = (seg["start"][0] + seg["end"][0]) / 2.0
            clon = (seg["start"][1] + seg["end"][1]) / 2.0

            # Compute grid cell indices
            ix = int((clon - min_lon) / lon_step) if lon_step > 0 else 0
            iy = int((clat - min_lat) / lat_step) if lat_step > 0 else 0

            # Clamp to grid bounds
            ix = min(ix, grid_x - 1)
            iy = min(iy, grid_y - 1)

            # Compute cluster ID
            cluster_id = iy * grid_x + ix
            clusters.setdefault(cluster_id, []).append(i)

    logger.info(f"Grid clustering complete: {len(clusters)} clusters")

    return ClusterResult(clusters=clusters, noise_count=0, method_used="grid")


def cluster_segments(
    segments: List[Dict], method: ClusteringMethod = ClusteringMethod.GRID, **kwargs
) -> ClusterResult:
    """Cluster road segments using specified method.

    Args:
        segments: List of segment dicts with 'start', 'end', 'coords'
        method: Clustering method to use
        **kwargs: Method-specific parameters:
            For DBSCAN: eps_km (float), min_samples (int)
            For K-means: k_clusters (int)
            For grid: grid_x (int), grid_y (int)

    Returns:
        ClusterResult with clusters and metadata

    Example:
        >>> # Auto-select DBSCAN method
        >>> result = cluster_segments(segments, ClusteringMethod.DBSCAN, eps_km=5.0)
        >>> # Use grid clustering
        >>> result = cluster_segments(segments, ClusteringMethod.GRID, grid_x=10)

    Raises:
        ValueError: If invalid method specified
        RuntimeError: If required library not available
    """
    if len(segments) == 0:
        logger.warning("No segments to cluster")
        return ClusterResult(clusters={}, noise_count=0, method_used="none")

    if method == ClusteringMethod.DBSCAN:
        return cluster_segments_dbscan(
            segments,
            eps_km=kwargs.get("eps_km", 5.0),
            min_samples=kwargs.get("min_samples", 3),
            handle_noise=kwargs.get("handle_noise", True),
        )
    elif method == ClusteringMethod.KMEANS:
        return cluster_segments_kmeans(segments, k_clusters=kwargs.get("k_clusters", 40))
    elif method == ClusteringMethod.GRID:
        return cluster_segments_grid(
            segments, grid_x=kwargs.get("grid_x", 10), grid_y=kwargs.get("grid_y", 10)
        )
    else:
        raise ValueError(f"Unknown clustering method: {method}")
