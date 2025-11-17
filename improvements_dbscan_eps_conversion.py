"""
IMPROVEMENT 5: Accurate DBSCAN eps Conversion
==============================================

Problem (line 273): eps_deg = eps_km / 111.0
  - Inaccurate for large latitude spans
  - 1 degree latitude = 111 km (accurate)
  - 1 degree longitude = 111 km × cos(latitude) (varies!)
  - At 60° latitude: 1 degree lon ≈ 55 km (50% error!)

Solutions:
  1. Convert coordinates to meters using Mercator/UTM projection
  2. Use haversine metric directly in DBSCAN
  3. Calculate latitude-adjusted eps
  4. Warn users and fallback to grid clustering for large areas
"""

from math import radians, cos, sin, asin, sqrt, degrees
import numpy as np
from typing import List, Tuple, Dict


# ============================================================================
# SOLUTION 1: HAVERSINE METRIC IN DBSCAN (Best)
# ============================================================================

def dbscan_cluster_segments_haversine(segments, eps_km=5.0, min_samples=3):
    """
    DBSCAN clustering with TRUE geographic distance (haversine).

    Uses sklearn's DBSCAN with haversine metric directly.

    Benefits:
      - Geographically accurate for any latitude
      - No approximation errors
      - Works globally

    Requirements:
      - scikit-learn 0.19+ (supports haversine metric)

    Args:
        segments: List of segment dicts with 'start' and 'end' coords
        eps_km: Maximum distance between segments in same cluster (kilometers)
        min_samples: Minimum segments to form a cluster

    Returns:
        Dict[cluster_id] -> [seg_idx, ...]
    """
    try:
        from sklearn.cluster import DBSCAN
        import numpy as np
    except ImportError:
        print("  ⚠️ scikit-learn not available, falling back to grid")
        return grid_cluster_segments_fallback(segments)

    # Extract centroids
    centroids = []
    for seg in segments:
        lat = (seg['start'][0] + seg['end'][0]) / 2.0
        lon = (seg['start'][1] + seg['end'][1]) / 2.0
        centroids.append([lat, lon])

    X = np.array(centroids)

    # Check for large latitude spans (warn user)
    lat_span = X[:, 0].max() - X[:, 0].min()
    if lat_span > 1.0:
        print(f"  ⚠️ Warning: Large latitude span ({lat_span:.1f}°) detected")
        print(f"     Using haversine metric for geographic accuracy")

    # Convert coordinates to radians (required for haversine metric)
    X_radians = np.radians(X)

    # Convert eps from km to radians
    # Earth radius = 6371 km
    # arc_length = radius × angle_radians
    # angle_radians = arc_length / radius
    eps_radians = eps_km / 6371.0

    # Run DBSCAN with haversine metric
    db = DBSCAN(
        eps=eps_radians,
        min_samples=min_samples,
        metric='haversine'  # TRUE geographic distance
    )

    labels = db.fit_predict(X_radians)

    # Group segments by cluster
    clusters = {}
    for seg_idx, label in enumerate(labels):
        cluster_id = int(label)
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(seg_idx)

    # Handle noise points (label = -1)
    if -1 in clusters:
        max_label = max(label for label in clusters.keys() if label != -1) if len(clusters) > 1 else 0
        clusters[max_label + 1] = clusters.pop(-1)

    print(f"  DBSCAN (haversine): {len(clusters)} clusters (eps={eps_km}km, min_samples={min_samples})")

    return clusters


# ============================================================================
# SOLUTION 2: MERCATOR PROJECTION (Alternative)
# ============================================================================

def lat_lon_to_mercator(lat, lon):
    """
    Convert lat/lon to Mercator projection (meters).

    Accurate for latitudes up to ~85° (breaks near poles)

    Returns:
        (x, y) in meters
    """
    R = 6378137.0  # Earth radius in meters (WGS84)

    x = R * radians(lon)
    y = R * np.log(np.tan(np.pi/4 + radians(lat)/2))

    return x, y


def dbscan_cluster_segments_mercator(segments, eps_km=5.0, min_samples=3):
    """
    DBSCAN using Mercator projection coordinates.

    Benefits:
      - Works with standard Euclidean distance
      - More accurate than degree approximation
      - Fast (no haversine calculations)

    Limitations:
      - Distortion near poles (>85° latitude)

    Args:
        segments: List of segment dicts
        eps_km: Maximum distance (kilometers)
        min_samples: Minimum segments per cluster

    Returns:
        Dict[cluster_id] -> [seg_idx, ...]
    """
    try:
        from sklearn.cluster import DBSCAN
        import numpy as np
    except ImportError:
        print("  ⚠️ scikit-learn not available, falling back to grid")
        return grid_cluster_segments_fallback(segments)

    # Extract centroids and convert to Mercator
    centroids_mercator = []
    for seg in segments:
        lat = (seg['start'][0] + seg['end'][0]) / 2.0
        lon = (seg['start'][1] + seg['end'][1]) / 2.0

        # Warn if near poles
        if abs(lat) > 85:
            print(f"  ⚠️ Warning: Segment near pole (lat={lat:.1f}°), projection distortion possible")

        x, y = lat_lon_to_mercator(lat, lon)
        centroids_mercator.append([x, y])

    X = np.array(centroids_mercator)

    # eps in meters (no conversion needed)
    eps_meters = eps_km * 1000.0

    # Run DBSCAN with Euclidean distance on projected coordinates
    db = DBSCAN(eps=eps_meters, min_samples=min_samples, metric='euclidean')
    labels = db.fit_predict(X)

    # Group segments
    clusters = {}
    for seg_idx, label in enumerate(labels):
        cluster_id = int(label)
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(seg_idx)

    # Handle noise
    if -1 in clusters:
        max_label = max(label for label in clusters.keys() if label != -1) if len(clusters) > 1 else 0
        clusters[max_label + 1] = clusters.pop(-1)

    print(f"  DBSCAN (Mercator): {len(clusters)} clusters (eps={eps_km}km, min_samples={min_samples})")

    return clusters


# ============================================================================
# SOLUTION 3: LATITUDE-ADJUSTED EPS (Quick Fix)
# ============================================================================

def dbscan_cluster_segments_adjusted_eps(segments, eps_km=5.0, min_samples=3):
    """
    DBSCAN with latitude-adjusted eps (quick fix for current code).

    Uses average latitude to adjust eps conversion.

    Accuracy: Good for small areas (<100km span), poor for large areas

    Args:
        segments: List of segment dicts
        eps_km: Maximum distance (kilometers)
        min_samples: Minimum segments per cluster

    Returns:
        Dict[cluster_id] -> [seg_idx, ...]
    """
    try:
        from sklearn.cluster import DBSCAN
        import numpy as np
    except ImportError:
        print("  ⚠️ scikit-learn not available, falling back to grid")
        return grid_cluster_segments_fallback(segments)

    # Extract centroids
    centroids = []
    for seg in segments:
        lat = (seg['start'][0] + seg['end'][0]) / 2.0
        lon = (seg['start'][1] + seg['end'][1]) / 2.0
        centroids.append([lat, lon])

    X = np.array(centroids)

    # Calculate average latitude
    avg_lat = np.mean(X[:, 0])
    lat_span = X[:, 0].max() - X[:, 0].min()
    lon_span = X[:, 1].max() - X[:, 1].min()

    # Warn if area is too large for simple approximation
    if lat_span > 1.0 or lon_span > 1.0:
        print(f"  ⚠️ Warning: Large geographic span (lat={lat_span:.1f}°, lon={lon_span:.1f}°)")
        print(f"     Consider using haversine metric for better accuracy")
        print(f"     Falling back to grid clustering for safety")
        return grid_cluster_segments_fallback(segments)

    # Latitude-adjusted conversion
    # 1 degree latitude ≈ 111 km (constant)
    # 1 degree longitude ≈ 111 km × cos(latitude) (varies)
    km_per_degree_lat = 111.0
    km_per_degree_lon = 111.0 * cos(radians(avg_lat))

    # Convert eps to degrees (using geometric mean for elliptical distance)
    eps_deg_lat = eps_km / km_per_degree_lat
    eps_deg_lon = eps_km / km_per_degree_lon

    # Use average for isotropic approximation
    eps_deg = (eps_deg_lat + eps_deg_lon) / 2.0

    print(f"  DBSCAN (adjusted eps): avg_lat={avg_lat:.1f}°, eps_deg={eps_deg:.4f}°")

    # Run DBSCAN
    db = DBSCAN(eps=eps_deg, min_samples=min_samples, metric='euclidean')
    labels = db.fit_predict(X)

    # Group segments
    clusters = {}
    for seg_idx, label in enumerate(labels):
        cluster_id = int(label)
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(seg_idx)

    # Handle noise
    if -1 in clusters:
        max_label = max(label for label in clusters.keys() if label != -1) if len(clusters) > 1 else 0
        clusters[max_label + 1] = clusters.pop(-1)

    print(f"  DBSCAN: {len(clusters)} clusters (eps={eps_km}km, min_samples={min_samples})")

    return clusters


# ============================================================================
# SOLUTION 4: AUTO-SELECT BEST METHOD
# ============================================================================

def dbscan_cluster_segments_smart(segments, eps_km=5.0, min_samples=3):
    """
    Smart DBSCAN clustering - auto-selects best method based on data.

    Decision tree:
      - Large area (lat/lon span > 1°): Use haversine metric
      - Medium area (0.1° < span < 1°): Use Mercator projection
      - Small area (span < 0.1°): Use adjusted eps (fast)
      - Very large area (span > 10°): Warn and use grid clustering

    Args:
        segments: List of segment dicts
        eps_km: Maximum distance (kilometers)
        min_samples: Minimum segments per cluster

    Returns:
        Dict[cluster_id] -> [seg_idx, ...]
    """
    # Analyze geographic span
    lats = [(seg['start'][0] + seg['end'][0]) / 2.0 for seg in segments]
    lons = [(seg['start'][1] + seg['end'][1]) / 2.0 for seg in segments]

    lat_span = max(lats) - min(lats)
    lon_span = max(lons) - min(lons)
    avg_lat = sum(lats) / len(lats)

    print(f"  Geographic analysis: lat_span={lat_span:.2f}°, lon_span={lon_span:.2f}°, avg_lat={avg_lat:.1f}°")

    # Decision tree
    if lat_span > 10.0 or lon_span > 10.0:
        print(f"  → Very large area detected, using grid clustering (DBSCAN not recommended)")
        return grid_cluster_segments_fallback(segments)

    elif lat_span > 1.0 or lon_span > 1.0 or abs(avg_lat) > 60:
        print(f"  → Large area or high latitude, using haversine metric")
        return dbscan_cluster_segments_haversine(segments, eps_km, min_samples)

    elif lat_span > 0.1 or lon_span > 0.1:
        print(f"  → Medium area, using Mercator projection")
        return dbscan_cluster_segments_mercator(segments, eps_km, min_samples)

    else:
        print(f"  → Small area, using adjusted eps (fast)")
        return dbscan_cluster_segments_adjusted_eps(segments, eps_km, min_samples)


# ============================================================================
# FALLBACK: GRID CLUSTERING
# ============================================================================

def grid_cluster_segments_fallback(segments, gx=8, gy=8):
    """Fallback grid clustering (from original code)"""
    from collections import defaultdict

    lats = [(s['start'][0] + s['end'][0]) / 2.0 for s in segments]
    lons = [(s['start'][1] + s['end'][1]) / 2.0 for s in segments]

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    lat_step = (max_lat - min_lat) / gy if max_lat > min_lat else 1.0
    lon_step = (max_lon - min_lon) / gx if max_lon > min_lon else 1.0

    clusters = {}
    for i, seg in enumerate(segments):
        clat = (seg['start'][0] + seg['end'][0]) / 2.0
        clon = (seg['start'][1] + seg['end'][1]) / 2.0

        ix = int((clon - min_lon) / lon_step) if lon_step > 0 else 0
        iy = int((clat - min_lat) / lat_step) if lat_step > 0 else 0

        if ix == gx:
            ix = gx - 1
        if iy == gy:
            iy = gy - 1

        cid = iy * gx + ix
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(i)

    print(f"  Grid clustering: {len(clusters)} clusters")

    return clusters


# ============================================================================
# ACCURACY COMPARISON
# ============================================================================

def compare_eps_accuracy(eps_km=5.0):
    """
    Compare accuracy of different eps conversion methods.

    Shows error percentage for different latitudes.
    """
    latitudes = [0, 30, 45, 60, 75, 85]

    print("="*70)
    print(f"EPS CONVERSION ACCURACY COMPARISON (eps = {eps_km} km)")
    print("="*70)
    print(f"{'Latitude':<12} {'Original':<15} {'Adjusted':<15} {'Error %':<10}")
    print("-"*70)

    for lat in latitudes:
        # Original method (current code)
        eps_deg_original = eps_km / 111.0

        # Latitude-adjusted method
        km_per_deg_lon = 111.0 * cos(radians(lat))
        eps_deg_adjusted = eps_km / km_per_deg_lon

        # Error percentage (longitude direction)
        error_pct = abs(eps_deg_adjusted - eps_deg_original) / eps_deg_adjusted * 100

        print(f"{lat:>4}°        {eps_deg_original:.6f}°      {eps_deg_adjusted:.6f}°      {error_pct:>6.1f}%")

    print("="*70)
    print("CONCLUSION: At high latitudes, original method has significant error!")
    print("RECOMMENDATION: Use haversine metric or Mercator projection")
    print("="*70)


# ============================================================================
# DROP-IN REPLACEMENT
# ============================================================================

# Replace line 237-292 in parallel_processing_addon_greedy_v2.py with:
dbscan_cluster_segments = dbscan_cluster_segments_smart


if __name__ == '__main__':
    # Show accuracy comparison
    compare_eps_accuracy(eps_km=5.0)
