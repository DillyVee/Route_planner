"""
Legacy implementations of DRPP algorithms.

This package contains historical implementations that have been superseded
by the V4 production code in drpp_core/. These files are kept for:
- Backward compatibility
- Historical reference
- Algorithm comparison and benchmarking

For new projects, use the V4 implementation in drpp_core/ instead.
"""

__all__ = [
    'parallel_processing_addon',
    'parallel_processing_addon_greedy',
    'parallel_processing_addon_greedy_v2',
    'parallel_processing_addon_greedy_v3',
    'parallel_processing_addon_rfcs',
]
