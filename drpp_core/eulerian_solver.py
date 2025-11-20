"""
Eulerian tour solver for DRPP using node balancing and Hierholzer's algorithm.

This is the core industry-standard DRPP methodology:
1. Analyze node degree balance (in-degree vs out-degree)
2. Add minimum-cost edges to balance the graph (make it Eulerian)
3. Construct Eulerian tour using Hierholzer's algorithm
4. Simplify and optimize the tour

References:
- Edmonds & Johnson (1973): Matching, Euler tours and the Chinese postman
- Hierholzer's algorithm (1873): Finding Eulerian circuits
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import heapq

from .topology import TopologyEdge, NodeID, build_adjacency_list, get_node_degrees


@dataclass
class NodeBalance:
    """Node degree balance information."""
    node_id: NodeID
    in_degree: int
    out_degree: int
    balance: int  # out_degree - in_degree

    @property
    def is_balanced(self) -> bool:
        """Check if node is balanced (in-degree == out-degree)."""
        return self.balance == 0

    @property
    def is_source(self) -> bool:
        """Check if node is a source (out-degree > in-degree)."""
        return self.balance > 0

    @property
    def is_sink(self) -> bool:
        """Check if node is a sink (in-degree > out-degree)."""
        return self.balance < 0


@dataclass
class EulerianTour:
    """Result of Eulerian tour computation."""
    edges: List[TopologyEdge]  # Ordered list of edges in tour
    total_distance: float
    num_required: int
    num_deadhead: int
    deadhead_distance: float
    is_valid: bool
    message: str


class EulerianSolver:
    """
    Solves DRPP using Eulerian augmentation approach.

    This is the industry-standard method used by DOTs and mapping companies.
    """

    def __init__(self, edges: List[TopologyEdge], num_nodes: int):
        """
        Initialize Eulerian solver.

        Args:
            edges: List of all edges (required + optional/deadhead)
            num_nodes: Total number of nodes in graph
        """
        self.original_edges = edges.copy()
        self.num_nodes = num_nodes
        self.augmented_edges: List[TopologyEdge] = []
        self.node_balances: List[NodeBalance] = []

    def solve(self, shortest_path_func) -> EulerianTour:
        """
        Solve DRPP using Eulerian augmentation.

        Args:
            shortest_path_func: Function(from_node, to_node) -> (path_coords, distance)

        Returns:
            EulerianTour object with solution
        """
        # Step 1: Analyze node balance
        self.node_balances = self._compute_node_balances()

        # Step 2: Check if graph is already Eulerian
        imbalanced_nodes = [nb for nb in self.node_balances if not nb.is_balanced]

        if not imbalanced_nodes:
            # Graph is already Eulerian, just compute tour
            tour = self._hierholzer_algorithm(self.original_edges)
            return tour

        # Step 3: Add edges to balance the graph
        balancing_edges = self._compute_balancing_edges(shortest_path_func)

        # Step 4: Construct augmented edge list
        all_edges = self.original_edges + balancing_edges
        self.augmented_edges = balancing_edges

        # Step 5: Compute Eulerian tour
        tour = self._hierholzer_algorithm(all_edges)

        return tour

    def _compute_node_balances(self) -> List[NodeBalance]:
        """
        Compute in-degree and out-degree balance for all nodes.

        Returns:
            List of NodeBalance objects
        """
        degrees = get_node_degrees(self.original_edges, self.num_nodes)

        balances = []
        for node_id in range(self.num_nodes):
            in_deg, out_deg = degrees[node_id]
            balance = out_deg - in_deg

            balances.append(NodeBalance(
                node_id=node_id,
                in_degree=in_deg,
                out_degree=out_deg,
                balance=balance
            ))

        return balances

    def _compute_balancing_edges(self, shortest_path_func) -> List[TopologyEdge]:
        """
        Compute minimum-cost edges to balance the graph.

        This uses a greedy matching approach:
        1. Identify sources (out-degree > in-degree) and sinks (in-degree > out-degree)
        2. Match each source to sinks using shortest paths
        3. Add duplicate edges along shortest paths

        For optimal solution, use min-cost flow (scipy or OR-Tools).
        This greedy approach is fast and gives good results for typical road networks.

        Args:
            shortest_path_func: Function to compute shortest path between nodes

        Returns:
            List of balancing edges to add
        """
        # Identify sources and sinks
        sources = []  # (node_id, excess)
        sinks = []    # (node_id, deficit)

        for nb in self.node_balances:
            if nb.is_source:
                sources.append((nb.node_id, nb.balance))
            elif nb.is_sink:
                sinks.append((nb.node_id, -nb.balance))  # Convert to positive deficit

        if not sources or not sinks:
            return []  # Graph is balanced

        # Greedy matching: match each unit of excess to nearest deficit
        balancing_edges = []
        edge_counter = 0

        # Use priority queue for efficient greedy matching
        # Format: (distance, source_node, sink_node)
        matches: List[Tuple[float, NodeID, NodeID, List, int]] = []

        # Compute all source-sink distances
        for source_node, excess in sources:
            for sink_node, deficit in sinks:
                path_coords, distance = shortest_path_func(source_node, sink_node)
                if path_coords and distance < float('inf'):
                    matches.append((distance, source_node, sink_node, path_coords, min(excess, deficit)))

        # Sort by distance (greedy)
        matches.sort(key=lambda x: x[0])

        # Track remaining capacity
        source_remaining = {node: excess for node, excess in sources}
        sink_remaining = {node: deficit for node, deficit in sinks}

        # Greedy matching
        for distance, source_node, sink_node, path_coords, _ in matches:
            # How much can we send on this path?
            source_avail = source_remaining.get(source_node, 0)
            sink_avail = sink_remaining.get(sink_node, 0)
            flow = min(source_avail, sink_avail)

            if flow > 0:
                # Add 'flow' copies of this edge
                for _ in range(flow):
                    edge = TopologyEdge(
                        edge_id=f"balance_{edge_counter}",
                        from_node=source_node,
                        to_node=sink_node,
                        cost=distance,
                        required=False,  # Balancing edge (deadhead)
                        segment_id=None,
                        coordinates=path_coords,
                        metadata={"type": "balancing_edge", "purpose": "node_balance"}
                    )
                    balancing_edges.append(edge)
                    edge_counter += 1

                # Update remaining capacity
                source_remaining[source_node] -= flow
                sink_remaining[sink_node] -= flow

        return balancing_edges

    def _hierholzer_algorithm(self, edges: List[TopologyEdge]) -> EulerianTour:
        """
        Construct Eulerian tour using Hierholzer's algorithm.

        This is the classic O(E) algorithm for finding Eulerian circuits.

        Args:
            edges: List of all edges (should form an Eulerian graph)

        Returns:
            EulerianTour with ordered edge sequence
        """
        if not edges:
            return EulerianTour(
                edges=[],
                total_distance=0,
                num_required=0,
                num_deadhead=0,
                deadhead_distance=0,
                is_valid=False,
                message="No edges to traverse"
            )

        # Build adjacency list with edge tracking
        adjacency: Dict[NodeID, List[TopologyEdge]] = defaultdict(list)
        for edge in edges:
            adjacency[edge.from_node].append(edge)

        # Track used edges
        edge_used = {id(edge): False for edge in edges}

        # Find starting node (any node with outgoing edges)
        start_node = None
        for node in adjacency:
            if adjacency[node]:
                start_node = node
                break

        if start_node is None:
            return EulerianTour(
                edges=[],
                total_distance=0,
                num_required=0,
                num_deadhead=0,
                deadhead_distance=0,
                is_valid=False,
                message="No starting node found"
            )

        # Hierholzer's algorithm
        tour: List[TopologyEdge] = []
        stack = [start_node]
        current_path: List[TopologyEdge] = []

        while stack:
            current = stack[-1]

            # Find an unused edge from current node
            found_edge = False
            for edge in adjacency.get(current, []):
                if not edge_used[id(edge)]:
                    # Use this edge
                    edge_used[id(edge)] = True
                    stack.append(edge.to_node)
                    current_path.append(edge)
                    found_edge = True
                    break

            if not found_edge:
                # No more edges from current node, backtrack
                stack.pop()
                if current_path:
                    # Add edge to tour (in reverse order)
                    tour.append(current_path.pop())

        # Reverse to get correct order
        tour.reverse()

        # Check if all edges were used
        unused_count = sum(1 for used in edge_used.values() if not used)
        is_valid = (unused_count == 0)

        # Compute statistics
        total_distance = sum(edge.cost for edge in tour)
        num_required = sum(1 for edge in tour if edge.required)
        num_deadhead = len(tour) - num_required
        deadhead_distance = sum(edge.cost for edge in tour if not edge.required)

        message = "Valid Eulerian tour" if is_valid else f"Incomplete tour ({unused_count} edges unused)"

        return EulerianTour(
            edges=tour,
            total_distance=total_distance,
            num_required=num_required,
            num_deadhead=num_deadhead,
            deadhead_distance=deadhead_distance,
            is_valid=is_valid,
            message=message
        )

    def get_balance_summary(self) -> str:
        """
        Get human-readable summary of node balances.

        Returns:
            Formatted string with balance statistics
        """
        if not self.node_balances:
            return "No balance analysis performed yet."

        balanced = sum(1 for nb in self.node_balances if nb.is_balanced)
        sources = sum(1 for nb in self.node_balances if nb.is_source)
        sinks = sum(1 for nb in self.node_balances if nb.is_sink)

        total_excess = sum(nb.balance for nb in self.node_balances if nb.is_source)
        total_deficit = sum(-nb.balance for nb in self.node_balances if nb.is_sink)

        lines = []
        lines.append(f"Node Balance Analysis:")
        lines.append(f"  - Balanced nodes: {balanced}")
        lines.append(f"  - Source nodes (out > in): {sources} (total excess: {total_excess})")
        lines.append(f"  - Sink nodes (in > out): {sinks} (total deficit: {total_deficit})")
        lines.append(f"  - Balancing edges needed: ~{total_excess} (one per excess unit)")

        return "\n".join(lines)


def solve_drpp_eulerian(edges: List[TopologyEdge], num_nodes: int,
                        shortest_path_func) -> EulerianTour:
    """
    Convenience function to solve DRPP using Eulerian approach.

    Args:
        edges: List of edges (required + optional)
        num_nodes: Total number of nodes
        shortest_path_func: Function(from_node, to_node) -> (path, distance)

    Returns:
        EulerianTour with solution
    """
    solver = EulerianSolver(edges, num_nodes)
    return solver.solve(shortest_path_func)
