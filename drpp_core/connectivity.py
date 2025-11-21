"""
Graph connectivity analysis for DRPP feasibility checking.

This module implements:
- Tarjan's algorithm for strongly connected components (SCC)
- Reachability analysis
- Feasibility checking for DRPP
- Component visualization and reporting
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from .topology import TopologyEdge, NodeID, build_adjacency_list


@dataclass
class ConnectedComponent:
    """A strongly connected component in the graph."""
    component_id: int
    nodes: Set[NodeID]
    required_edges: List[TopologyEdge]
    all_edges: List[TopologyEdge]


class ConnectivityAnalyzer:
    """
    Analyzes graph connectivity using Tarjan's algorithm.

    This is essential for DRPP feasibility: we need to ensure all required
    edges are reachable and the graph structure supports an Eulerian tour.
    """

    def __init__(self, nodes: Set[NodeID], edges: List[TopologyEdge]):
        """
        Initialize connectivity analyzer.

        Args:
            nodes: Set of all node IDs
            edges: List of all edges (required + optional)
        """
        self.nodes = nodes
        self.edges = edges
        self.adjacency = build_adjacency_list(edges)

        # Tarjan's algorithm state
        self.index_counter = 0
        self.stack: List[NodeID] = []
        self.lowlinks: Dict[NodeID, int] = {}
        self.index: Dict[NodeID, int] = {}
        self.on_stack: Set[NodeID] = set()
        self.components: List[ConnectedComponent] = []

    def find_strongly_connected_components(self) -> List[ConnectedComponent]:
        """
        Find all strongly connected components using Tarjan's algorithm.

        Returns:
            List of ConnectedComponent objects
        """
        self.components = []
        self.index_counter = 0
        self.stack = []
        self.lowlinks = {}
        self.index = {}
        self.on_stack = set()

        # Run Tarjan's algorithm from each unvisited node
        for node in self.nodes:
            if node not in self.index:
                self._strongconnect(node)

        return self.components

    def _strongconnect(self, node: NodeID) -> None:
        """
        Tarjan's algorithm iterative implementation.

        This is the core SCC detection algorithm (O(V + E) complexity).
        Iterative version to avoid RecursionError on large graphs.
        """
        # Call stack for iterative DFS: (node, successor_iter, is_return_visit)
        call_stack = [(node, 0, False)]

        while call_stack:
            current, succ_idx, is_return = call_stack.pop()

            if not is_return:
                # First visit to this node
                if current in self.index:
                    continue

                # Set the depth index for this node
                self.index[current] = self.index_counter
                self.lowlinks[current] = self.index_counter
                self.index_counter += 1
                self.stack.append(current)
                self.on_stack.add(current)

            # Get successors
            successors = self.adjacency.get(current, [])

            # Process successors starting from succ_idx
            processed_successor = False
            for i in range(succ_idx, len(successors)):
                edge = successors[i]
                successor = edge.to_node

                if successor not in self.index:
                    # Need to visit successor first
                    # Push current back with updated index to continue after
                    call_stack.append((current, i, True))
                    # Push successor to visit
                    call_stack.append((successor, 0, False))
                    processed_successor = True
                    break
                elif successor in self.on_stack:
                    # Successor is in stack and hence in the current SCC
                    self.lowlinks[current] = min(self.lowlinks[current], self.index[successor])

            if processed_successor:
                continue

            # All successors processed or returning from recursive call
            if is_return and call_stack:
                # Update parent's lowlink
                parent_node = call_stack[-1][0] if call_stack else None
                if parent_node and parent_node in self.lowlinks:
                    self.lowlinks[parent_node] = min(
                        self.lowlinks[parent_node],
                        self.lowlinks[current]
                    )

            # If current is a root node, pop the stack and create a new SCC
            if self.lowlinks[current] == self.index[current]:
                component_nodes: Set[NodeID] = set()

                while True:
                    w = self.stack.pop()
                    self.on_stack.remove(w)
                    component_nodes.add(w)
                    if w == current:
                        break

                # Create component with edges
                component = self._create_component(len(self.components), component_nodes)
                self.components.append(component)

    def _create_component(self, component_id: int, nodes: Set[NodeID]) -> ConnectedComponent:
        """Create a ConnectedComponent from a set of nodes."""
        required_edges = []
        all_edges = []

        for edge in self.edges:
            if edge.from_node in nodes and edge.to_node in nodes:
                all_edges.append(edge)
                if edge.required:
                    required_edges.append(edge)

        return ConnectedComponent(
            component_id=component_id,
            nodes=nodes,
            required_edges=required_edges,
            all_edges=all_edges
        )

    def check_feasibility(self, start_node: Optional[NodeID] = None) -> Tuple[bool, str]:
        """
        Check if DRPP is feasible from start_node.

        Args:
            start_node: Starting node (if None, checks general feasibility)

        Returns:
            Tuple of (is_feasible, message)
        """
        components = self.find_strongly_connected_components()

        # Check 1: All required edges should be in same SCC if we care about start_node
        required_edges = [e for e in self.edges if e.required]

        if not required_edges:
            return True, "No required edges to traverse."

        # Find which components contain required edges
        components_with_required = []
        for comp in components:
            if comp.required_edges:
                components_with_required.append(comp)

        if len(components_with_required) == 0:
            return True, "No required edges."

        if len(components_with_required) > 1:
            return False, f"Required edges span {len(components_with_required)} disconnected components. Graph is not strongly connected."

        # Check 2: If start_node specified, ensure it's in the component with required edges
        if start_node is not None:
            target_component = components_with_required[0]
            if start_node not in target_component.nodes:
                return False, f"Start node {start_node} is not in the component containing required edges."

        return True, "Graph is feasible for DRPP."

    def get_reachable_nodes(self, start_node: NodeID) -> Set[NodeID]:
        """
        Get all nodes reachable from start_node via DFS.

        Args:
            start_node: Starting node

        Returns:
            Set of reachable node IDs
        """
        reachable: Set[NodeID] = set()
        stack = [start_node]

        while stack:
            node = stack.pop()
            if node in reachable:
                continue

            reachable.add(node)

            # Add all successors
            for edge in self.adjacency.get(node, []):
                if edge.to_node not in reachable:
                    stack.append(edge.to_node)

        return reachable

    def get_component_summary(self) -> str:
        """
        Generate a human-readable summary of components.

        Returns:
            Formatted string with component statistics
        """
        components = self.find_strongly_connected_components()

        lines = []
        lines.append(f"Found {len(components)} strongly connected component(s):")
        lines.append("")

        for comp in components:
            req_count = len(comp.required_edges)
            total_count = len(comp.all_edges)
            node_count = len(comp.nodes)

            lines.append(f"  Component {comp.component_id}:")
            lines.append(f"    - Nodes: {node_count}")
            lines.append(f"    - Edges: {total_count} ({req_count} required)")

            if req_count > 0:
                lines.append(f"    - Contains REQUIRED edges ⚠️")

        return "\n".join(lines)


def check_graph_connectivity(nodes: Set[NodeID], edges: List[TopologyEdge],
                             start_node: Optional[NodeID] = None) -> Tuple[bool, str, List[ConnectedComponent]]:
    """
    Convenience function to check graph connectivity.

    Args:
        nodes: Set of all node IDs
        edges: List of all edges
        start_node: Optional starting node

    Returns:
        Tuple of (is_feasible, message, components)
    """
    analyzer = ConnectivityAnalyzer(nodes, edges)
    is_feasible, message = analyzer.check_feasibility(start_node)
    components = analyzer.components

    return is_feasible, message, components


def add_connecting_edges(edges: List[TopologyEdge], components: List[ConnectedComponent],
                         shortest_path_func) -> List[TopologyEdge]:
    """
    Add shortest-path edges to connect disconnected components.

    This is used when the graph is not strongly connected but we still want to solve DRPP.
    We add "deadhead" edges to make the graph strongly connected.

    Args:
        edges: Current list of edges
        components: List of strongly connected components
        shortest_path_func: Function(from_node, to_node) -> (path, distance)

    Returns:
        List of new connecting edges to add
    """
    if len(components) <= 1:
        return []  # Already connected

    connecting_edges: List[TopologyEdge] = []

    # For simplicity, connect components in sequence
    # In production, use minimum spanning tree of component graph
    for i in range(len(components) - 1):
        comp1 = components[i]
        comp2 = components[i + 1]

        # Find closest pair of nodes between components
        min_distance = float('inf')
        best_path = None
        best_from = None
        best_to = None

        # Sample a few nodes from each component (for large components)
        sample1 = list(comp1.nodes)[:10] if len(comp1.nodes) > 10 else list(comp1.nodes)
        sample2 = list(comp2.nodes)[:10] if len(comp2.nodes) > 10 else list(comp2.nodes)

        for node1 in sample1:
            for node2 in sample2:
                path, distance = shortest_path_func(node1, node2)
                if path and distance < min_distance:
                    min_distance = distance
                    best_path = path
                    best_from = node1
                    best_to = node2

        # Create connecting edge
        if best_path:
            edge = TopologyEdge(
                edge_id=f"connect_{i}_{i+1}",
                from_node=best_from,
                to_node=best_to,
                cost=min_distance,
                required=False,  # Deadhead edge
                segment_id=None,
                coordinates=best_path,
                metadata={"type": "connecting_edge", "connects": f"comp{i}_to_comp{i+1}"}
            )
            connecting_edges.append(edge)

    return connecting_edges
