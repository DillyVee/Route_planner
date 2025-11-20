"""
Turn-by-turn navigation directions generator for DRPP routes.

This module generates human-readable driving instructions from DRPP solutions.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import csv
import json
import math

from .industry_drpp_solver import DRPPSolution
from .topology import TopologyEdge, Coordinate


@dataclass
class TurnInstruction:
    """A single turn-by-turn instruction."""
    step_number: int
    instruction: str
    distance_m: float
    cumulative_distance_m: float
    segment_id: Optional[str]
    is_required: bool
    metadata: Dict


class TurnByTurnGenerator:
    """
    Generate turn-by-turn navigation instructions from DRPP solution.
    """

    def __init__(self, solution: DRPPSolution):
        """
        Initialize turn-by-turn generator.

        Args:
            solution: Complete DRPP solution
        """
        self.solution = solution

    def generate_instructions(self) -> List[TurnInstruction]:
        """
        Generate turn-by-turn instructions from route.

        Returns:
            List of TurnInstruction objects
        """
        instructions = []
        cumulative_distance = 0.0

        for i, edge in enumerate(self.solution.tour.edges):
            step_number = i + 1

            # Generate instruction text
            instruction = self._generate_instruction_text(edge, i)

            # Create instruction
            turn_instruction = TurnInstruction(
                step_number=step_number,
                instruction=instruction,
                distance_m=edge.cost,
                cumulative_distance_m=cumulative_distance + edge.cost,
                segment_id=edge.segment_id,
                is_required=edge.required,
                metadata=edge.metadata.copy()
            )

            instructions.append(turn_instruction)
            cumulative_distance += edge.cost

        return instructions

    def _generate_instruction_text(self, edge: TopologyEdge, index: int) -> str:
        """
        Generate human-readable instruction for an edge.

        Args:
            edge: Topology edge
            index: Index in route

        Returns:
            Instruction text
        """
        # Check if this is a required or deadhead segment
        if edge.required:
            # Required segment - provide detailed instruction
            route_name = edge.metadata.get('RouteName', 'road')
            direction = edge.metadata.get('Dir', '')

            if direction:
                instruction = f"Continue on {route_name} {direction}"
            else:
                instruction = f"Continue on {route_name}"

            # Add distance
            distance_km = edge.cost / 1000.0
            if distance_km >= 1.0:
                instruction += f" for {distance_km:.2f} km"
            else:
                instruction += f" for {edge.cost:.0f} m"

        else:
            # Deadhead segment
            if edge.metadata.get('type') == 'balancing_edge':
                instruction = f"Travel to next required segment"
            else:
                instruction = f"Travel via connecting route"

            distance_km = edge.cost / 1000.0
            if distance_km >= 1.0:
                instruction += f" ({distance_km:.2f} km)"
            else:
                instruction += f" ({edge.cost:.0f} m)"

        return instruction

    def export_to_csv(self, output_path: str) -> None:
        """
        Export turn-by-turn instructions to CSV file.

        Args:
            output_path: Path to output CSV file
        """
        instructions = self.generate_instructions()

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Step',
                'Instruction',
                'Distance (m)',
                'Distance (km)',
                'Cumulative (km)',
                'Segment ID',
                'Required',
                'Route Name',
                'Direction'
            ])

            # Data rows
            for inst in instructions:
                writer.writerow([
                    inst.step_number,
                    inst.instruction,
                    f"{inst.distance_m:.2f}",
                    f"{inst.distance_m / 1000:.3f}",
                    f"{inst.cumulative_distance_m / 1000:.3f}",
                    inst.segment_id or '',
                    'Yes' if inst.is_required else 'No',
                    inst.metadata.get('RouteName', ''),
                    inst.metadata.get('Dir', '')
                ])

    def export_to_json(self, output_path: str) -> None:
        """
        Export turn-by-turn instructions to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        instructions = self.generate_instructions()

        # Convert to JSON-serializable format
        instructions_data = []
        for inst in instructions:
            instructions_data.append({
                'step': inst.step_number,
                'instruction': inst.instruction,
                'distance_m': round(inst.distance_m, 2),
                'distance_km': round(inst.distance_m / 1000, 3),
                'cumulative_km': round(inst.cumulative_distance_m / 1000, 3),
                'segment_id': inst.segment_id,
                'is_required': inst.is_required,
                'metadata': inst.metadata
            })

        # Create summary
        summary = {
            'total_steps': len(instructions),
            'total_distance_km': round(self.solution.total_distance_km, 2),
            'required_distance_km': round(self.solution.required_distance_km, 2),
            'deadhead_distance_km': round(self.solution.deadhead_distance_km, 2),
            'deadhead_percentage': round(self.solution.deadhead_percentage, 1),
            'is_valid': self.solution.is_valid
        }

        # Combine
        output = {
            'summary': summary,
            'instructions': instructions_data
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

    def export_to_text(self, output_path: str) -> None:
        """
        Export turn-by-turn instructions to human-readable text file.

        Args:
            output_path: Path to output text file
        """
        instructions = self.generate_instructions()

        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("DRPP ROUTE - TURN-BY-TURN DIRECTIONS\n")
            f.write("=" * 80 + "\n\n")

            # Summary
            f.write(f"Total Distance: {self.solution.total_distance_km:.2f} km\n")
            f.write(f"Required Distance: {self.solution.required_distance_km:.2f} km\n")
            f.write(f"Deadhead Distance: {self.solution.deadhead_distance_km:.2f} km ")
            f.write(f"({self.solution.deadhead_percentage:.1f}%)\n\n")

            f.write("=" * 80 + "\n")
            f.write("DIRECTIONS\n")
            f.write("=" * 80 + "\n\n")

            # Instructions
            for inst in instructions:
                f.write(f"{inst.step_number}. {inst.instruction}\n")

                # Add details for required segments
                if inst.is_required and inst.segment_id:
                    f.write(f"   Segment: {inst.segment_id}\n")

                f.write(f"   Distance: {inst.distance_m / 1000:.2f} km ")
                f.write(f"(Total: {inst.cumulative_distance_m / 1000:.2f} km)\n")
                f.write("\n")


def generate_turn_by_turn(solution: DRPPSolution, output_csv: Optional[str] = None,
                          output_json: Optional[str] = None, output_text: Optional[str] = None) -> List[TurnInstruction]:
    """
    Convenience function to generate turn-by-turn directions.

    Args:
        solution: DRPP solution
        output_csv: Optional path to CSV output
        output_json: Optional path to JSON output
        output_text: Optional path to text output

    Returns:
        List of TurnInstruction objects
    """
    generator = TurnByTurnGenerator(solution)

    if output_csv:
        generator.export_to_csv(output_csv)

    if output_json:
        generator.export_to_json(output_json)

    if output_text:
        generator.export_to_text(output_text)

    return generator.generate_instructions()
