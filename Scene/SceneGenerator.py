"""
Scene Generator for Path Planning
Creates obstacles (circles and polygons) with configurable density and patterns.
Supports sparse random obstacles, dense mazes, and narrow passages.
"""

import json
import random
import math
from typing import List, Dict, Any
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class ObstaclePattern(Enum):
    """Types of obstacle patterns that can be generated."""
    SPARSE = "sparse"           # Few random obstacles
    MODERATE = "moderate"       # Medium density obstacles
    DENSE = "dense"             # High density obstacles
    MAZE = "maze"               # Maze-like structure with narrow passages
    CORRIDOR = "corridor"       # Narrow passages between obstacles


class SceneGenerator:
    """Generates random obstacle scenes for path planning."""
    
    def __init__(self, map_length: float = 100.0, map_width: float = 50.0):
        """
        Initialize the scene generator.
        
        Args:
            map_length: Length of the map (x-axis)
            map_width: Width of the map (y-axis)
        """
        self.map_length = map_length
        self.map_width = map_width
        self.obstacles: List[Dict[str, Any]] = []
    
    def generate_scene(self, pattern: ObstaclePattern = ObstaclePattern.MODERATE, 
                      num_obstacles: int = 5, narrow_passage: bool = False, seed: int = None) -> List[Dict[str, Any]]:
        """
        Generate a scene with obstacles.
        
        Args:
            pattern: Type of obstacle pattern (sparse, moderate, dense, maze, corridor)
            num_obstacles: Number of obstacles to generate
            narrow_passage: Whether to include narrow passages between obstacles
            seed: Random seed for reproducibility
            
        Returns:
            List of obstacle dictionaries
        """
        if seed is not None:
            random.seed(seed)
        
        self.obstacles.clear()
        
        if pattern == ObstaclePattern.SPARSE:
            self._generate_sparse(num_obstacles)
        elif pattern == ObstaclePattern.MODERATE:
            self._generate_moderate(num_obstacles)
        elif pattern == ObstaclePattern.DENSE:
            self._generate_dense(num_obstacles)
        elif pattern == ObstaclePattern.MAZE:
            self._generate_maze(num_obstacles, narrow_passage)
        elif pattern == ObstaclePattern.CORRIDOR:
            self._generate_corridor(num_obstacles)
        
        return self.obstacles
    
    def _generate_sparse(self, num_obstacles: int) -> None:
        """Generate sparse random obstacles (few circles)."""
        for _ in range(num_obstacles):
            # Random circles
            if random.random() < 0.7:
                x = random.uniform(5, self.map_length - 5)
                y = random.uniform(5, self.map_width - 5)
                radius = random.uniform(1.5, 3.0)
                
                self.obstacles.append({
                    "Shape": "Circle",
                    "Center": [x, y],
                    "Radius": radius
                })
    
    def _generate_moderate(self, num_obstacles: int) -> None:
        """Generate moderate density obstacles (mix of circles and small polygons)."""
        for i in range(num_obstacles):
            if random.random() < 0.6:
                # Circle
                x = random.uniform(5, self.map_length - 5)
                y = random.uniform(5, self.map_width - 5)
                radius = random.uniform(1.0, 2.5)
                
                self.obstacles.append({
                    "Shape": "Circle",
                    "Center": [x, y],
                    "Radius": radius
                })
            else:
                # Small polygon
                x = random.uniform(5, self.map_length - 5)
                y = random.uniform(5, self.map_width - 5)
                width = random.uniform(1.5, 3.0)
                height = random.uniform(1.5, 3.0)
                
                points = [
                    [x, y],
                    [x + width, y],
                    [x + width, y + height],
                    [x, y + height]
                ]
                
                self.obstacles.append({
                    "Shape": "Polygon",
                    "Points": points
                })
    
    def _generate_dense(self, num_obstacles: int) -> None:
        """Generate dense obstacles (high density, mostly polygons)."""
        grid_cols = int(math.sqrt(num_obstacles * 1.5))
        grid_rows = int(self.map_width / (self.map_length / grid_cols))
        
        cell_width = self.map_length / grid_cols
        cell_height = self.map_width / grid_rows
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                if random.random() < 0.7:
                    # Place obstacle in this cell
                    x_base = col * cell_width + cell_width * 0.1
                    y_base = row * cell_height + cell_height * 0.1
                    
                    if random.random() < 0.4:
                        # Circle
                        x = x_base + random.uniform(0, cell_width * 0.8)
                        y = y_base + random.uniform(0, cell_height * 0.8)
                        radius = min(cell_width, cell_height) * 0.3
                        
                        self.obstacles.append({
                            "Shape": "Circle",
                            "Center": [x, y],
                            "Radius": radius
                        })
                    else:
                        # Polygon
                        x = x_base
                        y = y_base
                        width = cell_width * 0.7
                        height = cell_height * 0.7
                        
                        points = [
                            [x, y],
                            [x + width, y],
                            [x + width, y + height],
                            [x, y + height]
                        ]
                        
                        self.obstacles.append({
                            "Shape": "Polygon",
                            "Points": points
                        })
    
    def _generate_maze(self, num_obstacles: int, narrow_passage: bool = False) -> None:
        """Generate maze-like structure with corridors."""
        # Create a denser grid-based maze with walls
        grid_size = 5 if narrow_passage else 4  # Denser grid for narrow passages
        cell_width = self.map_length / grid_size
        cell_height = self.map_width / grid_size
        
        wall_thickness = 1.0
        passage_width = 1.2 if narrow_passage else 2.0  # Very narrow passages
        
        # Generate horizontal and vertical walls with passages
        # Use higher probability to create actual maze structure
        for row in range(grid_size):
            for col in range(grid_size):
                if random.random() < 0.85:  # 85% chance to place wall (much denser)
                    x = col * cell_width
                    y = row * cell_height
                    
                    # Create wall cells with optional passages
                    if random.random() < 0.5:
                        # Horizontal wall (spans across)
                        wall_start = x
                        wall_end = x + cell_width
                        wall_y = y + cell_height / 2 - wall_thickness / 2
                        
                        if narrow_passage and random.random() < 0.5:
                            # Create wall with a passage through the middle
                            passage_x = x + cell_width / 2 - passage_width / 2
                            
                            # Left wall segment
                            points_left = [
                                [wall_start, wall_y],
                                [passage_x, wall_y],
                                [passage_x, wall_y + wall_thickness],
                                [wall_start, wall_y + wall_thickness]
                            ]
                            self.obstacles.append({
                                "Shape": "Polygon",
                                "Points": points_left
                            })
                            
                            # Right wall segment
                            points_right = [
                                [passage_x + passage_width, wall_y],
                                [wall_end, wall_y],
                                [wall_end, wall_y + wall_thickness],
                                [passage_x + passage_width, wall_y + wall_thickness]
                            ]
                            self.obstacles.append({
                                "Shape": "Polygon",
                                "Points": points_right
                            })
                        else:
                            # Solid horizontal wall
                            points = [
                                [wall_start, wall_y],
                                [wall_end, wall_y],
                                [wall_end, wall_y + wall_thickness],
                                [wall_start, wall_y + wall_thickness]
                            ]
                            self.obstacles.append({
                                "Shape": "Polygon",
                                "Points": points
                            })
                    else:
                        # Vertical wall (spans across)
                        wall_start = y
                        wall_end = y + cell_height
                        wall_x = x + cell_width / 2 - wall_thickness / 2
                        
                        if narrow_passage and random.random() < 0.5:
                            # Create wall with a passage through the middle
                            passage_y = y + cell_height / 2 - passage_width / 2
                            
                            # Bottom wall segment
                            points_bottom = [
                                [wall_x, wall_start],
                                [wall_x + wall_thickness, wall_start],
                                [wall_x + wall_thickness, passage_y],
                                [wall_x, passage_y]
                            ]
                            self.obstacles.append({
                                "Shape": "Polygon",
                                "Points": points_bottom
                            })
                            
                            # Top wall segment
                            points_top = [
                                [wall_x, passage_y + passage_width],
                                [wall_x + wall_thickness, passage_y + passage_width],
                                [wall_x + wall_thickness, wall_end],
                                [wall_x, wall_end]
                            ]
                            self.obstacles.append({
                                "Shape": "Polygon",
                                "Points": points_top
                            })
                        else:
                            # Solid vertical wall
                            points = [
                                [wall_x, wall_start],
                                [wall_x + wall_thickness, wall_start],
                                [wall_x + wall_thickness, wall_end],
                                [wall_x, wall_end]
                            ]
                            self.obstacles.append({
                                "Shape": "Polygon",
                                "Points": points
                            })
    
    def _generate_corridor(self, num_obstacles: int) -> None:
        """Generate obstacles with narrow corridors/passages - randomized!"""
        corridor_width = random.uniform(1.2, 1.5)  # Very narrow & randomized
        obstacle_width = random.uniform(3.5, 5.0)  # Randomized width
        
        num_rows = num_obstacles // 2
        row_spacing = self.map_length / (num_rows + 1)
        
        for i in range(num_rows):
            x = (i + 1) * row_spacing
            # Randomize the center position and height
            obstacle_height = self.map_width / random.uniform(2.5, 3.5)  # Randomized height
            y = random.uniform(0, self.map_width - obstacle_height - corridor_width)  # Random vertical position
            
            # Add some randomization to obstacle width and position
            left_width = obstacle_width + random.uniform(-0.5, 0.5)
            left_x_offset = random.uniform(-1, 1)
            
            # Create two obstacles with a narrow gap in the middle for a corridor
            # Left obstacle
            left_points = [
                [x - left_width + left_x_offset, 0],
                [x + random.uniform(0.5, 1.5) + left_x_offset, 0],
                [x + random.uniform(0.5, 1.5) + left_x_offset, y],
                [x - left_width + left_x_offset, y]
            ]
            
            self.obstacles.append({
                "Shape": "Polygon",
                "Points": left_points
            })
            
            # Right obstacle (starting after the narrow corridor)
            right_width = obstacle_width + random.uniform(-0.5, 0.5)
            right_x_offset = random.uniform(-1, 1)
            
            right_points = [
                [x - right_width + right_x_offset, y + obstacle_height + corridor_width],
                [x + random.uniform(0.5, 1.5) + right_x_offset, y + obstacle_height + corridor_width],
                [x + random.uniform(0.5, 1.5) + right_x_offset, self.map_width],
                [x - right_width + right_x_offset, self.map_width]
            ]
            
            self.obstacles.append({
                "Shape": "Polygon",
                "Points": right_points
            })
    
    def save_to_json(self, filepath: str, map_length: float = None, map_width: float = None) -> None:
        """
        Save generated obstacles to JSON file in Configuration.json format.
        
        Args:
            filepath: Path to save the JSON file
            map_length: Map length (uses existing if None)
            map_width: Map width (uses existing if None)
        """
        if map_length is None:
            map_length = self.map_length
        if map_width is None:
            map_width = self.map_width
        
        config = {
            "Map": {
                "Dimensions": {
                    "Length": map_length,
                    "Width": map_width
                },
                "FrictionCoefficient": 0.8,
                "Obstacles": self.obstacles
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"✓ Scene saved to {filepath}")
        print(f"  Generated {len(self.obstacles)} obstacles")
    
    def visualize(self, title: str = "Generated Scene", show_grid: bool = True) -> None:
        """
        Visualize the generated scene.
        
        Args:
            title: Title for the plot
            show_grid: Whether to show a grid overlay
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Set map boundaries
        ax.set_xlim(0, self.map_length)
        ax.set_ylim(0, self.map_width)
        ax.set_aspect('equal')
        
        # Draw map boundary
        boundary = patches.Rectangle(
            (0, 0), self.map_length, self.map_width,
            linewidth=2, edgecolor='black', facecolor='lightgray', 
            alpha=0.1, label='Map Boundary'
        )
        ax.add_patch(boundary)
        
        # Draw grid if requested
        if show_grid:
            grid_spacing = 10.0
            for x in range(0, int(self.map_length) + 1, int(grid_spacing)):
                ax.axvline(x, color='gray', linewidth=0.5, alpha=0.3, linestyle='--')
            for y in range(0, int(self.map_width) + 1, int(grid_spacing)):
                ax.axhline(y, color='gray', linewidth=0.5, alpha=0.3, linestyle='--')
        
        # Draw obstacles
        for i, obstacle in enumerate(self.obstacles):
            if obstacle["Shape"] == "Circle":
                circle = patches.Circle(
                    (obstacle["Center"][0], obstacle["Center"][1]),
                    obstacle["Radius"],
                    linewidth=2, edgecolor='red', facecolor='red', 
                    alpha=0.4, label='Obstacles' if i == 0 else ''
                )
                ax.add_patch(circle)
            
            elif obstacle["Shape"] == "Polygon":
                polygon = patches.Polygon(
                    obstacle["Points"],
                    linewidth=2, edgecolor='red', facecolor='red', 
                    alpha=0.4, label='Obstacles' if i == 0 else ''
                )
                ax.add_patch(polygon)
        
        # Add start and goal markers for reference
        ax.plot(5, 5, 'go', markersize=12, label='Start (example)', zorder=10)
        ax.plot(self.map_length - 5, self.map_width - 5, 'r*', markersize=20, 
                label='Goal (example)', zorder=10)
        
        # Labels and legend
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_title(f"{title} ({len(self.obstacles)} obstacles)", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.show()


def main():
    """Example usage of the scene generator."""
    import sys
    
    # Configuration
    map_length = 100.0
    map_width = 50.0
    output_path = "Scene/Configuration.json"
    
    # Print menu
    print("\n" + "="*60)
    print("Scene Generator for Path Planning")
    print("="*60)
    print("\nSelect obstacle pattern:")
    print("1. Sparse (few random obstacles)")
    print("2. Moderate (medium density mix)")
    print("3. Dense (high density obstacles)")
    print("4. Maze (maze-like structure)")
    print("5. Corridor (narrow passages)")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    pattern_map = {
        "1": (ObstaclePattern.SPARSE, 5),
        "2": (ObstaclePattern.MODERATE, 10),
        "3": (ObstaclePattern.DENSE, 15),
        "4": (ObstaclePattern.MAZE, 12),
        "5": (ObstaclePattern.CORRIDOR, 6)
    }
    
    if choice not in pattern_map:
        print("Invalid choice!")
        return
    
    pattern, default_num = pattern_map[choice]
    
    num_obstacles = input(f"Enter number of obstacles (default {default_num}): ").strip()
    num_obstacles = int(num_obstacles) if num_obstacles else default_num
    
    narrow_passage = False
    if pattern in [ObstaclePattern.MAZE, ObstaclePattern.CORRIDOR]:
        narrow = input("Use narrow passages? (y/n, default n): ").strip().lower()
        narrow_passage = narrow == 'y'
    
    # Generate scene
    print("\nGenerating scene...")
    generator = SceneGenerator(map_length=map_length, map_width=map_width)
    generator.generate_scene(pattern=pattern, num_obstacles=num_obstacles, narrow_passage=narrow_passage)
    
    # Display statistics
    print(f"\n✓ Generated scene with {len(generator.obstacles)} obstacles")
    print(f"  Pattern: {pattern.value}")
    
    
    # Visualize
    vis = input("Visualize scene? (y/n, default y): ").strip().lower()
    if vis != 'n':
        generator.visualize(title=f"Generated Scene - {pattern.value.capitalize()}")

    # Save to JSON
    save = input("\nSave to JSON? (y/n, default y): ").strip().lower()
    if save != 'n':
        generator.save_to_json(output_path, map_length=map_length, map_width=map_width)

if __name__ == "__main__":
    main()
