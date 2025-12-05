from abc import ABC, abstractmethod
from typing import List, Optional
from Types import State
from dataclasses import dataclass


class Planner(ABC):
    """Abstract base class for path planners."""
    
    @abstractmethod
    def plan(self, start: State, goal: State) -> Optional[List[State]]:
        """Plan a path from start to goal. Return list of states or None if no path found."""
        pass
    
    

@dataclass
class Node:
    """Represents a state node in the Hybrid A* search tree."""
    state: State  # (x, y, theta)
    g_cost: float  # Cost from start
    h_cost: float  # Heuristic cost to goal
    parent: Optional["Node"] = None
    
    @property
    def f_cost(self) -> float:
        """Total estimated cost: g + h"""
        return self.g_cost + self.h_cost
    
    def __lt__(self, other: "Node") -> bool:
        """For heap ordering - lower f_cost has priority."""
        return self.f_cost < other.f_cost
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self.state == other.state

