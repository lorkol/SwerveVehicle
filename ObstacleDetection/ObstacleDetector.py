
from abc import ABC, abstractmethod
from Types import State


class ObstacleChecker(ABC):
    """Abstract base class for obstacle checking - implement with your obstacle detection."""
    
    @abstractmethod
    def is_collision(self, state: State) -> bool:
        """Check if the given state (x, y, theta) collides with obstacles."""
        pass
    
    @abstractmethod
    def is_path_clear(self, state1: State, state2: State) -> bool:
        """Check if the path between two states is collision-free."""
        pass
