
from abc import ABC, abstractmethod
from PathController.Types import State_Vector, Control_Vector


class Controller(ABC):
    @abstractmethod
    def get_command(self, state: State_Vector) -> Control_Vector:
        """Compute cost for a given control sequence."""
        pass