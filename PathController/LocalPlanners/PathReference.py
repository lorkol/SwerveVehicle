from abc import ABC, abstractmethod
import numpy as np


class ReferenceGenerator(ABC):
    @abstractmethod
    """TODO: Class docstring.

    Attributes:
        TODO: describe attributes
    """
    def get_reference_state(self) -> np.ndarray:
        """TODO: Add docstring.

        Args:
            TODO: describe parameters

        Returns:
            TODO: describe return value
        """
        pass
    
    @abstractmethod
    def get_reference_velocity(self) -> np.ndarray:
        """TODO: Add docstring.

        Args:
            TODO: describe parameters

        Returns:
            TODO: describe return value
        """
        pass
    
class SimpleReferenceGenerator(ReferenceGenerator):
    """TODO: Class docstring.

    Attributes:
        TODO: describe attributes
    """
    def __init__(self):
        """TODO: Add docstring.

        Args:
            TODO: describe parameters

        Returns:
            TODO: describe return value
        """
        pass
    
    # TODO: Generalize so it can either get time or current_state
    def get_reference_state(self, t: float) -> np.ndarray:
        """TODO: Add docstring.

        Args:
            TODO: describe parameters

        Returns:
            TODO: describe return value
        """
        return np.array([20.0, 20., 0.0])
    
    def get_reference_velocity(self, t: float) -> np.ndarray:
        """TODO: Add docstring.

        Args:
            TODO: describe parameters

        Returns:
            TODO: describe return value
        """
        return np.array([0.0, 0.0, 0.0])

    def get_reference_acceleration(self, t: float) -> np.ndarray:
        """TODO: Add docstring.

        Args:
            TODO: describe parameters

        Returns:
            TODO: describe return value
        """
        return np.array([0.0, 0.0, 0.0])
