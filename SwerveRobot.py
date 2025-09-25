import numpy as np
from numpy.typing import NDArray

class SwerveRobot:
    def __init__(self, l: float, d: float, r: float):
        self.l = l
        '''Robot length'''
        self.d = d
        '''Robot width'''
        self.r = r
        '''Robot wheel radius'''

    def create_A_Matrix(self, phi1: float, phi2: float, phi3: float, phi4: float) -> NDArray[np.float64]:
        phi1 = np.radians(phi1)
        phi2 = np.radians(phi2)
        phi3 = np.radians(phi3)
        phi4 = np.radians(phi4)

        A: NDArray[np.float64] = np.array([
            [np.cos(phi1), np.sin(phi1), self.l * np.sin(phi1) + self.d * np.cos(phi1)],
            [np.cos(phi2), np.sin(phi2), self.l * np.sin(phi2) - self.d * np.cos(phi2)],
            [np.cos(phi3), np.sin(phi3), -self.l * np.sin(phi3) - self.d * np.cos(phi3)],
            [np.cos(phi4), np.sin(phi4), -self.l * np.sin(phi4) + self.d * np.cos(phi4)]]
        )
        return A