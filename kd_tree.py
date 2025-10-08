from typing import List

import numpy as np

class RadiusNP:
    def __init__(self, candidates_on_each_frames: List[np.ndarray]):
        self.points = candidates_on_each_frames

    def query(self, frame_idx: int, center: np.ndarray, radius: float) -> np.ndarray:
        p = self.points[frame_idx]
        c = np.astype(center, np.float32)
        d2 = np.sum((p - c) ** 2, axis = 1)
        mask = d2 <= radius
        return p[mask]

    def query_batch(self, frame_idx: int, centers: np.ndarray, radius: float):
        p = self.points[frame_idx]
        c = np.astype(centers, np.float32)
        print(p.shape, c.shape)