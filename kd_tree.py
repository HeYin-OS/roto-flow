from typing import List

from scipy.spatial import cKDTree
import numpy as np


class BatchKDTree:
    def __init__(self, candidates_on_each_frames: List[np.ndarray]):
        self.frameIdx_to_candidates = candidates_on_each_frames
        self.frameIdx_to_tree = []
        for candidates in self.frameIdx_to_candidates:
            self.frameIdx_to_tree.append(cKDTree(candidates))

    def query(self, frame_idx: int, center: np.ndarray, radius: float) -> np.ndarray:
        ans = self.frameIdx_to_tree[frame_idx].query_ball_point(
            x=center,
            r=radius,
            p=2,
            eps=1e-6,
            workers=-1
        )
        return self.frameIdx_to_candidates[frame_idx][ans]

    # return point lists with [query_len, neighbor_len, 2] with point y-x format
    def query_batch(self, frame_idx: int, centers_yx: np.ndarray, radius: float) -> List[np.ndarray]:
        ans = self.frameIdx_to_tree[frame_idx].query_ball_point(
            x=centers_yx,
            r=radius,
            p=2,
            eps=1e-6,
            workers=-1
        )
        out = []
        for neighbor_indices in ans:
            candidates = self.frameIdx_to_candidates[frame_idx][neighbor_indices]
            out.append(candidates)
        return out


if __name__ == '__main__':
    pass
