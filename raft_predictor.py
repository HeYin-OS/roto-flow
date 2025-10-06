import gc
from typing import Tuple, Literal

import torch
from PySide6.QtCore import QThread, Signal
from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small


class RAFTPredictor:
    def __init__(self, model_size: Literal["large", "small"] = "large"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        weights = None
        if model_size == "large":
            weights = Raft_Large_Weights.DEFAULT
            self.model = raft_large(weights=weights).to(self.device)
        elif model_size == "small":
            weights = Raft_Small_Weights.DEFAULT
            self.model = raft_small(weights=weights).to(self.device)

        self.model.eval()

        self.transforms = weights.transforms()

    def make_divisible_by_8(self, size: Tuple[int, int]) -> Tuple[int, int]:
        width, height = size
        new_width = ((width + 7) // 8) * 8
        new_height = ((height + 7) // 8) * 8
        return new_width, new_height

    def compute_optical_flow(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        if frame1.dim() == 3:
            frame1 = frame1.unsqueeze(0)
        if frame2.dim() == 3:
            frame2 = frame2.unsqueeze(0)
        if frame1.device != self.device:
            frame1 = frame1.to(self.device)
        if frame2.device != self.device:
            frame2 = frame2.to(self.device)

        with torch.no_grad():
            list_of_flows = self.model(frame1, frame2)
            if frame1.dim() == 3:
                list_of_flows = list_of_flows[-1][0]
            else: list_of_flows = list_of_flows[-1]
        if frame1.dim() == 3:
            list_of_flows = list_of_flows.permute(2, 1, 0)
        else:
            list_of_flows = list_of_flows.permute(0, 3, 2, 1)
        return list_of_flows

class RAFTWorker(QThread):
    flow_computed = Signal(torch.Tensor)  # completion signal
    error_occurred = Signal(str)  # error signal
    progress_updated = Signal(int)  # progress update signal

    def __init__(self):
        super().__init__()
        self.frame1 = None
        self.frame2 = None
        self.predictor = None

    def set_frames(self, frame1: torch.Tensor, frame2: torch.Tensor):
        self.frame1 = frame1
        self.frame2 = frame2

    def run(self):
        try:
            self.progress_updated.emit(10)

            if self.predictor is None:
                self.predictor = RAFTPredictor()
                self.progress_updated.emit(30)

            self.progress_updated.emit(50)
            flow_result = self.predictor.compute_optical_flow(self.frame1, self.frame2)
            self.progress_updated.emit(90)

            self.flow_computed.emit(flow_result)
            self.progress_updated.emit(100)

        except Exception as e:
            self.error_occurred.emit(str(e))

# TODO: Optimize Predictor Code