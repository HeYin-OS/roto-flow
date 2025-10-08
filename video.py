import os

from PySide6.QtGui import QPixmap
from tqdm import tqdm

from edge_snapping import compute_candidates, EdgeSnappingConfig
from kd_tree import RadiusNP
from raft_predictor import RAFTPredictor
from yaml_reader import YamlUtil
from PIL import Image
import torch
import torchvision.transforms.functional as F

class Video:
    def __init__(self, yaml_url):
        self.url_head: str = YamlUtil.read(yaml_url)['video']['url_head']
        self.frame_num: int = YamlUtil.read(yaml_url)['video']['frame_num']
        self.frame_format: str = YamlUtil.read(yaml_url)['video']['format']

        # init snapping config
        EdgeSnappingConfig.load(self.getFrameImagePath(), 'config/snapping_init.yaml')

        print(f"Reading video frames from designated URL: {self.url_head}*****.{self.frame_format} ({self.frame_num} frames)...")
        self.tensor_format = self.loadFrameSequenceTensor()
        print(f"✓ Pre-loaded frames as tensor, shape: {self.tensor_format.shape}")
        self.qPixmap_format = self.loadFrameSequenceQPixmap()
        print(f"✓ Pre-loaded frames as QPixmap, size: {len(self.qPixmap_format)}")
        self.optical_flow_cache = self.makeOpticalFlowCache()
        print(f"✓ Pre-computed Optical Flow Cache, shape: {len(self.optical_flow_cache.shape)}")
        candidate_on_each_frame = compute_candidates(self.tensor_format)
        self.candidate_trees = RadiusNP(candidate_on_each_frame)
        print(f"✓ Cached candidate points on all frames, shape: {len(candidate_on_each_frame)} of {type(candidate_on_each_frame[0])}")

        self.channel: int = self.tensor_format.shape[1]
        self.height: int = self.tensor_format.shape[2]
        self.width: int = self.tensor_format.shape[3]
        print(f"Video Info: Width {self.width}, Height {self.height}, {self.channel} Channels")

    def loadFrameSequenceTensor(self):
        tensors = []
        for i in range(self.frame_num):
            path = f"{self.url_head}{i:05d}.{self.frame_format}"
            img = Image.open(path)
            tensor = F.to_tensor(img)
            tensors.append(tensor)

        return torch.stack(tensors)

    def loadFrameSequenceQPixmap(self):
        frames = []
        for i in range(self.frame_num):
            path = f"{self.url_head}{i:05d}.{self.frame_format}"
            img = QPixmap(path)
            frames.append(img)

        return frames

    def makeOpticalFlowCache(self):
        strs = self.url_head.split("/")
        cache_rul = "./caches/" + strs[-2] + ".pt"

        if os.path.exists(cache_rul):
            print(f"✓ Loaded optical flow cache from: {cache_rul}")
            return torch.load(cache_rul)

        frames1 = self.tensor_format[:-1]
        frames2 = self.tensor_format[1:]

        flow_list = []

        for i in tqdm(range(frames1.shape[0]), desc="Handling frame batch:", unit="batch"):
            flow = RAFTPredictor().compute_optical_flow(frames1[i], frames2[i])
            flow_list.append(flow)

        result = torch.stack(flow_list, dim=0).squeeze(1)

        torch.save(result, cache_rul)
        print(f"✓ Saved optical flow cache to: {cache_rul}")

        return result

    def getFrameImagePath(self):
        return f"{self.url_head}{0:05d}.{self.frame_format}"