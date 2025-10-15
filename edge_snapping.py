from typing import List, Any
import cv2
import numba

import numpy as np
import torch
from PIL import Image
from PySide6.QtCore import QPoint
from numpy import ndarray
from torch import Tensor
from yaml_reader import YamlUtil


def getDpi(imgUrl):
    img = Image.open(imgUrl)
    dpi = img.info.get('dpi', (96, 96))
    return dpi


def mm_to_pixels(mm, imgUrl):
    inches = mm / 25.4
    dpi = getDpi(imgUrl)
    pixel_length = dpi[0] * inches
    return pixel_length


def create_gaussian_kernel(size, sigma, direction):
    kernel = cv2.getGaussianKernel(size, sigma, cv2.CV_32F)
    if direction == 0:
        return kernel
    elif direction == 1:
        return kernel.T
    return None


def create_fdog_kernel(size, sigma_c, sigma_s, rho, direction):
    kernel1 = create_gaussian_kernel(size, sigma_c, direction)
    kernel2 = create_gaussian_kernel(size, sigma_s, direction)
    dog_kernel = kernel1 - rho * kernel2
    return dog_kernel


def compute_candidates(frames_tensor_rgb: Tensor):
    out = []
    for i, frame_tensor_rgb in enumerate(frames_tensor_rgb):
        image_np_rgb = (frame_tensor_rgb.permute(1, 2, 0).data.cpu().numpy() * 255).astype(np.uint8)
        image_np_gray = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2GRAY)

        # gradient magnitude
        gx = cv2.Sobel(image_np_gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_DEFAULT)
        gy = cv2.Sobel(image_np_gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_DEFAULT)
        mag = cv2.magnitude(gx, gy)

        # normalization
        mag_norm = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # neighbor max in 3*3 window
        k = np.ones((3, 3), np.uint8)
        k[1, 1] = 0
        nbr_max = cv2.dilate(mag_norm, k)

        # local maximum
        local_max = (mag_norm > nbr_max) & (mag_norm >= float(EdgeSnappingConfig.theta))

        # if i == 0: cv2.imshow('local_max', local_max.astype(np.uint8) * 255)

        # all candidate points on this frame
        ys, xs = np.nonzero(local_max)
        out.append(np.stack([ys, xs], axis=1))
    return out


class EdgeSnappingConfig:
    theta = None
    alpha = None
    beta = None
    sigma_c = None
    sigma_s = None
    sigma_m = None
    rho = None
    X_MAX = None
    Y_MAX = None
    r_s = None
    candidate_num = None
    sampling_num = None

    fdog_kernel = None
    gaussian_kernel = None

    isConfigInit: bool = False

    @staticmethod
    def load(frame_path, config_yaml_path='config/snapping_init.yaml'):
        if EdgeSnappingConfig.isConfigInit:
            return

        settings = YamlUtil.read(config_yaml_path)
        s = settings['snapping']

        EdgeSnappingConfig.theta = s['theta']
        EdgeSnappingConfig.alpha = s['alpha']
        EdgeSnappingConfig.beta = s['beta']
        EdgeSnappingConfig.sigma_c = s['sigma_c']
        EdgeSnappingConfig.sigma_s = s['sigma_s']
        EdgeSnappingConfig.sigma_m = s['sigma_m']
        EdgeSnappingConfig.rho = s['rho']
        EdgeSnappingConfig.X_MAX = s['x']
        EdgeSnappingConfig.Y_MAX = s['y']
        EdgeSnappingConfig.r_s = mm_to_pixels(s['r_s'], frame_path)
        EdgeSnappingConfig.candidate_num = s['candidate_num']
        EdgeSnappingConfig.sampling_num = s['sampling_num']

        EdgeSnappingConfig.isConfigInit = True

        EdgeSnappingConfig.fdog_kernel = create_fdog_kernel(
            2 * EdgeSnappingConfig.X_MAX + 1,
            EdgeSnappingConfig.sigma_c,
            EdgeSnappingConfig.sigma_s,
            EdgeSnappingConfig.rho,
            1
        )
        EdgeSnappingConfig.gaussian_kernel = create_gaussian_kernel(
            2 * EdgeSnappingConfig.Y_MAX + 1,
            EdgeSnappingConfig.sigma_m,
            0
        )


def local_snapping(stroke_np_yx: np.ndarray,
                   image_tensor_rgb: Tensor,
                   candidate_points_yx: List[np.ndarray]):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True

    # convert np gray image [H, W] to tensor [B=1, C=1, H, W]
    H = image_tensor_rgb.shape[0]
    W = image_tensor_rgb.shape[1]
    image_tensor_gray_gpu = rgb_np_to_gray_tensor(device, image_tensor_rgb)

    # convert jagged array to flatten array with index pointer page
    candidates_flatten_xy, flatten_index_ptr = pack_candidates_yx_to_integrity_xy(candidate_points_yx)
    candidate_len = flatten_index_ptr[-1]

    # order to xy and get point number of stroke
    stroke_len = stroke_np_yx.shape[0]
    stroke_xy = stroke_np_yx[:, [1, 0]].astype(np.float32)

    # ready for dp
    # energy -> accumulated energy for each candidate point
    # prev -> the best previous candidate point idx
    energy = np.full(candidate_len, np.inf, dtype=np.float32)
    prev = np.full(candidate_len, -1, dtype=np.int32)

    # accumulated energy for first candidate group is zero
    energy[flatten_index_ptr[0]: flatten_index_ptr[1]] = 0.0

    for i in range(stroke_len - 1):
        # candidate point groups of current stroke point index
        Qi_xy, Qj_xy = slice_candidates_by_index(candidates_flatten_xy, flatten_index_ptr, i)

        # print(f"Qi_xy: {Qi_xy.shape[0]} * Qj_xy: {Qj_xy.shape[0]} = {Qi_xy.shape[0] * Qj_xy.shape[0]}")

        # weights between each two points in two groups
        p_i, p_j = stroke_xy[i], stroke_xy[i + 1]
        weights = compute_weights(H, W,
                                  p_i, p_j,
                                  Qi_xy, Qj_xy,
                                  device,
                                  image_tensor_gray_gpu)  # shape: [K_i, K_j]

        # print(f"weights.shape: {weights.shape} ")

        dp_energy_iteration(i, flatten_index_ptr, energy, prev, weights)

    last_start, last_end = flatten_index_ptr[-2], flatten_index_ptr[-1]
    best_idx = np.argmin(energy[last_start:last_end]) + last_start

    # TODO: add checks of two conditions based on the paper

    best_path_indices = []
    while best_idx != -1:
        best_path_indices.insert(0, best_idx)
        best_idx = prev[best_idx]

    return candidates_flatten_xy[best_path_indices]

@numba.njit
def dp_energy_iteration(i: int, flatten_index_ptr: np.ndarray,
                        energy: np.ndarray,
                        prev: np.ndarray,
                        weights: np.ndarray):
    start_i, end_i = flatten_index_ptr[i], flatten_index_ptr[i + 1]
    start_j, end_j = flatten_index_ptr[i + 1], flatten_index_ptr[i + 2]

    for idx_j in range(end_j - start_j):
        best_prev = -1
        best_energy = np.inf

        for idx_i in range(end_i - start_i):
            bi_energy = energy[start_i + idx_i] + weights[idx_i, idx_j]

            if bi_energy < best_energy:
                best_prev = start_i + idx_i
                best_energy = bi_energy

        energy[start_j + idx_j] = best_energy
        prev[start_j + idx_j] = best_prev


def compute_weights(H: int, W: int,
                    p_i: np.ndarray, p_j: np.ndarray,
                    Qi_xy, Qj_xy,
                    device,
                    image_tensor_gray_gpu: Tensor):
    theta_flatten_gpu = batch_neighbor_candidates_to_reverse_affine(Qi_xy,
                                                                    Qj_xy,
                                                                    H, W).to(device)  # shape: [K_{i} * K_{i+1}, 2, 3]

    grid_gpu = torch.nn.functional.affine_grid(
        theta_flatten_gpu,
        size=[theta_flatten_gpu.shape[0],
              1,
              2 * EdgeSnappingConfig.Y_MAX + 1,
              2 * EdgeSnappingConfig.X_MAX + 1],
        align_corners=False
    ).to(device)  # shape: [K_{i} * K_{i+1}, 2*Y+1, 2*X+1, 2]

    image_affine_and_trimmed = (torch.nn.functional.grid_sample(image_tensor_gray_gpu.expand(grid_gpu.shape[0], -1, -1, -1),
                                                                grid_gpu,
                                                                mode='bilinear',
                                                                padding_mode='zeros',
                                                                align_corners=False)
                                .squeeze(1)  # eliminate dim=1 since len=1 (gray channel)
                                .reshape(Qi_xy.shape[0],
                                         Qj_xy.shape[0],
                                         2 * EdgeSnappingConfig.Y_MAX + 1,
                                         2 * EdgeSnappingConfig.X_MAX + 1)  # return batch size to 2 dims, len of can1 and can2
                                .cpu().numpy())  # change to nparray such that do tensor dot afterward

    # print(f"theta_flatten_gpu: {theta_flatten_gpu.shape}")
    # print(f"grid_gpu.shape = {grid_gpu.shape}")
    # print(f"image_affine_and_trimmed: {image_affine_and_trimmed.shape}")
    # print(f"fdog.shape: {EdgeSnappingConfig.fdog_kernel.shape}")
    # print(f"gaus.shape: {EdgeSnappingConfig.gaussian_kernel.shape}")

    res_dot_on_x = np.tensordot(image_affine_and_trimmed,
                                EdgeSnappingConfig.fdog_kernel.squeeze(),
                                axes=([-1], [0]))

    res_dot_on_x_y = np.tensordot(res_dot_on_x,
                                  EdgeSnappingConfig.gaussian_kernel.squeeze(),
                                  axes=([-1], [0])).squeeze()

    tilde_H_response = np.where(res_dot_on_x_y < 0, 1.0 + np.tanh(res_dot_on_x_y), 1.0)

    # print(f"temp.shape = {res_dot_on_x.shape}, res_dot_on_x_y.shape = {res_dot_on_x_y.shape}, tilde_H.shape = {tilde_H_response.shape}")

    p_diff = (p_j - p_i).astype(np.float32)
    q_diff = Qi_xy.astype(np.float32)[:, None, :] - Qj_xy.astype(np.float32)[None, :, :]
    diff = p_diff.reshape(1, 1, 2) - q_diff
    square_norm = np.sum(diff * diff, axis=-1)
    r_s_square = float(EdgeSnappingConfig.r_s) ** 2

    weights = (square_norm / r_s_square) + EdgeSnappingConfig.alpha * tilde_H_response

    # print(f"p_diff.shape = {p_diff.shape}")
    # print(f"q_diff.shape = {q_diff.shape}")
    # print(f"diff.shape = {diff.shape}")
    # print(f"square_norm.shape = {square_norm.shape}")
    # print(f"weights.shape = {weights.shape}")

    return weights


def slice_candidates_by_index(candidates_flatten_xy: np.ndarray, flatten_index_ptr: np.ndarray, i: int) \
        -> tuple[np.ndarray, np.ndarray]:
    Ui = slice(flatten_index_ptr[i], flatten_index_ptr[i + 1])
    Uj = slice(flatten_index_ptr[i + 1], flatten_index_ptr[i + 2])
    Qi_xy = candidates_flatten_xy[Ui]
    Qj_xy = candidates_flatten_xy[Uj]
    return Qi_xy, Qj_xy


def rgb_np_to_gray_tensor(device, image_tensor_rgb: Tensor) -> Tensor:
    image_np_rgb = (image_tensor_rgb.permute(1, 2, 0).data.cpu().numpy() * 255).astype(np.uint8)
    image_np_gray = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2GRAY)
    image_np_gray = image_np_gray.astype(np.float32) / 255.0
    image_tensor_gray_gpu = (
        torch.from_numpy(image_np_gray)
        .unsqueeze(0).unsqueeze(0)
        .to(device, non_blocking=True)
        .contiguous()
    )  # shape: [1, 1, H, W]
    return image_tensor_gray_gpu


# make jagged array to a integrated long array with index range pointer page
def pack_candidates_yx_to_integrity_xy(candidate_points: List[np.ndarray]):
    # total number of stroke points
    stroke_points_num = len(candidate_points)

    # index range pointer, index_ptr[i] means the start idx such that index_ptr[i+1] - index_ptr[i] means total number of current index
    index_ptr = np.zeros(stroke_points_num + 1, dtype=np.int32)
    for i in range(stroke_points_num):
        index_ptr[i + 1] = index_ptr[i] + (0 if candidate_points[i] is None else len(candidate_points[i]))
    candidates_total_num = index_ptr[-1]

    # flatten candidate points array, dim 1: total candidate number, dim 2: x and y, meanwhile the order of yx is swapped to xy
    candidates_flatten_xy = np.empty((candidates_total_num, 2), dtype=np.float32)
    current_index = 0
    for i in range(stroke_points_num):
        candidate = candidate_points[i]
        if candidate is None or len(candidate) == 0:
            continue
        # make sure it is in float format
        candidate_float = candidate.astype(np.float32)
        # build flatten array and do swapping
        candidates_flatten_xy[current_index: current_index + len(candidate), 0] = candidate_float[:, 1]
        candidates_flatten_xy[current_index: current_index + len(candidate), 1] = candidate_float[:, 0]
        current_index += len(candidate)

    return candidates_flatten_xy, index_ptr


def batch_neighbor_candidates_to_reverse_affine(Qi: np.ndarray,
                                                Qj: np.ndarray,
                                                H: int, W: int,
                                                eps: np.float32 = 1e-6):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Qi = torch.from_numpy(Qi.copy().astype(np.float32)).to(device)
    Qj = torch.from_numpy(Qj.copy().astype(np.float32)).to(device)

    # print(f"new Qi: {Qi.shape}")
    # print(f"new Qj: {Qj.shape}")

    len_i = Qi.shape[0]
    len_j = Qj.shape[0]

    m = 0.5 * (Qi[:, None, :] + Qj[None, :, :])  # [len_i, len_j, 2]
    d = (Qj[None, :, :] - Qi[:, None, :])  # [len_i, len_j, 2]
    L = torch.linalg.norm(d, dim=-1, keepdim=True)  # [len_i, len_j, 1]
    v = d / L.clamp(min=eps)  # [len_i, len_j, 2]

    u = torch.empty_like(v)  # [len_i, len_j, 2]
    u[..., 0] = -v[..., 1]
    u[..., 1] = v[..., 0]

    # print(f"m.shape: {m.shape}")
    # print(f"d.shape: {d.shape}")
    # print(f"L.shape: {L.shape}")
    # print(f"u.shape: {u.shape}")
    # print(f"v.shape: {v.shape}")

    X = float(EdgeSnappingConfig.X_MAX)
    Y = float(EdgeSnappingConfig.Y_MAX)

    # in sample grid, (x_norm. y_norm) means the center of a pixel
    sx = 2.0 / W
    sy = 2.0 / H
    bx = 1.0 / W - 1.0
    by = 1.0 / H - 1.0

    # image coordinates to NDC
    #                       [a00, a01, t0]
    # target affine matrix: [a10, a11, t1] in NDC (for pytorch grid sampling)

    # print(type(sy))
    # print(sy)

    a00 = sx * (X * u[..., 0])
    a01 = sx * (Y * v[..., 0])
    a10 = sy * (X * u[..., 1])
    a11 = sy * (Y * v[..., 1])
    t0 = sx * m[..., 0] + bx
    t1 = sy * m[..., 1] + by

    # print(f"a00.shape: {a00.shape}")
    # print(f"a01.shape: {a01.shape}")
    # print(f"a10.shape: {a10.shape}")
    # print(f"a11.shape: {a11.shape}")
    # print(f"t0.shape: {t0.shape}")
    # print(f"t1.shape: {t1.shape}")

    # build affine matrix - theta
    theta = torch.stack([
        torch.stack([a00, a01, t0], dim=-1),
        torch.stack([a10, a11, t1], dim=-1)
    ], dim=-2)  # [len_i, len_j, 2, 3]

    theta_flat = theta.reshape(len_i * len_j, 2, 3).contiguous()  # [len_i * len_j, 2, 3]

    # print(f"theta.shape: {theta.shape}")
    # print(f"theta_flat.shape: {theta_flat.shape}")

    return theta_flat
