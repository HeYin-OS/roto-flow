from typing import List, Any
import cv2

import numpy as np
import torch
from PIL import Image
from PySide6.QtCore import QPoint
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


def local_snapping(stroke_np_yx: np.ndarray, image_tensor_rgb: Tensor, stroke_point_idx_to_candidates: List[np.ndarray]):
    image_np_rgb = (image_tensor_rgb.permute(1, 2, 0).data.cpu().numpy() * 255).astype(np.uint8)
    image_np_gray = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2GRAY)

    compute_weights(stroke_np_yx, image_np_gray, stroke_point_idx_to_candidates)

    # TODO: complete weight computation


def compute_weights(stroke_np_yx: np.ndarray,
                    image_np_gray: np.ndarray,
                    candidate_points: List[np.ndarray]):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True

    # convert np gray image [H, W] to tensor [B=1, C=1, H, W]
    H = image_np_gray.shape[0]
    W = image_np_gray.shape[1]
    image_np_gray = image_np_gray.astype(np.float32) / 255.0
    image_tensor_gray_gpu = (
        torch.from_numpy(image_np_gray)
        .unsqueeze(0).unsqueeze(0)
        .to(device, non_blocking=True)
        .contiguous()
    )   # shape: [1, 1, H, W]

    print(image_tensor_gray_gpu.shape)

    # convert jagged array to flatten array with index pointer page
    candidates_flatten_xy, flatten_index_ptr = pack_candidates_yx_to_integrity_xy(candidate_points)
    candidate_len = flatten_index_ptr[-1]

    # order to xy and get point number of stroke
    stroke_len = stroke_np_yx.shape[0]
    stroke_xy = stroke_np_yx[:, [1, 0]].astype(np.float32)

    # ready for dp
    # energy -> accumulated energy for each candidate point
    # prev -> the best previous candidate point idx
    energy = np.full(candidate_len, np.inf, dtype=np.float32)
    prev = np.full(candidate_len, -1, dtype=np.float32)

    # accumulated energy for first candidate group is zero
    energy[flatten_index_ptr[0]: flatten_index_ptr[1]] = 0.0

    for i in range(stroke_len - 1):
        # candidate point groups of current stroke point index
        Ui = slice(flatten_index_ptr[i], flatten_index_ptr[i + 1])
        Uj = slice(flatten_index_ptr[i + 1], flatten_index_ptr[i + 2])
        Qi_xy = candidates_flatten_xy[Ui]
        Qj_xy = candidates_flatten_xy[Uj]

        p_i, p_j = stroke_xy[i], stroke_xy[i + 1]

        theta_flatten_gpu = batch_neighbor_candidates_to_reverse_affine(Qi_xy,
                                                                        Qi_xy,
                                                                        H, W).to(device)  # shape: [K_{i} * K_{i+1}, 2, 3]

        grid_gpu = torch.nn.functional.affine_grid(
            theta_flatten_gpu,
            size=[theta_flatten_gpu.shape[0],
                  1,
                  2 * EdgeSnappingConfig.Y_MAX + 1,
                  2 * EdgeSnappingConfig.X_MAX + 1],
            align_corners=False
        ).to(device)  # shape: [K_{i} * K_{i+1}, 2*Y+1, 2*X+1, 2]

        image_affine_and_trimmed = torch.nn.functional.grid_sample(image_tensor_gray_gpu.expand(grid_gpu.shape[0], -1, -1, -1),
                                                                   grid_gpu,
                                                                   mode='bilinear',
                                                                   padding_mode='zeros',
                                                                   align_corners=False).to(device)

        # TODO: compute weights based on trimmed image and do integral

    # temp = np.tensordot(image_np_trimmed,
    #                     EdgeSnappingConfig.fdog_kernel,
    #                     axes=([-1], [-1]))
    # H = np.tensordot(temp,
    #                  EdgeSnappingConfig.gaussian_kernel,
    #                  axes=([1], [0])).squeeze()
    # tilde_H = np.where(H < 0, 1.0 + np.tanh(H), 1.0)


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

    len_i = Qj.shape[0]
    len_j = Qi.shape[0]

    m = 0.5 * (Qi[:, None, :] + Qj[None, :, :])  # [len_i, len_j, 2]
    d = (Qj[None, :, :] - Qi[:, None, :])  # [len_i, len_j, 2]
    L = torch.linalg.norm(d, dim=-1, keepdim=True)  # [len_i, len_j, 1]
    v = d / L.clamp(min=eps)  # [len_i, len_j, 2]

    u = torch.empty_like(v)  # [len_i, len_j, 2]
    u[..., 0] = -v[..., 1]
    u[..., 1] = v[..., 0]

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
    a00 = sx * (X * u[..., 0])
    a01 = sx * (Y * v[..., 0])
    a10 = sy * (X * u[..., 1])
    a11 = sy * (Y * v[..., 1])
    t0 = sx * m[..., 0] + bx
    t1 = sy * m[..., 1] + by

    # build affine matrix - theta
    theta = torch.stack([
        torch.stack([a00, a01, t0], dim=-1),
        torch.stack([a10, a11, t1], dim=-1)
    ], dim=-2)  # [len_i, len_j, 2, 3]

    theta_flat = theta.reshape(len_i * len_j, 2, 3).contiguous()  # [len_i * len_j, 2, 3]

    return theta_flat

# def batch_affine(Qi: np.ndarray, Qj: np.ndarray, eps: np.float32 = 1e-6):
# q0 = stroke_xy[:-1]
# q1 = stroke_xy[1:]

# middle point
# m = np.float32(0.5) * (Qi[:, None, :] + Qj[None, :, :])

# directional vector
# d = (Qj[None, :, :] - Qi[:, None, :]).astype(np.float32)

# construct v
# L = np.linalg.norm(d, axis=2, keepdims=True)
# v = np.divide(d, L, out=np.zeros_like(d), where=L > eps)

# construct u
# u = np.empty_like(v)
# u[:, :, 0] = -v[:, :, 1]
# u[:, :, 1] = v[:, :, 0]
#
# X = EdgeSnappingConfig.X_MAX
# Y = EdgeSnappingConfig.Y_MAX
# t = m - X * u - Y * v

# construct affine matrix

#                             stack along new axis = 3
#                             ----------------------->
# makes [u0, u1], [v0, v1] to [u0 v0]
#                             [u1 v1]
# A = np.stack([u, v], axis=3)

#                                  concatenate along existed axis = 3
#                                  ----------------------->
#             [u0 v0]      [m0]    [u0 v0 m0]
# concatenate [u1 v1] with [m1] to [u1 v1 m1]
# M = np.concatenate([A, t[:, :, :, None]], axis=3)
#
# # inverse affine matrix
# # transpose axis=1 and axis=2
# A_T = np.transpose(A, (0, 2, 1))
#
# # new_p = M @ p + m
# # -----> M @ p = new_p - m
# # -----> p = inv(M) @ new_p - inv(M) @ m
# # let "t_inv" be "- inv(M) @ m"
# # -----> p = inv(M) @ new_p + t_inv
# t_inv = -(A_T @ m[:, :, None])
# M_inv = np.concatenate([A_T, t_inv], axis=2)

# return M


# def batch_wrap_with_affine_and_trim(image_np_gray: np.ndarray, M_affines: np.ndarray):
#     M_affines = M_affines.astype(np.float32)
#     Hwin, Wwin = 2 * EdgeSnappingConfig.Y_MAX + 1, 2 * EdgeSnappingConfig.X_MAX + 1
#
#     out = []
#     for i in range(M_affines.shape[0]):
#         wrapped = cv2.warpAffine(
#             image_np_gray,
#             M_affines[i],
#             dsize=(Wwin, Hwin),
#             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
#             borderMode=cv2.BORDER_CONSTANT,
#             borderValue=0
#         )
#         out.append(wrapped)
#     image_np_affine = np.stack(out, axis=0)
#     return image_np_affine  # [:, :2 * EdgeSnappingConfig.Y_MAX + 1, :2 * EdgeSnappingConfig.X_MAX + 1]
