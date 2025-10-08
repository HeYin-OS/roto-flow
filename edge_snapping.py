import dataclasses
from typing import List, Any

import cv2
import numpy as np
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


# image [C, H, W]
def local_snapping(stroke: np.ndarray, image_tensor_rgb: Tensor):

    stroke_np_xy = np.array([[p.x(), p.y()] for p in stroke], dtype=np.float32)

    # candidate_points = compute_candidates(image_np_gray)

    # compute_weights(stroke_np_xy, image_np_gray, candidate_points)


def compute_candidates(frames_tensor_rgb: Tensor):
    out = []
    for frame_tensor_rgb in frames_tensor_rgb:
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

        # cv2.imshow('local_max', local_max.astype(np.uint8) * 255)

        # all candidate points on this frame
        ys, xs = np.nonzero(local_max)
        out.append(np.stack([ys, xs], axis=1))
    return out


def compute_weights(stroke_xy: np.ndarray,
                    image_np_gray: np.ndarray,
                    candidate_points:
                    np.ndarray):
    affine = batch_stroke_to_affine(stroke_xy)

    # do padding on image to make it ready for convolution
    # image_np_gray_padded = np.pad(image_np_gray,
    #                               ((EdgeSnappingConfig.Y_MAX, EdgeSnappingConfig.Y_MAX),
    #                                (EdgeSnappingConfig.X_MAX, EdgeSnappingConfig.X_MAX)),
    #                               mode='constant',
    #                               constant_values=0)
    image_np_trimmed = batch_wrap_with_affine_and_trim(image_np_gray, affine)
    # cv2.imshow('image_np_wrapped', image_np_trimmed[0].astype(np.uint8) * 255)

    temp = np.tensordot(image_np_trimmed,
                        EdgeSnappingConfig.fdog_kernel,
                        axes=([-1], [-1]))
    H = np.tensordot(temp,
                     EdgeSnappingConfig.gaussian_kernel,
                     axes=([1], [0])).squeeze()
    tilde_H = np.where(H < 0, 1.0 + np.tanh(H), 1.0)


def batch_stroke_to_affine(stroke_xy: np.ndarray, eps: float = 1e-6):
    q0 = stroke_xy[:-1]
    q1 = stroke_xy[1:]

    # middle point
    m = 0.5 * (q0 + q1)

    # directional vector
    d = (q1 - q0).astype(np.float32)  # [N-1, 2]

    # construct v
    L = np.linalg.norm(d, axis=1, keepdims=True)  # [N-1, 1]
    v = np.divide(d, L, out=np.zeros_like(d), where=L > eps)

    # construct u
    u = np.empty_like(v)
    u[:, 0] = -v[:, 1]
    u[:, 1] = v[:, 0]

    X = EdgeSnappingConfig.X_MAX
    Y = EdgeSnappingConfig.Y_MAX
    t = m - X * u - Y * v

    # construct affine matrix

    #                             stack along new axis = 2
    #                             ----------------------->
    #                             [u0 v0]
    # makes [u0, u1], [v0, v1] to [u1 v1]
    A = np.stack([u, v], axis=2)

    #             [u0 v0]      [m0]    [u0 v0 m0]
    # concatenate [u1 v1] with [m1] to [u1 v1 m1], None means new axis with dim=1
    M = np.concatenate([A, t[:, :, None]], axis=2)
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

    return M


def batch_wrap_with_affine_and_trim(image_np_gray: np.ndarray, M_fwd: np.ndarray):
    M_fwd = M_fwd.astype(np.float32)
    Hwin, Wwin = 2 * EdgeSnappingConfig.Y_MAX + 1, 2 * EdgeSnappingConfig.X_MAX + 1


    out = []
    for i in range(M_fwd.shape[0]):
        wrapped = cv2.warpAffine(
            image_np_gray,
            M_fwd[i],
            dsize=(Wwin, Hwin),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        out.append(wrapped)
    image_np_affine = np.stack(out, axis=0)
    return image_np_affine#[:, :2 * EdgeSnappingConfig.Y_MAX + 1, :2 * EdgeSnappingConfig.X_MAX + 1]


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
