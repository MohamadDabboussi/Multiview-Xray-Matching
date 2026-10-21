"""View preprocessing used to train the released correspondence checkpoints (internal research code):
a global histogram equalisation (16-bit histogram clipped at 2x its mean, smoothed, CDF mapping),
followed by padding to a square.  The public `src/utils/utils.postprocess_view` uses a 1-99 percentile
stretch + CLAHE instead; the two are NOT equivalent, so checkpoints trained with one should be
evaluated with the same one (see make_gifs.py --preprocess)."""
from __future__ import annotations

import numpy as np
import torch


def equalize_image(image_array: np.ndarray) -> np.ndarray:
    def rectangular_blur(array, kernel_size):
        kernel_size = int(np.floor(kernel_size / 2) * 2 + 1)
        filtered = (np.concatenate([array, np.zeros(kernel_size)]).cumsum()
                    - np.concatenate([np.zeros(kernel_size), array]).cumsum()) / kernel_size
        crop_size = kernel_size // 2
        return filtered[crop_size: -crop_size - 1]

    image_array = np.asarray(image_array, dtype=np.float64)
    image_array = image_array * (2 ** 16 / (image_array.max() - image_array.min() + 1e-7))
    image_array = image_array - image_array.min()
    max_value = int(image_array.max()) + 1
    hist, _ = np.histogram(image_array.flatten(), max_value, (0, max_value))
    hist = np.clip(hist, 0, hist.mean() * 2.0)
    for _ in range(10):
        hist = rectangular_blur(hist, kernel_size=0.01 * max_value)
    hist = hist + hist.mean() * np.arange(len(hist)) / (len(hist) - 1) * 0.4
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype(np.float32)
    return cdf[image_array.astype(int)]


def pad_img_to_max_shape(img: np.ndarray) -> np.ndarray:
    max_shape = max(img.shape)
    padded_img = np.zeros((max_shape, max_shape), dtype=img.dtype) + np.min(img)
    x, y = img.shape
    x_start = (max_shape - x) // 2
    y_start = (max_shape - y) // 2
    padded_img[x_start: x_start + x, y_start: y_start + y] = img
    return padded_img


def postprocess_view(img):
    img = img.squeeze()
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    result = equalize_image(img)
    if len(result.shape) > 2:
        result = np.mean(result, axis=2)
    result = pad_img_to_max_shape(result)
    return torch.from_numpy(result).unsqueeze(0)
