import os
import cv2
import torch
import random
import pydicom
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms
from scipy.ndimage import zoom, shift
from skimage import exposure


class NamedLambda(transforms.Lambda):
    def __init__(self, func, name):
        super().__init__(func)
        self.name = name


def sigmoid(x, x0=0, k=1):
    return 1 / (1 + np.exp(-k * (x - x0)))


def equalize_image(image_array):
    if len(image_array.shape) != 2:
        # Convert to grayscale
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    low, high = np.percentile(image_array, [1, 99])
    image_clipped = np.clip(image_array, low, high)
    image_array = (image_clipped - low) / (high - low + 1e-8)
    image_array = image_array.clip(0, 1)
    return exposure.equalize_adapthist(image_array, clip_limit=0.001)


def interpolate_pos_encoding(
    position_embeddings: torch.Tensor, height: int, width: int, patch_size: int = 16
) -> torch.Tensor:
    """
    This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
    images. This method is also adapted to support torch.jit tracing.

    Adapted from:
    - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
    - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
    """

    num_positions = position_embeddings.shape[1] - 1

    class_pos_embed = position_embeddings[:, :1]
    patch_pos_embed = position_embeddings[:, 1:]

    dim = position_embeddings.shape[-1]

    new_height = int(height // patch_size)
    new_width = int(width // patch_size)

    sqrt_num_positions = num_positions**0.5
    sqrt_num_positions = (
        sqrt_num_positions.to(torch.int64)
        if torch.jit.is_tracing() and isinstance(sqrt_num_positions, torch.Tensor)
        else int(sqrt_num_positions)
    )
    patch_pos_embed = patch_pos_embed.reshape(
        1, sqrt_num_positions, sqrt_num_positions, dim
    )
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed,
        size=(new_height, new_width),
        mode="bicubic",
        align_corners=False,
    )

    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

    return torch.cat((class_pos_embed, patch_pos_embed), dim=1)


def pad_img_to_max_shape(img):
    max_shape = max(img.shape)
    padded_img = np.zeros((max_shape, max_shape), dtype=img.dtype) + np.min(img)
    x, y = img.shape  # img.shape[1], img.shape[0]
    x_start = (max_shape - x) // 2
    y_start = (max_shape - y) // 2
    padded_img[x_start : x_start + x, y_start : y_start + y] = img
    return padded_img


def postprocess_view(img):
    img = img.squeeze()
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    result = equalize_image(img)
    if len(result.shape) > 2:
        result = np.mean(result, axis=2)
    h, w = result.shape
    result = pad_img_to_max_shape(result)
    result = torch.from_numpy(result).unsqueeze(0)
    return result


# Vizualization functions


def get_random_patch_pairs(corr_matrix, num_matches=100, patch_size=16, threshold=0.3):
    dim_1, dim_2 = corr_matrix.shape
    dim_1 = int(np.sqrt(dim_1))
    dim_2 = int(np.sqrt(dim_2))

    # Get all indices where the correspondence value is above the threshold
    valid_indices = np.argwhere(corr_matrix > threshold)

    # If the number of valid matches is less than the desired number, adjust num_matches
    if len(valid_indices) < num_matches:
        num_matches = len(valid_indices)

    # Randomly select the specified number of valid matches
    selected_indices = valid_indices[
        np.random.choice(len(valid_indices), num_matches, replace=False)
    ]

    patch_pairs = []
    for idx, idx_2 in selected_indices:
        i_curr = idx // dim_1
        j_curr = idx % dim_1
        k = idx_2 // dim_2
        l = idx_2 % dim_2
        patch_1_coord = (i_curr, j_curr)
        patch_2_coord = (k, l)
        patch_pairs.append((patch_1_coord, patch_2_coord))

    return patch_pairs

def get_random_patch_pairs(corr_matrix, num_matches=100, threshold=0.3, patch_size=16):
    dim_1, dim_2 = corr_matrix.shape
    dim_1 = int(np.sqrt(dim_1))
    dim_2 = int(np.sqrt(dim_2))

    total_rows = dim_1 * dim_1
    if num_matches > total_rows:
        num_matches = total_rows

    selected_rows = np.random.choice(total_rows, num_matches, replace=False)

    patch_pairs = []

    for idx in selected_rows:
        i_curr = idx // dim_1
        j_curr = idx % dim_1

        for idx_2 in range(dim_2 * dim_2):
            if corr_matrix[idx, idx_2] > threshold:
                k = idx_2 // dim_2
                l = idx_2 % dim_2
                patch_1_coord = (i_curr, j_curr)
                patch_2_coord = (k, l)
                patch_pairs.append((patch_1_coord, patch_2_coord))

    return patch_pairs


def get_specified_patch_pairs(corr_matrix, indices, patch_size=16, threshold=0.3):
    dim_1, dim_2 = corr_matrix.shape
    dim_1 = int(np.sqrt(dim_1))
    dim_2 = int(np.sqrt(dim_2))

    patch_pairs = []

    for idx in indices:
        if idx >= dim_1 * dim_1:
            continue  # Skip invalid indices
        i_curr = idx // dim_1
        j_curr = idx % dim_1
        for idx_2 in range(dim_2 * dim_2):
            if corr_matrix[idx, idx_2] > threshold:
                k = idx_2 // dim_2
                l = idx_2 % dim_2
                patch_1_coord = (i_curr, j_curr)
                patch_2_coord = (k, l)
                patch_pairs.append((patch_1_coord, patch_2_coord))

    return patch_pairs


def visualize_patch_pairs(image1, image2, patch_pairs, patch_size=16):
    # Create a canvas large enough to place both images diagonally
    image1 = equalize_image(image1)
    image2 = equalize_image(image2)
    if image1.dtype == np.float64:
        image1 = image1.astype(np.float32)
    if image2.dtype == np.float64:
        image2 = image2.astype(np.float32)
    if len(image1.shape) == 2:
        image1_rgb = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
        image2_rgb = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    height1, width1, _ = image1_rgb.shape
    height2, width2, _ = image2_rgb.shape

    # Create a canvas that accommodates both images diagonally
    canvas_height = height1 + 50
    canvas_width = width1 + width2
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)

    # Place the first image at the top-left corner
    canvas[:height1, :width1, :] = image1_rgb

    # Place the second image at the bottom-right corner
    canvas[50:, width1:, :] = image2_rgb

    # Generate a list of unique colors for each row in the patch matrix
    colors = {}

    # Plot the matching lines and circles
    for (i_curr, j_curr), (k, l) in patch_pairs:
        # Choose a unique color for each row
        if i_curr not in colors:
            # Generate a random color for the row
            colors[i_curr] = tuple(np.random.randint(0, 256, 3).tolist())

        # Define the color for this row
        color = (0, 255, 0)  # colors[i_curr]

        # Calculate the center of the patch in both images
        pt1 = (
            j_curr * patch_size + patch_size // 2,
            i_curr * patch_size + patch_size // 2,
        )
        pt2 = (
            l * patch_size + patch_size // 2 + width1,
            k * patch_size + patch_size // 2 + 50,
        )

        # Draw the line for correspondence
        cv2.line(canvas, pt1, pt2, color, 1)

        # Draw circles at both ends of the line to mark the correspondence
        cv2.circle(canvas, pt1, radius=3, color=color, thickness=-1)  # Circle on image1
        cv2.circle(canvas, pt2, radius=3, color=color, thickness=-1)  # Circle on image2

    # Display the result
    plt.figure(
        figsize=(12, 12)
    )  # Increased the figure size to accommodate the diagonal layout
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def read_csv_all(dir, mode=None):
    df = pd.read_csv(dir)
    if mode is not None:
        df_ = df[df["train_val"] == mode]
    else:
        df_ = df
    return df_[
        [
            "image_path",
            "type",
            "angles",
            "translation_x",
            "translation_y",
            "sid",
            "sad",
            "train_val",
            "organ_name",
            "num_split_on_z",
            "y_split",
        ]
    ]


def preprocess(ds):
    pixel_value = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
    # pixel_value = np.maximum(pixel_value, ds.RescaleIntercept)
    if pixel_value.shape[0] != pixel_value.shape[1]:
        a, b = pixel_value.shape
        if a > b:
            pad = ((0, 0), ((a - b) // 2, (a - b) // 2))
        else:
            pad = (((b - a) // 2, (b - a) // 2), (0, 0))
        pixel_value = np.pad(pixel_value, pad, mode="constant", constant_values=0)
    return pixel_value


def read_volume(vol_dir):
    img_dir = os.path.abspath(vol_dir)
    img_dir = Path(img_dir)
    frames = img_dir.glob("*")
    # Create the 3D volume
    volume = []
    del_zs = []
    for j, f in enumerate(frames):
        ds = pydicom.dcmread(f, force=True)
        img = preprocess(ds)
        volume.append(img)
        del_zs.append(ds.ImagePositionPatient[2])
    # Sort the volumes based on the z position
    sorted_indices = np.argsort(del_zs)
    volume = [volume[i] for i in sorted_indices]
    volume = np.array(volume)
    del_zs = np.diff(np.sort(del_zs))
    del_z = np.abs(np.unique(del_zs)[0])
    del_x, del_y = ds.PixelSpacing
    spacing = [float(del_x), float(del_y), del_z]
    return spacing, volume


def interpolate_and_rescale(
    volume, target_shape=(512, 512, 512), original_spacing=(1.0, 1.0, 1.0)
):
    original_shape = np.array(volume.shape)
    scale_factors = np.divide(target_shape, original_shape.astype(float))
    interpolated_volume = zoom(volume, zoom=scale_factors, order=1, mode="nearest")
    new_spacing = np.multiply(original_spacing, scale_factors)
    return new_spacing, interpolated_volume


def pad_volume_to_max_shape(volume):
    max_shape = max(volume.shape)
    padded_volume = np.zeros(
        (max_shape, max_shape, max_shape), dtype=volume.dtype
    ) + np.min(volume)
    z, y, x = volume.shape
    z_start = (max_shape - z) // 2
    y_start = (max_shape - y) // 2
    x_start = (max_shape - x) // 2
    padded_volume[
        z_start : z_start + z, y_start : y_start + y, x_start : x_start + x
    ] = volume
    return padded_volume


def image_from_volume(volume, spacing):
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing(spacing)
    return image


def apply_translation(volume, translation, fill_value=-1000):
    assert len(volume.shape) == 3, "Input volume must be 3D."
    translated_volume = shift(
        volume, shift=translation, cval=fill_value, mode="constant"
    )
    return translated_volume


def resample_volume(volume, spacing):
    image = image_from_volume(volume, spacing)
    original_spacing = image.GetSpacing()
    min_spacing = min(original_spacing)
    min_spacing = max(min_spacing, 0.1)
    output_spacing = (min_spacing, min_spacing, min_spacing)
    original_size = image.GetSize()
    # Calculate the new size based on the desired output spacing
    new_size = [
        int(np.round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, output_spacing)
    ]
    # Resample the volume
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(output_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkLinear)
    resampled_image = resample.Execute(image)
    volume = sitk.GetArrayFromImage(resampled_image)
    spacing = resampled_image.GetSpacing()
    return spacing, volume


def augment_corr_transformation(corr, view_1, view_2, N=16):
    """
    corr: torch.Tensor of shape [N*N, N*N]
    T1, T2: transformation names
    Returns the warped correlation matrix using inverse permutations.
    """
    COORDINATE_TRANSFORM_RULES = {
        "identity": lambda r, c, N: (r, c),
        "rotate90": lambda r, c, N: (N - 1 - c, r),
        "rotate180": lambda r, c, N: (N - 1 - r, N - 1 - c),
        "rotate270": lambda r, c, N: (c, N - 1 - r),
        "flip_horizontal": lambda r, c, N: (r, N - 1 - c),
        "flip_vertical": lambda r, c, N: (N - 1 - r, c),
        "flip_main_diagonal": lambda r, c, N: (c, r),
        "flip_anti_diagonal": lambda r, c, N: (N - 1 - c, N - 1 - r),
    }

    def get_permutation(N, transform_type):
        if transform_type not in COORDINATE_TRANSFORM_RULES:
            raise ValueError(f"Unknown transform: {transform_type}")
        f = COORDINATE_TRANSFORM_RULES[transform_type]
        size = N * N
        perm = np.zeros(size, dtype=int)
        for r in range(N):
            for c in range(N):
                old_idx = r * N + c
                r2, c2 = f(r, c, N)
                perm[old_idx] = r2 * N + c2
        return perm.tolist()

    def apply_view_transform(x, transform_type):
        """
        x: torch.Tensor [N, N]  (2D single-channel image)
        """
        if transform_type == "identity":
            return x
        elif transform_type.startswith("rotate"):
            k = int(transform_type.replace("rotate", "")) // 90
            return torch.rot90(x, k, dims=(0, 1))
        elif transform_type == "flip_horizontal":
            return torch.flip(x, dims=(1,))
        elif transform_type == "flip_vertical":
            return torch.flip(x, dims=(0,))
        elif transform_type == "flip_main_diagonal":
            return x.T
        elif transform_type == "flip_anti_diagonal":
            return torch.flip(x, dims=(0, 1)).T
        else:
            raise ValueError(f"Unknown view transform {transform_type}")

    T1 = random.choice(list(COORDINATE_TRANSFORM_RULES.keys()))
    T2 = random.choice(list(COORDINATE_TRANSFORM_RULES.keys()))
    view1 = apply_view_transform(view_1, T1)
    view2 = apply_view_transform(view_2, T2)
    p1 = get_permutation(N, T1)
    p2 = get_permutation(N, T2)
    inv1 = torch.argsort(torch.tensor(p1, dtype=torch.long))
    inv2 = torch.argsort(torch.tensor(p2, dtype=torch.long))
    return corr[inv1, :][:, inv2], view1, view2


def augment_corr_masked_crop(corr, view_1, view_2, N=16, crop_size=10, patch_size=16):
    """
    Crop and pad in patch space. View_1 and view_2 are of shape [patch_size * N, patch_size * N].
    """
    not_found = False

    def get_valid_crop_coords(corr, N, crop_size):
        crop_size = random.randint(crop_size, N - 2)
        for _ in range(1000):
            r1 = random.randint(0, N - crop_size)
            c1 = random.randint(0, N - crop_size)
            r2 = random.randint(0, N - crop_size)
            c2 = random.randint(0, N - crop_size)
            idxs1 = torch.arange(r1, r1 + crop_size)[:, None] * N + torch.arange(
                c1, c1 + crop_size
            )
            idxs2 = torch.arange(r2, r2 + crop_size)[:, None] * N + torch.arange(
                c2, c2 + crop_size
            )
            if (corr[idxs1.flatten()][:, idxs2.flatten()] > 0.3).sum() > 0:
                return (r1, c1), (r2, c2)
        # raise RuntimeError("No valid crop with correspondence found.")
        not_found = True
        # print("No valid crop with correspondence found.")
        return (0, 0), (0, 0)

    if not_found:
        return corr, view_1, view_2

    (r1, c1), (r2, c2) = get_valid_crop_coords(corr, N, crop_size)

    ps = patch_size
    view1_crop = view_1[
        r1 * ps : (r1 + crop_size) * ps, c1 * ps : (c1 + crop_size) * ps
    ]
    view2_crop = view_2[
        r2 * ps : (r2 + crop_size) * ps, c2 * ps : (c2 + crop_size) * ps
    ]

    pad_amt = N - crop_size
    pad1 = (pad_amt // 2, pad_amt - pad_amt // 2)
    pad2 = (pad_amt // 2, pad_amt - pad_amt // 2)

    view1_padded = torch.nn.functional.pad(
        view1_crop, (pad1[1] * ps, pad1[0] * ps, pad1[1] * ps, pad1[0] * ps), value=0
    )
    view2_padded = torch.nn.functional.pad(
        view2_crop, (pad2[1] * ps, pad2[0] * ps, pad2[1] * ps, pad2[0] * ps), value=0
    )

    corr_new = torch.zeros_like(corr)

    idxs1 = torch.arange(r1, r1 + crop_size)[:, None] * N + torch.arange(
        c1, c1 + crop_size
    )
    idxs2 = torch.arange(r2, r2 + crop_size)[:, None] * N + torch.arange(
        c2, c2 + crop_size
    )
    idxs1 = idxs1.flatten()
    idxs2 = idxs2.flatten()

    for i_local, old_i in enumerate(idxs1):
        for j_local, old_j in enumerate(idxs2):
            if corr[old_i, old_j] != 0:
                new_i = (pad1[0] + i_local // crop_size) * N + (
                    pad1[0] + i_local % crop_size
                )
                new_j = (pad2[0] + j_local // crop_size) * N + (
                    pad2[0] + j_local % crop_size
                )
                corr_new[new_i, new_j] = corr[old_i, old_j]

    return corr_new, view1_padded, view2_padded


def augment_corr_cutout_in(
    corr, view1, view2, N=16, patch_size=16, num_cutouts=1, cutout_size=6
):
    """
    Applies black square cutouts to the views and zeros out the corresponding entries in the correspondence matrix.
    Works on patch space: cutouts are defined in patch units.

    Parameters:
    - view1, view2: [patch_size * N, patch_size * N] tensors.
    - corr: [N*N, N*N] binary correspondence matrix.
    - num_cutouts: number of cutouts per view.
    - cutout_size: side of cutout square (in patch units).

    Returns:
    - view1_cut, view2_cut: views with blacked-out regions.
    - corr_updated: correspondence matrix with affected entries set to 0.
    """
    corr_updated = corr.clone()

    def cut(view, corr, is_view1, cutout_size):
        cutout_size = random.randint(3, cutout_size)
        for _ in range(num_cutouts):
            i = random.randint(0, N - cutout_size)
            j = random.randint(0, N - cutout_size)
            # Zero the image region
            view_padded = view.clone()
            view_padded[
                i * patch_size : (i + cutout_size) * patch_size,
                j * patch_size : (j + cutout_size) * patch_size,
            ] = 0
            # Zero correspondence
            patch_idxs = torch.arange(i, i + cutout_size)[:, None] * N + torch.arange(
                j, j + cutout_size
            )
            patch_idxs = patch_idxs.flatten()
            if is_view1:
                corr[patch_idxs, :] = 0
            else:
                corr[:, patch_idxs] = 0
        return view_padded, corr

    view1_new, corr_updated = cut(
        view1, corr_updated, is_view1=True, cutout_size=cutout_size
    )
    view2_new, corr_updated = cut(
        view2, corr_updated, is_view1=False, cutout_size=cutout_size
    )

    return corr_updated, view1_new, view2_new


def augment_corr_shift_pad(corr, view1, view2, N=16, patch_size=16, max_shift=3):
    """
    corr:    [N*N, N*N] correspondence matrix
    view1:   [C, H, W] first view
    view2:   [C, H, W] second view
    """
    if max_shift is None:
        max_shift = N

    # 1) sample shifts (in patches)
    dr1 = random.randint(-max_shift, max_shift)
    dc1 = random.randint(-max_shift, max_shift)
    dr2 = random.randint(-max_shift, max_shift)
    dc2 = random.randint(-max_shift, max_shift)

    # 2) shift the raw views (pixel pad & slice)
    def shift_view(v, dr, dc):
        H, W = v.shape
        dy, dx = dr * patch_size, dc * patch_size
        pad = (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0))
        v_pad = F.pad(v, pad, value=0)
        # if dy>=0: take from top; else take from pad_bottom
        sy = 0 if dy >= 0 else pad[3]
        sx = 0 if dx >= 0 else pad[1]
        return v_pad[sy : sy + H, sx : sx + W]

    v1s = shift_view(view1, dr1, dc1)
    v2s = shift_view(view2, dr2, dc2)

    # 3) reshape corr to 4D and pad each axis
    P = N * N
    corr4 = corr.view(N, N, N, N)

    # pad order for F.pad on a 4D tensor is:
    # (c2_left, c2_right, r2_top, r2_bottom, c1_left, c1_right, r1_top, r1_bottom)
    pad4 = (
        max(dc2, 0),
        max(-dc2, 0),
        max(dr2, 0),
        max(-dr2, 0),
        max(dc1, 0),
        max(-dc1, 0),
        max(dr1, 0),
        max(-dr1, 0),
    )
    corr4_pad = F.pad(corr4, pad4, value=0)

    # 4) slice each axis back to length N
    r1_start = 0 if dr1 >= 0 else pad4[7]
    c1_start = 0 if dc1 >= 0 else pad4[5]
    r2_start = 0 if dr2 >= 0 else pad4[3]
    c2_start = 0 if dc2 >= 0 else pad4[1]

    corr4_shift = corr4_pad[
        r1_start : r1_start + N,
        c1_start : c1_start + N,
        r2_start : r2_start + N,
        c2_start : c2_start + N,
    ]

    # 5) flatten back to [P, P]
    corr_shift = corr4_shift.contiguous().view(P, P)
    return corr_shift, v1s, v2s
