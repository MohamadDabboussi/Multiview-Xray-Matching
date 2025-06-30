import itk
import math
import numpy as np
from itk import RTK as rtk
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn.functional as F
from torchvision import transforms
from src.utils.utils import sigmoid


def transform_array_to_F3(array):
    InputPixelType = itk.D
    OutputPixelType = itk.F
    Dimension = 3
    InputImageType = itk.Image[InputPixelType, Dimension]
    OutputImageType = itk.Image[OutputPixelType, Dimension]
    itk_image = itk.image_from_array(array)
    cast_filter = itk.CastImageFilter[InputImageType, OutputImageType].New()
    cast_filter.SetInput(itk_image)
    cast_filter.Update()
    output_image = cast_filter.GetOutput()
    assert (
        type(output_image) == itk.itkImagePython.itkImageF3
    ), "cannot transform to itkImageF3"
    return output_image


def downsample_volume(volume, dim, pooling_type):
    # Convert volume to a PyTorch tensor
    volume_tensor = torch.from_numpy(volume)
    volume_tensor = volume_tensor.unsqueeze(0)
    # Apply max pooling or average pooling
    if pooling_type == "max":
        # output = F.max_pool3d(volume_tensor, kernel_size=scale_factor, stride=scale_factor)
        output = F.adaptive_max_pool3d(volume_tensor, output_size=(dim, dim, dim))
    elif pooling_type == "average":
        # output = F.avg_pool3d(volume_tensor, kernel_size=scale_factor, stride=scale_factor)
        output = F.adaptive_avg_pool3d(volume_tensor, output_size=(dim, dim, dim))
    else:
        raise ValueError("Invalid pooling type. Choose 'max' or 'average'.")
    # Remove the channel dimensions
    output = output.squeeze(0)
    # Convert output tensor back to a NumPy array
    downsampled_volume = output.numpy()
    return downsampled_volume


def drr_projections(volume, angles, axis, sid, sad, dim, nb_drr=2):
    ImageType = itk.Image[itk.F, 3]
    geometry = rtk.ThreeDCircularProjectionGeometry.New()
    for ang, sid_, sad_ in zip(angles, sid, sad):
        sdd_ = sid_ + sad_
        geometry.AddProjection(
            sid_, sdd_, 90 * axis[1], 0, 0, ang[0] * axis[0] + 90, ang[1] * axis[2] + 90
        )
    ConstantImageSourceType = rtk.ConstantImageSource[ImageType]
    constantImageSource = ConstantImageSourceType.New()
    origin = [-(dim - 1), -(dim - 1), 0.0]
    sizeOutput = [dim, dim, nb_drr]
    spacing = [2.0, 2.0, 2.0]
    constantImageSource.SetOrigin(origin)
    constantImageSource.SetSpacing(spacing)
    constantImageSource.SetSize(sizeOutput)
    constantImageSource.SetConstant(0.0)
    if volume is not None:
        inputCT = transform_array_to_F3(volume)
    else:
        print("Unknown Iput type")
    inputCT.SetOrigin(
        [
            -0.5
            * (inputCT.GetLargestPossibleRegion().GetSize()[0] - 1)
            * inputCT.GetSpacing()[0],
            -0.5
            * (inputCT.GetLargestPossibleRegion().GetSize()[1] - 1)
            * inputCT.GetSpacing()[1],
            -0.5
            * (inputCT.GetLargestPossibleRegion().GetSize()[2] - 1)
            * inputCT.GetSpacing()[2],
        ]
    )

    # Pr
    JosephType = rtk.JosephForwardProjectionImageFilter[
        ImageType, ImageType
    ]  # [tuple([ImageType] * nb_drr)]
    joseph = JosephType.New()
    joseph.SetGeometry(geometry)
    joseph.SetInput(constantImageSource.GetOutput())
    joseph.SetInput(1, inputCT)
    joseph.Update()
    drr = joseph.GetOutput()
    return itk.GetArrayFromImage(drr)


def process_chunk(vol, angles, axis, sid, sad, dim, i, j, k, nb_drr=2):
    vol_ = np.zeros_like(vol)
    vol_[i, j, k] = vol[i, j, k]
    drr = drr_projections(
        volume=vol_, angles=angles, axis=axis, sid=sid, sad=sad, dim=dim, nb_drr=nb_drr
    )
    return drr


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def get_index(i, j, N):
    index = i * N - (i * (i + 1)) // 2 + (j - i)
    return index - 1


def cross_correlation(d, voxel_value, corr_shape, nb_drr):

    correlations = np.zeros(corr_shape)

    for i in range(d.shape[0] - 1):
        for j in range(i + 1, d.shape[0]):
            x1 = d[i].ravel()
            x2 = d[j].ravel()
            if np.max(x1) != 0 and np.max(x2) != 0:
                x1 = normalize(x1)
                x2 = normalize(x2)
            correlation = np.outer(x1, x2) * voxel_value
            index = get_index(i, j, nb_drr)
            correlations[index] = correlation
    return correlations


def compute_correspondence_parallel_soft(vol, angles, axis, sid, sad, dim, nb_drr=2):
    x, y, z = vol.shape
    comb_2 = math.comb(nb_drr, 2)
    corr_shape = (comb_2, z * y, z * y)
    corr = np.zeros(corr_shape)

    def compute_voxel_correspondence(i, j, k, nb_drr, corr_shape):
        drr = process_chunk(vol, angles, axis, sid, sad, dim, i, j, k, nb_drr)
        return cross_correlation(drr, vol[i, j, k], corr_shape, nb_drr)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(compute_voxel_correspondence, i, j, k, nb_drr, corr_shape)
            for i in range(x)
            for j in range(y)
            for k in range(z)
        ]
        # Accumulate the results
        for future in futures:
            result = future.result()
            corr = np.maximum(corr, result)
    return corr


def apply_transform(
    volume, angles, axis, dim, sid, sad, scale_factor, pooling_type="average"
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    transform1 = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    nb_drr = len(angles)
    ds_vol = sigmoid(volume, x0=850, k=0.005)

    ds_vol_ = downsample_volume(
        ds_vol, int(dim / scale_factor), "average"  # "max"
    )  # adpate the scale factor
    corr = torch.tensor(
        compute_correspondence_parallel_soft(
            ds_vol_, angles, axis, sid, sad, int(dim / scale_factor), nb_drr
        )
    )
    volume = sigmoid(volume, x0=1500, k=0.005)
    drr = drr_projections(volume, angles, axis, sid, sad, dim, nb_drr)
    views = []
    for i in range(nb_drr):
        view = torch.from_numpy(drr[i])
        view = F.interpolate(
            view.unsqueeze(0).unsqueeze(0),
            size=(dim, dim),
            mode="bilinear",
            align_corners=False,
        )
        views.append(transform(np.array(view.squeeze(0).squeeze(0))))

    corr_soft_transformed = []
    for i in range(corr.shape[0]):
        corr_soft_transformed.append(
            transform1(np.array(corr[i].squeeze(0))).squeeze(0)
        )
    corr_soft = torch.stack(corr_soft_transformed)
    return corr_soft, views
