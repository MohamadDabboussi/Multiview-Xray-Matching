import numpy as np
import scipy.ndimage as ndi
from skimage.transform import hough_line, hough_line_peaks
from scipy.signal import medfilt


#  Step 1: Reformat Data to Get Sagittal Image
def get_sagittal_image(volume, slice_index):
    return volume[:, :, slice_index]


# Step 2: Binarize the Sagittal Image
def binarize_image(sagittal_image, threshold=-500):
    binary_image = np.where(sagittal_image >= threshold, 1, 0)
    return binary_image


# Step 3: Detect Vertical Edges
def detect_vertical_edges(binary_image):
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    edges = ndi.convolve(binary_image, sobel_y)
    return edges


# Step 4: Perform a Hough Transform
def perform_hough_transform(edges):
    hspace, angles, distances = hough_line(
        edges, theta=np.linspace(-1 * np.pi / 180, 1 * np.pi / 180, 21)
    )
    return hspace, angles, distances


# Step 5: Merge Close Peaks
def merge_close_peaks(peaks, max_distance=5):
    if len(peaks) < 2:
        return peaks
    merged_peaks = [peaks[0]]
    for peak in peaks[1:]:
        if peak - merged_peaks[-1] <= max_distance:
            merged_peaks[-1] = (merged_peaks[-1] + peak) // 2
        else:
            merged_peaks.append(peak)
    return merged_peaks


# Step 6: Determine the Table Top
def determine_table_top(hspace, angles, distances):
    hspace_peaks, angles_peaks, d_peaks = hough_line_peaks(
        hspace, angles, distances, min_distance=5
    )
    if len(hspace_peaks) == 0:
        return None, None  # No peaks detected

    merged_peaks = merge_close_peaks(d_peaks)

    if len(merged_peaks) == 1:
        return (
            angles_peaks[0],
            merged_peaks[0],
        )  # Only one peak, assume it's the table top

    # Evaluate the distance between the first two peaks
    first_peak, second_peak = merged_peaks[0], merged_peaks[1]
    distance = abs(second_peak - first_peak)

    # Assume typical cushion thickness to be around 5 pixels
    cushion_thickness_threshold = 5  # 5

    if distance <= cushion_thickness_threshold:
        table_top_peak = second_peak
    else:
        table_top_peak = first_peak

    return angles_peaks[0], table_top_peak


# Step 7: Smooth the Table Top Profile
def smooth_table_profile(intercepts, window_size=5):
    smoothed_intercepts = medfilt(intercepts, kernel_size=window_size)
    return smoothed_intercepts


# Step 8: Create Parabola Mask
def create_parabola_mask(width, height, x_center, y_center, extremities):
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    # Unpack extremities
    left_x, left_y = extremities[0]
    middle_x, middle_y = extremities[1]
    right_x, right_y = extremities[2]

    # Calculate a for the parabola passing through the points
    # Using middle point as vertex: (middle_x, middle_y) = (h, k)
    h, k = middle_x, middle_y
    a = (left_y - k) / (left_x - h) ** 2

    # Calculate the parabolic shape
    y_parabola = k + a * (X - h) ** 2

    # Create mask based on the parabolic shape
    mask = np.where(Y >= y_parabola, 0, 1)
    return mask


# Step 9: Remove the Table in the Transverse Slices
def remove_table(volume, intercepts_all, angles_all, air_value=-1000):
    height, width, depth = volume.shape

    # Find the first, middle, and last non-None intercepts and angles
    valid_indices = [i for i in range(depth) if angles_all[i] is not None]
    if len(valid_indices) < 3:
        raise ValueError(
            "Not enough valid slices with detected table tops to determine parabola."
        )

    # first_idx, middle_idx, last_idx = valid_indices[0], valid_indices[len(valid_indices)//2], valid_indices[-1]
    first_idx, middle_idx, last_idx = valid_indices[0], depth // 2, valid_indices[-1]
    first_intercept, first_angle = intercepts_all[first_idx], angles_all[first_idx]
    middle_intercept, middle_angle = intercepts_all[middle_idx], angles_all[middle_idx]
    # last_intercept, last_angle = intercepts_all[last_idx], angles_all[last_idx]

    # Use the median of the middle intercept array
    first_intercept_median = np.median(first_intercept) - (
        20 * depth // 512
    )  # 20 pixels normal size in 512 size
    middle_intercept_median = np.median(middle_intercept) - (20 * depth // 512)
    # last_intercept_median = np.median(last_intercept)

    # Calculate the extremities for the parabolic mask
    extremities = [
        (first_idx, first_intercept_median + height // 2 * np.tan(first_angle)),
        (middle_idx, middle_intercept_median + height // 2 * np.tan(middle_angle)),
        (last_idx, first_intercept_median + height // 2 * np.tan(first_angle)),
    ]
    # (last_idx, last_intercept_median + height // 2 * np.tan(last_angle))]

    x_center = middle_idx  # width // 2
    y_center = int(middle_intercept_median + height // 2 * np.tan(middle_angle))
    mask = create_parabola_mask(width, depth, x_center, y_center, extremities)
    for y in range(height):
        volume[y, :, :][mask == 0] = air_value
    return volume


# Main function to process the CT volume
def apply_remove_table(volume):
    height, width, depth = volume.shape
    intercepts_all = np.zeros((depth, height))
    angles_all = [None] * depth

    for i in range(depth):
        sagittal_image = get_sagittal_image(volume, i)
        binary_image = binarize_image(sagittal_image)
        edges = detect_vertical_edges(binary_image)

        hspace, angles, distances = perform_hough_transform(edges)
        angle, intercepts = determine_table_top(hspace, angles, distances)

        if intercepts is None:
            intercepts = np.zeros(height)  # Default intercepts if no peaks are detected
        intercepts_all[i, :] = intercepts
        angles_all[i] = angle

    # Smoothing intercepts for each y coordinate across all slices
    for y in range(height):
        intercepts_all[:, y] = smooth_table_profile(intercepts_all[:, y], window_size=5)

    volume_processed = remove_table(volume, intercepts_all, angles_all)
    return volume_processed
