import os
import ast
import json
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path

import sys
base_dir = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(base_dir)

from src.utils.utils import (
    read_volume,
    interpolate_and_rescale,
    resample_volume,
    pad_volume_to_max_shape,
    read_csv_all,
    apply_translation,
)
from src.utils.utils_drr_fast import apply_transform
from src.utils.utils_remove_table import apply_remove_table


def check_if_exists(hdf5_file, data_type, ind):
    with h5py.File(hdf5_file, "r") as hf:
        existing_groups = [
            key for key in hf[data_type].keys() if key.startswith(str(ind))
        ]
        return len(existing_groups) > 0


def check_views_exist(hdf5_file, data_type, name, nb_drr):
    with h5py.File(hdf5_file, "r") as hf:
        if data_type not in hf:
            print(f"Group '{data_type}' not found in the file.")
            return False

        matching_groups = [
            key for key in hf[data_type].keys() if key.startswith(str(name))
        ]

        # If no matching groups found, return False
        if not matching_groups:
            print(f"No groups found for index {name}")
            return False

        # Assuming you're checking for the first matching group
        group_name = matching_groups[0]
        group = hf[f"{data_type}/{group_name}"]

        # Check for the existence of all views (view1 to view_{nb_drr})
        for i in range(1, nb_drr + 1):
            if f"view{i}" not in group:
                print(f"view{i} not found in group {group_name}")
                return False

        # Check for the existence of all soft and hard correlations between views
        for i in range(1, nb_drr + 1):
            for j in range(i + 1, nb_drr + 1):
                if f"corr{i}_{j}" not in group:
                    print(
                        f"Correlation corr{i}_{j} not found in group {group_name}"
                    )
                    return False
        # If all checks pass
        return True


def create_json_object(row, z_split_number=1, left_right=None):
    json_obj = row.to_dict()
    json_obj.update({"z_split_number": z_split_number, "left_right": left_right})
    return json.dumps(json_obj)


def save_drr(
    hdf5_file,
    unique_id,
    nb_drr,
    views,
    corr,
    json_data,
    data_type,
):
    for i in range(nb_drr - 1):
        for j in range(i + 1, nb_drr):

            def get_index(i, j, N):
                index = i * N - (i * (i + 1)) // 2 + (j - i)
                return index - 1

            index = get_index(i, j, nb_drr)
            save_drr_single(
                hdf5_file,
                unique_id,
                i + 1,
                j + 1,
                views[i],
                views[j],
                corr[index],
                json_data,
                data_type,
            )


def save_drr_single(
    hdf5_file,
    unique_id,
    ind_i,
    ind_j,
    view1,
    view2,
    corr,
    json_data,
    data_type,
):
    with h5py.File(hdf5_file, "a") as hf:
        group_name = f"{data_type}/{unique_id}"
        if group_name not in hf:
            group = hf.create_group(group_name)
        else:
            group = hf[group_name]

        if f"view{ind_i}" in group:
            del group[f"view{ind_i}"]
        group.create_dataset(f"view{ind_i}", data=view1, compression="gzip")
        if f"view{ind_j}" in group:
            del group[f"view{ind_j}"]
        group.create_dataset(f"view{ind_j}", data=view2, compression="gzip")
        if f"corr{ind_i}_{ind_j}" in group:
            del group[f"corr{ind_i}_{ind_j}"]
        group.create_dataset(
            "corr" + f"{ind_i}_" + f"{ind_j}", data=corr, compression="gzip"
        )
        if "metadata" in group:
            del group["metadata"]
        group.create_dataset("metadata", data=json_data)


def apply_drr_to_volume(
    volume,
    save_name,
    angle,
    translation,
    axis,
    dim,
    sid,
    sad,
    scale_factor,
    pooling_type,
    metadata,
    data_type,
    h5_file,
):
    volume = apply_translation(volume, translation, fill_value=-1000)
    volume = np.clip(volume, -1000, 3000) + 1000
    corr, drr = apply_transform(
        volume, angle, axis, dim, sid, sad, scale_factor, pooling_type
    )
    save_name = save_name
    save_drr(
        h5_file,
        save_name,
        len(drr),
        drr,
        corr.numpy(),
        metadata,
        data_type,
    )


def process_ct(row, ind, axis, dim, scale_factor, pooling_type, h5_file):
    (
        vol_path,
        db_type,
        angles,
        translation_x,
        translation_y,
        sid,
        sad,
        train_val,
        organ_name,
        num_split_on_z,
        y_split,
    ) = row
    vol_name = Path(vol_path).name

    angle = ast.literal_eval(angles)
    translation = (0, 0, translation_x)
    sid = ast.literal_eval(sid)
    sad = ast.literal_eval(sad)

    # if not dicom:
    # vol_image = sitk.ReadImage(vol_path)
    # spacing, volume = vol_image.GetSpacing(), sitk.GetArrayFromImage(vol_image)
    # else:
    spacing, volume = read_volume(vol_path)

    if y_split == 1:
        volume = apply_remove_table(volume)

    # try:
    if num_split_on_z == 1:
        spacing, volume = preprocess_volume(volume, spacing, dim)
        save_split_volume(
            volume,
            row,
            ind,
            vol_name,
            angle,
            translation,
            axis,
            dim,
            sid,
            sad,
            scale_factor,
            pooling_type,
            h5_file,
            train_val,
        )
    else:
        z, _, _ = volume.shape
        num_splits = num_split_on_z
        for i in range(num_splits):
            sub_volume = volume[
                i * (z // num_splits) : (i + 1) * (z // num_splits), :, :
            ]
            new_spacing, sub_volume = preprocess_volume(sub_volume, spacing, dim)
            save_split_volume(
                sub_volume,
                row,
                ind,
                vol_name,
                angle,
                translation,
                axis,
                dim,
                sid,
                sad,
                scale_factor,
                pooling_type,
                h5_file,
                train_val,
                i,
            )

    # except Exception as e:
    #     print(f"Error in {ind}: {e}")
    #     print(f"Skipping {ind}")
    #     return


def preprocess_volume(volume, spacing, dim):
    spacing, volume = resample_volume(volume, spacing)
    volume = pad_volume_to_max_shape(volume)
    spacing, volume = interpolate_and_rescale(
        volume, target_shape=(dim, dim, dim), original_spacing=spacing
    )
    return spacing, volume


def save_split_volume(
    volume,
    row,
    ind,
    vol_name,
    angle,
    translation,
    axis,
    dim,
    sid,
    sad,
    scale_factor,
    pooling_type,
    h5_file,
    train_val,
    split_num=1,
):
    nb_drr = len(angle)
    if os.path.exists(h5_file):
        if check_views_exist(
            h5_file, row["train_val"], f"{ind}_{vol_name}_{split_num}", nb_drr
        ):
            print(f"Skipping {ind}")
            return
    metadata = create_json_object(row, z_split_number=split_num)
    apply_drr_to_volume(
        volume,
        f"{ind}_{vol_name}_{split_num}",
        angle,
        translation,
        axis,
        dim,
        sid,
        sad,
        scale_factor,
        pooling_type,
        metadata,
        train_val,
        h5_file,
    )


def generate_drr(ind, row, axis, dim, scale_factor, pooling_type, h5_file):
    if row["type"] == "ct":
        process_ct(row, ind, axis, dim, scale_factor, pooling_type, h5_file)
    else:
        print("Invalid data type")


def main():
    csv_dir = "dataset/data.csv"
    h5_file = "dataset/data.h5"

    data = read_csv_all(csv_dir)
    axis = [1, 1, -1]
    scale_factor = 16
    pooling_type = "average"
    dim = 256

    for ind, row in tqdm(data.iterrows()):
        print(f"generating index: {ind}")
        generate_drr(ind, row, axis, dim, scale_factor, pooling_type, h5_file)
    print("finished")


if __name__ == "__main__":
    main()
