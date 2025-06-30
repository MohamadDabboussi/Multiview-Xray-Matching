import os
import random
import numpy as np
import pandas as pd

NUMBER_OF_GENERATION_PER_CT = 5  # Number of generations for each CT volume
NUMBER_OF_VIEWS_PER_GENERATION = 2  # Number of views for each generation

# Fill these lists with the names of CT volumes that need specific handling
auto_remove = []  # list of ct volumes that needs to have auto table removal
split_z = {
    # "ct_name": number_of_splits_on_z_axis (e.g when having a ct scan of the whole body and we want to split it into multiple parts)
    # if not in the list, the default value is 1
    "demo_CT": 1,
}
organ_name_dict = {
    "demo_CT": "knee",
}


def biased_multiview_angle():
    """Generates a random angle biased towards 0, 90, 180, and 270 degrees."""
    key_angles = [0, 90, 180, 270]  # Key angles for X-ray acquisition
    probabilities = [0.4, 0.3, 0.2, 0.1]  # Probability weights for each key angle
    # Pick a key angle based on weights
    chosen_angle = random.choices(key_angles, probabilities)[0]
    # Add small noise to allow for variation around the chosen key angle
    noise = np.random.normal(0, 45)  # Std dev of 15 degrees for small variation
    return (chosen_angle + noise) % 360  # Ensure the angle stays within 0-360 degrees


def biased_angle(mean, std_dev, lower_bound, upper_bound):
    """Generates a random angle biased towards the mean using a normal distribution, clipped within bounds."""
    angle = np.random.normal(mean, std_dev)
    return max(lower_bound, min(angle, upper_bound))


def get_random_angle(x):
    if x == 0:
        # Use the biased multiview angle for more realistic X-ray acquisition
        first_axis_angle = biased_multiview_angle()
        second_axis_angle = biased_angle(180, 15, 150, 210)

        return [first_axis_angle, second_axis_angle]

    elif x == 1:
        # Rotation around the first axis is fixed, second axis varies
        first_axis_angle = biased_angle(0, 15, -20, 20) % 360
        second_axis_angle = biased_angle(180, 45, 0, 360)

        return [first_axis_angle, second_axis_angle]

    elif x == 2:
        # Both axes are free to vary
        # Use the biased multiview angle for the first axis
        first_axis_angle = biased_multiview_angle()
        second_axis_angle = biased_angle(180, 60, 0, 360)
        return [first_axis_angle, second_axis_angle]


def get_random_translation(x):
    if x == 0:
        return 0
    elif x == 1:
        return random.randint(-50, -1)
    elif x == 2:
        return random.randint(1, 50)


def get_random_sid_sad(x):
    if x == 0:
        return random.randint(1000, 1100)
    elif x == 1:
        return random.randint(800, 1200)


def create_augmented_csv(folders, output_csv):
    rows = []

    for folder_path in folders:
        number_of_generation = NUMBER_OF_GENERATION_PER_CT
        number_of_views = NUMBER_OF_VIEWS_PER_GENERATION
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isdir(file_path):

                    ## if not dicom
                    # if not os.path.isdir(file_path):

                    organ_name = organ_name_dict.get(file_name, "null")
                    split_on_z = split_z.get(file_name, 1)

                    if file_name in auto_remove:
                        y_split = 1
                    else:
                        y_split = -1

                    file_type = "ct"
                    train_or_val = "train" if random.random() > 0.1 else "val"
                    
                    for _ in range(number_of_generation):
                        for split_on_z_i in range(1, split_on_z + 1):
                            angles = [
                                get_random_angle(random.randint(0, 2))
                                for _ in range(number_of_views)
                            ]
                            translation_x = get_random_translation(random.randint(0, 2))
                            translation_y = get_random_translation(random.randint(0, 2))
                            sid = [get_random_sid_sad(0) for _ in range(number_of_views)]
                            sad = [get_random_sid_sad(1) for _ in range(number_of_views)]

                            row = {
                                "image_path": file_path,
                                "type": file_type,
                                "angles": angles,
                                "translation_x": translation_x,
                                "translation_y": translation_y,
                                "sid": sid,
                                "sad": sad,
                                "train_val": train_or_val,
                                "organ_name": organ_name,
                                "num_split_on_z": split_on_z_i,
                                "y_split": y_split,
                            }
                            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=True)


# folders that contains the ct volumes
folders = [
    "dataset/ct_volumes/",
]
output_csv = "dataset/data.csv"

create_augmented_csv(folders, output_csv)
