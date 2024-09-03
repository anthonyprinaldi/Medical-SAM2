import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
from PIL import Image


def convert_3d_data(
    root_path: Path,
    save_path: Path,
    slice_dim: int,
    ) -> None:
    """Take in original 3D data for NeuroSAM and convert
    it to the format expected by this repo.

    The dataset should have an overall Training/ and Test/ dir.
    Inside each of these folders there is an image/ and mask/ dir.
    Inside each of these folders are image name dirs e.g., image0001/.
    Inside each image dir there are .png slice files.
    Inside each mask dir there are .npy binary mask files.


    :param root_path: Location of existing NeuroSam data.
    :type root_path: Path
    :param save_path: Location to save the converted data.
    :type save_path: Path
    """

    assert root_path.exists(), f"Path {root_path} does not exist."

    # Create the save path if it doesn't exist
    save_path.mkdir(parents=True, exist_ok=True)

    # Create the Training/ and Test/ dirs
    training_path = save_path / "Training"
    test_path = save_path / "Test"

    training_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    # Create the image/ and mask/ dirs
    training_image_path = training_path / "image"
    training_mask_path = training_path / "mask"

    test_image_path = test_path / "image"
    test_mask_path = test_path / "mask"

    training_image_path.mkdir(parents=True, exist_ok=True)
    training_mask_path.mkdir(parents=True, exist_ok=True)

    test_image_path.mkdir(parents=True, exist_ok=True)
    test_mask_path.mkdir(parents=True, exist_ok=True)

    # grep all the json files
    json_files = list(root_path.glob("*.json"))
    json_files = [x for x in json_files if "overall" not in x.name]

    # loop over json files
    for json_file in json_files:

        image_iter = 0
        dataset_name = re.sub(r"\_[\S]+", "", json_file.stem).lower()

        # Load the json file
        with open(json_file, "r") as f:
            data: List[Dict[str, str]] = json.load(f)

        # Iterate files
        for sample in data:
            train_image_slice_dir = training_image_path / f"{dataset_name}{image_iter:06d}"
            train_mask_slice_dir = training_mask_path / f"{dataset_name}{image_iter:06d}"

            test_image_slice_dir = test_image_path / f"{dataset_name}{image_iter:06d}"
            test_mask_slice_dir = test_mask_path / f"{dataset_name}{image_iter:06d}"

            train_image_slice_dir.mkdir(parents=True, exist_ok=True)
            train_mask_slice_dir.mkdir(parents=True, exist_ok=True)

            # test_image_slice_dir.mkdir(parents=True, exist_ok=True)
            # test_mask_slice_dir.mkdir(parents=True, exist_ok=True)

            image_path = Path(sample["image"])
            mask_path = Path(sample["label"])

            image_path = root_path.parent.parent / image_path
            mask_path = root_path.parent.parent / mask_path

            # Load the image and mask
            image = nib.load(image_path)
            image_array = image.get_fdata()
            mask = nib.load(mask_path)
            mask_array = mask.get_fdata()

            x, y, z = image_array.shape
            
            if slice_dim == 0:
                num_slices = x
            elif slice_dim == 1:
                num_slices = y
            elif slice_dim == 2:
                num_slices = z

            assert image_array.shape == mask_array.shape, \
                f"Image and mask shapes do not match: {image_array.shape} vs {mask_array.shape}"

            # Split the data into training and test
            num_training_slices = int(num_slices * 0.7)
            num_test_slices = num_slices - num_training_slices

            # Save the training data
            # for i in range(num_training_slices):
            for i in range(num_slices):
                
                if slice_dim == 0:
                    image_slice = image_array[i, :, :]
                    mask_slice = mask_array[i, :, :]
                elif slice_dim == 1:
                    image_slice = image_array[:, i, :]
                    mask_slice = mask_array[:, i, :]
                elif slice_dim == 2:
                    image_slice = image_array[:, :, i]
                    mask_slice = mask_array[:, :, i]

                Image.fromarray(image_slice).convert("L").save(train_image_slice_dir / f"{i}.png")
                np.save(train_mask_slice_dir / f"{i}.npy", mask_slice)

            # Save the test data
            # for i in range(num_training_slices, num_slices):

            #     if slice_dim == 0:
            #         image_slice = image_array[i, :, :]
            #         mask_slice = mask_array[i, :, :]
            #     elif slice_dim == 1:
            #         image_slice = image_array[:, i, :]
            #         mask_slice = mask_array[:, i, :]
            #     elif slice_dim == 2:
            #         image_slice = image_array[:, :, i]
            #         mask_slice = mask_array[:, :, i]

            #     Image.fromarray(image_slice).convert("L").save(test_image_slice_dir / f"{i}.png")
            #     np.save(test_mask_slice_dir / f"{i}.npy", mask_slice)

            print(f"Processed {train_image_slice_dir.name}")

            image_iter += 1



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--slice-dim", type=int, default=2)

    args = parser.parse_args()
    args = vars(args)

    root_path = Path(args.get("root_path"))
    save_path = Path(args.get("save_path"))
    slice_dim = args.get("slice_dim")

    convert_3d_data(root_path, save_path, slice_dim)