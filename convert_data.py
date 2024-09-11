import json
import os.path as osp
from pathlib import Path

import nibabel as nib
import numpy as np
import torchio as tio
from PIL import Image
from tqdm import tqdm


class AbdomenCTJSONGenerator:
    dir = Path('/home/arinaldi/project/aiconsgrp/data/AbdomenCT_1K')
    name = "AbdomenCT"
class AMOSJSONGenerator:
    dir = Path('/home/arinaldi/project/aiconsgrp/data/AMOS')
    name = "AMOS"
class BratsJSONGenerator:
    dir = Path('/home/arinaldi/project/aiconsgrp/data/BraTs2020')
    name = "BRATS"
class CovidCTJSONGenerator:
    dir = Path('/home/arinaldi/project/aiconsgrp/data/COVID_CT_Lung')
    name = "CovidCT"
class CTStrokeJSONGenerator:
    dir = Path('/home/arinaldi/project/aiconsgrp/data/CTStroke')
    name = "CTStroke"
class HealthyTotalBodyJSONGenerator:
    dir = Path('/home/arinaldi/project/aiconsgrp/data/Healthy-Total-Body CTs NIfTI Segmentations and Segmentation Organ Values spreadsheet')
    name = "HealthyTotalBody"
class ISLESJSONGenerator:
    dir = Path('/home/arinaldi/project/aiconsgrp/data/ISLES-2022')
    name = "ISLES"
class KitsJSONGenerator:
    dir = Path('/home/arinaldi/project/aiconsgrp/data/kits23')
    name = "Kits"
class KneeJSONGenerator:
    dir = Path('/home/arinaldi/project/aiconsgrp/data/KneeMRI')
    name = "StanfordKnee"
class LITSJSONGenerator:
    dir = Path('/home/arinaldi/project/aiconsgrp/data/LITS')
    name = "LiTS"
class LUNAJSONGenerator:
    dir = Path('/home/arinaldi/project/aiconsgrp/data/LUNA16')
    name = "LUNA"
class MMWHSJSONGenerator:
    dir = Path('/home/arinaldi/project/aiconsgrp/data/MM-WHS 2017 Dataset')
    name = "MultiModalWholeHeart"
class MSDJSONGenerator:
    dir = Path('/home/arinaldi/project/aiconsgrp/data/MSD')
    name = "MedSamDecathlon"
class CTORGJSONGenerator:
    dir = Path('/home/arinaldi/project/aiconsgrp/data/PKG - CT-ORG')
    name = "CTOrgan"
class UpennJSONGenerator:
    dir = Path('/home/arinaldi/project/aiconsgrp/data/PKG - UPENN-GBM-NIfTI')
    name = "MRIGlioblastoma"
class ProstateJSONGenerator:
    dir = Path('/home/arinaldi/project/aiconsgrp/data/Prostate MR Image Segmentation')
    name = "ProstateMRI"
class SegTHORJSONGenerator:
    dir = Path('/home/arinaldi/project/aiconsgrp/data/SegTHOR')
    name = "SegThoracicOrgans"
class TCIAPancreasJSONGenerator:
    dir = Path('/home/arinaldi/project/aiconsgrp/data/TCIA_pancreas_labels-02-05-2017')
    name = "PancreasCt"
class TotalSegmentatorJSONGenerator:
    dir = Path('/home/arinaldi/project/aiconsgrp/data/Totalsegmentator_dataset_v201')
    name = "TotalSegmentator"
class ONDRIJSONGenerator:
    dir = Path('/home/arinaldi/project/aiconsgrp/data/wmh_hab')
    name = "ONDRI"
class WORDJSONGenerator:
    dir = Path('/home/arinaldi/project/aiconsgrp/data/WORD-V0.1.0')
    name = "WORD"

DATASET_LIST = [
    AbdomenCTJSONGenerator,
    AMOSJSONGenerator,
    BratsJSONGenerator,
    CovidCTJSONGenerator,
    CTStrokeJSONGenerator,
    HealthyTotalBodyJSONGenerator,
    ISLESJSONGenerator,
    KitsJSONGenerator,
    KneeJSONGenerator,
    LITSJSONGenerator,
    LUNAJSONGenerator,
    MMWHSJSONGenerator,
    MSDJSONGenerator,
    CTORGJSONGenerator,
    UpennJSONGenerator,
    ProstateJSONGenerator,
    SegTHORJSONGenerator,
    TCIAPancreasJSONGenerator,
    TotalSegmentatorJSONGenerator,
    ONDRIJSONGenerator,
    WORDJSONGenerator,
]


def main(args):
    dt = args.dataset_type
    slice_dim = args.slice_dim

    save_dir = Path("/scratch/scratch01/mggrp/arinaldi/data")
    training_save_dir = save_dir / "Training"
    training_save_dir_image = training_save_dir / "image"
    training_save_dir_mask = training_save_dir / "mask"
    testing_save_dir = save_dir / "Test"
    testing_save_dir_image = testing_save_dir / "image"
    testing_save_dir_mask = testing_save_dir / "mask"
    validation_save_dir = save_dir / "Validation"
    validation_save_dir_image = validation_save_dir / "image"
    validation_save_dir_mask = validation_save_dir / "mask"

    if dt == "Tr":
        main_img_dir = training_save_dir_image
        main_seg_dir = training_save_dir_mask
    elif dt == "Ts":
        main_img_dir = testing_save_dir_image
        main_seg_dir = testing_save_dir_mask
    elif dt == "Val":
        main_img_dir = validation_save_dir_image
        main_seg_dir = validation_save_dir_mask


    for dataset in DATASET_LIST:
        image_iter = 0
        dataset_dir = dataset.dir

        meta_info = json.load(open(osp.join(dataset_dir, "dataset.json"))) if \
            dataset.name != "TotalSegmentator" else json.load(open(osp.join(dataset_dir, "dataset_medsam2.json")))

        print(meta_info["name"], meta_info["modality"])
        num_classes = len(meta_info["labels"]) - 1
        print("num_classes:", num_classes, meta_info["labels"])

        dataset_name = dataset.name

        data_list = meta_info[
            {"Tr": "training", "Val": "validation", "Ts": "testing"}[dt]
        ]

        if data_list is None:
            continue

        # only get the unique values f the data_list
        data_list = [(item["image"], item["seg"]) for item in data_list]
        data_list = list(set(data_list))

        for item in tqdm(data_list, desc=f"{dataset_name} - {len(data_list)} images"):

            img, seg = item
            # if dataset_name == "TotalSegmentator":
            #     cls_name = (
            #         Path(seg).parts[-1].split("_", maxsplit=1)[1].replace(".nii.gz", "")
            #     )
            # elif dataset_name == "MedSamDecathlon":
            #     task = Path(seg).parts[-3]
            #     cls_name = meta_info["labels"][task][str(seg_idx)].replace(" ", "_")
            # elif dataset_name == "CovidCT":
            #     task = Path(seg).parts[-2].split("_")[1]
            #     cls_name = meta_info["labels"][task][str(seg_idx)].replace(" ", "_")
            # elif dataset_name == "CTStroke":
            #     task = Path(seg).parts[-2].split("_")[1]
            #     cls_name = meta_info["labels"][task][str(seg_idx)].replace(" ", "_")
            # else:
            #     cls_name = meta_info["labels"][str(seg_idx)].replace(" ", "_")

            # img_parent_folder = Path(img).parent.parts[-1]
            # img_ext = (
            #     img_parent_folder.split("_")[-1] if "_" in img_parent_folder else ""
            # )

            # seg_parent_folder = Path(seg).parent.parts[-1]
            # seg_ext = (
            #     seg_parent_folder.split("_")[-1] if "_" in seg_parent_folder else ""
            # )

            img_dir = main_img_dir / f"{dataset_name}{image_iter:06d}"
            seg_dir = main_seg_dir / f"{dataset_name}{image_iter:06d}"

            img_dir.mkdir(parents=True, exist_ok=True)
            seg_dir.mkdir(parents=True, exist_ok=True)

            # Load the image and mask
            image = nib.load(img)
            image_array = image.get_fdata()
            mask = nib.load(seg)
            mask_array = mask.get_fdata()

            try:
                if image_array.ndim == 4:
                    image_array = image_array[:, :, :, 0]
                
                x, y, z = image_array.shape
            except Exception as e:
                print(f"Error: {e}")
                print(f"image: {image_array.shape}, mask: {mask_array.shape}")
                print(f"image file: {img}, mask file: {seg}")
                exit(-1)
            
            if slice_dim == 0:
                num_slices = x
            elif slice_dim == 1:
                num_slices = y
            elif slice_dim == 2:
                num_slices = z

            assert image_array.shape == mask_array.shape, \
                f"Image and mask shapes do not match: {image_array.shape} vs {mask_array.shape}"

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

                Image.fromarray(image_slice).convert("L").save(img_dir / f"{i}.png")
                np.save(seg_dir / f"{i}.npy", mask_slice)

            image_iter += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["Tr", "Val", "Ts"],
        default="Tr",
        help="Dataset type to convert.",
    )
    parser.add_argument(
        "--slice-dim",
        type=int,
        default=2,
        help="Dimension to slice the 3D image",
    )
    args = parser.parse_args()

    main(args)