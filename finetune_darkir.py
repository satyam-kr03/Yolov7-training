import os
import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from pytorch_accelerated.callbacks import (
    EarlyStoppingCallback,
    SaveBestModelCallback,
    get_default_callbacks,
)
from pytorch_accelerated.schedulers import CosineLrScheduler
from torch.utils.data import Dataset

from yolov7 import create_yolov7_model
from yolov7.dataset import Yolov7Dataset, create_yolov7_transforms, yolov7_collate_fn
from yolov7.evaluation import CalculateMeanAveragePrecisionCallback
from yolov7.loss_factory import create_yolov7_loss
from yolov7.trainer import Yolov7Trainer, filter_eval_predictions

def extract_id_from_filename(filename):
    # Extract numeric portion from format like "2015_00001.jpg" to get 1
    try:
        # Extract the part after underscore and before .jpg
        id_part = filename.split('_')[1].split('.')[0]
        # Remove leading zeros
        return int(id_part.lstrip('0'))
    except (IndexError, ValueError):
        print(f"Failed to extract ID from filename: {filename}")
        return -1  # Fallback value for invalid formats
    
def debug_dataset_loading(train_df, valid_df, lookups):
    """Print debug information about loaded datasets"""
    print(f"\nTotal training images: {len(train_df['image'].unique())}")
    print(f"Total validation images: {len(valid_df['image'].unique())}")
    print(f"Total unique classes: {len(lookups['class_id_to_label'])}")
    
    print("\nSample of image_id_to_image mapping:")
    for i, (id, img) in enumerate(list(lookups['image_id_to_image'].items())[:5]):
        print(f"  {id}: {img}")
    
    print("\nVerifying image IDs in dataframes:")
    print(f"Training df unique image_ids: {sorted(train_df['image_id'].unique())}")
    print(f"Validation df unique image_ids: {sorted(valid_df['image_id'].unique())}")


def load_exdark_df(annotations_file_path: Path, images_path: Path, 
                    background_class_name: str = "background",
                    n_empty: int = 100,
                    val_split: float = 0.2,
                    seed: int = 42):
    # 1) list all image files
    all_images = sorted(p.name for p in images_path.iterdir() if p.is_file())
    print(f"Found {len(all_images)} image files")
    
    # 2) read your CSV; assume it has columns:
    #    image,xmin,ymin,xmax,ymax,class_name
    ann = pd.read_csv(annotations_file_path)
    ann["has_annotation"] = True
    
    # Debug: Print unique images in annotations
    print(f"Found {len(ann['image'].unique())} unique images in annotations")
    
    # 3) find up to n_empty images with no annotations
    annotated_images = set(ann["image"].unique())
    empty_images = sorted(set(all_images) - annotated_images)[:n_empty]
    empty_df = pd.DataFrame({
        "image": empty_images,
        # fill bbox columns with NaN
        "xmin":   [pd.NA] * len(empty_images),
        "ymin":   [pd.NA] * len(empty_images),
        "xmax":   [pd.NA] * len(empty_images),
        "ymax":   [pd.NA] * len(empty_images),
        "class_name": [background_class_name] * len(empty_images),
        "has_annotation": [False] * len(empty_images),
    })
    
    # 4) concat
    df = pd.concat([ann, empty_df], ignore_index=True)
    
    # 5) build mappings
    #    a) images ↔ IDs
    image_to_image_id = {}
    for img in all_images:
        img_id = extract_id_from_filename(img)
        image_to_image_id[img] = img_id
        print(f"Mapping: {img} -> {img_id}")  # Debug print
    
    image_id_to_image = {id: img for img, id in image_to_image_id.items()}
    
    #    b) classes ↔ IDs
    #       include every class that appears (incl. background if present)
    class_names = list(df["class_name"].unique())
    class_id_to_label = {i: cls for i, cls in enumerate(class_names)}
    class_label_to_id = {cls: i for i, cls in class_id_to_label.items()}
    
    # 6) map into df
    df_images = set(df["image"])
    all_image_set = set(all_images)
    
    # Debug: Check for images in df not found in mapping
    missing_images = df_images - all_image_set
    if missing_images:
        print(f"WARNING: These images in dataframe aren't in the image folder: {missing_images}")
    
    # Map image to image_id - check for missing IDs
    df["image_id"] = df["image"].map(image_to_image_id)
    missing_ids = df[df["image_id"].isna()]["image"].unique()
    if len(missing_ids) > 0:
        print(f"WARNING: These images couldn't be mapped to IDs: {missing_ids}")
    
    df["class_id"] = df["class_name"].map(class_label_to_id)
    
    # 7) split by *unique* images
    random.seed(seed)
    unique_images = list(df["image"].unique())
    n_val = int(len(unique_images) * val_split)
    val_images = set(random.sample(unique_images, n_val))
    
    train_df = df[~df["image"].isin(val_images)].reset_index(drop=True)
    valid_df = df[ df["image"].isin(val_images)].reset_index(drop=True)
    
    # 8) return
    lookups = {
        "image_id_to_image": image_id_to_image,
        "image_to_image_id": image_to_image_id,
        "class_id_to_label": class_id_to_label,
        "class_label_to_id": class_label_to_id,
    }

    # Print missing image IDs in each dataframe
    print("\nMissing image_ids in train_df:", train_df[train_df["image_id"].isna()]["image"].unique())
    print("\nMissing image_ids in valid_df:", valid_df[valid_df["image_id"].isna()]["image"].unique())
    
    print(train_df.head())
    print(valid_df.head())

    # Print the class_id_to_label dictionary for debugging
    print("Class ID to Label Mapping:")
    for class_id, label in class_id_to_label.items():
        print(f"{class_id}: {label}")
    # print("Class Label to ID Mapping:")
    # for label, class_id in class_label_to_id.items():
    #     print(f"{label}: {class_id}")

    # debug_dataset_loading(train_df, valid_df, lookups)
    return train_df, valid_df, lookups


class ExDarkAdaptor(Dataset):
    def __init__(
        self,
        images_dir_path,
        annotations_dataframe,
        transforms=None,
    ):
        self.images_dir_path = Path(images_dir_path)
        self.annotations_df = annotations_dataframe
        self.transforms = transforms

        self.image_idx_to_image_id = {
            idx: image_id
            for idx, image_id in enumerate(self.annotations_df.image_id.unique())
        }
        self.image_id_to_image_idx = {
            v: k for k, v, in self.image_idx_to_image_id.items()
        }

    def __len__(self) -> int:
        return len(self.image_idx_to_image_id)

    def __getitem__(self, index):
        image_id = self.image_idx_to_image_id[index]
        image_info = self.annotations_df[self.annotations_df.image_id == image_id]
        if len(image_info) == 0:
            print(f"Warning: No annotation found for image_id {image_id}")
            return self.__getitem__((index + 1) % len(self))
        file_name = image_info.image.values[0]
        assert image_id == image_info.image_id.values[0]

        image = Image.open(self.images_dir_path / file_name).convert("RGB")
        image = np.array(image)

        image_hw = image.shape[:2]

        if image_info.has_annotation.any():
            xyxy_bboxes = image_info[["xmin", "ymin", "xmax", "ymax"]].values
            # Ensure bounding boxes are within image boundaries
            xyxy_bboxes[:, 0] = np.clip(xyxy_bboxes[:, 0], 0, image.shape[1] - 1)
            xyxy_bboxes[:, 1] = np.clip(xyxy_bboxes[:, 1], 0, image.shape[0] - 1)
            xyxy_bboxes[:, 2] = np.clip(xyxy_bboxes[:, 2], 0, image.shape[1])
            xyxy_bboxes[:, 3] = np.clip(xyxy_bboxes[:, 3], 0, image.shape[0])
            class_ids = image_info["class_id"].values
        else:
            xyxy_bboxes = np.array([])
            class_ids = np.array([])

        if self.transforms is not None:
            transformed = self.transforms(
                image=image, bboxes=xyxy_bboxes, labels=class_ids
            )
            image = transformed["image"]
            xyxy_bboxes = np.array(transformed["bboxes"])
            class_ids = np.array(transformed["labels"])

        return image, xyxy_bboxes, class_ids, image_id, image_hw


def main(
    data_path: str = '//content/Yolov7-training/data/ExDark_Darkir',
    # data_path: str = './ExDark_Darkir',
    image_size: int = 640,
    # image_size: int = 416,
    pretrained: bool = True,
    num_epochs: int = 50,
    batch_size: int = 8,
):

    # Load data
    data_path = Path(data_path)
    images_path = data_path / "training_images"
    annotations_file_path = data_path / "annotations.csv"

    train_df, valid_df, lookups = load_exdark_df(annotations_file_path, images_path)
    num_classes = 12

    # Create datasets
    train_ds = ExDarkAdaptor(
        images_path,
        train_df,
    )
    eval_ds = ExDarkAdaptor(images_path, valid_df)

    train_yds = Yolov7Dataset(
        train_ds,
        create_yolov7_transforms(training=True, image_size=(image_size, image_size)),
    )
    eval_yds = Yolov7Dataset(
        eval_ds,
        create_yolov7_transforms(training=False, image_size=(image_size, image_size)),
    )

    # Create model, loss function and optimizer
    model = create_yolov7_model(
        architecture="yolov7", num_classes=num_classes, pretrained=pretrained
    )

    loss_func = create_yolov7_loss(model, image_size=image_size)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, nesterov=True
    )
    # Create trainer and train
    trainer = Yolov7Trainer(
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        filter_eval_predictions_fn=partial(
            filter_eval_predictions, confidence_threshold=0.01, nms_threshold=0.3
        ),
        callbacks=[
            CalculateMeanAveragePrecisionCallback.create_from_targets_df(
                targets_df=valid_df.query("has_annotation == True")[
                    ["image_id", "xmin", "ymin", "xmax", "ymax", "class_id"]
                ],
                image_ids=set(valid_df.image_id.unique()),
                iou_threshold=0.2,
            ),
            SaveBestModelCallback(watch_metric="map", greater_is_better=True),
            EarlyStoppingCallback(
                early_stopping_patience=3,
                watch_metric="map",
                greater_is_better=True,
                early_stopping_threshold=0.001,
            ),
            *get_default_callbacks(progress_bar=True),
        ],
    )

    trainer.train(
        num_epochs=num_epochs,
        train_dataset=train_yds,
        eval_dataset=eval_yds,
        per_device_batch_size=batch_size,
        create_scheduler_fn=CosineLrScheduler.create_scheduler_fn(
            num_warmup_epochs=5,
            num_cooldown_epochs=5,
            k_decay=2,
        ),
        collate_fn=yolov7_collate_fn,
    )


if __name__ == "__main__":
    main()