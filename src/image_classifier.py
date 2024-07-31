from pathlib import Path
from os import PathLike
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from typing import Literal
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
import tempfile
import zipfile

type kernelTypes = Literal["linear", "poly", "rbf", "sigmoid", "precomputed"]


class ImageClassifier:
    def __init__(self, img_size: int = 64, kernel: kernelTypes = "linear") -> None:
        self.img_size: int = img_size
        self.model = svm.SVC(kernel=kernel)

    def load_images_from_dataset(self, dataset_directory: str):
        data = []
        temp_path = Path(dataset_directory)
        temp_path = temp_path / "PetImages"

        for root, dirs, _ in temp_path.walk():
            for label in dirs:
                label_dir = root / label
                for file in label_dir.iterdir():
                    if file.suffix == ".jpg":
                        file_path = file
                        data.append({"file_path": str(file_path), "label": label})

        df = pd.DataFrame(data)
        return df

    def extract_dataset(self, zip_file_path: PathLike):
        with tempfile.TemporaryDirectory() as tempdir:
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(tempdir)

        return tempdir
