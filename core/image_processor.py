# core/image_processor.py
import os
from PIL import Image
import numpy as np
import config.constants as Const
from typing import Tuple, List


class ImageProcessor:
    def __init__(self, target_size=(64, 64)):
        self.target_size = target_size

    @classmethod
    def process_load_image(self, folder_path: str) -> Tuple[List[str], List[str]]:
        """Load images from a folder and extract labels based on subfolder names."""
        image_paths = []
        labels = []
        if not os.path.exists(folder_path):
            return [], []

        for person_name in os.listdir(folder_path):
            person_dir = os.path.join(folder_path, person_name)
            if os.path.isdir(person_dir):
                for img_file in os.listdir(person_dir):
                    if img_file.endswith((".jpg", ".png", ".jpeg")):
                        img_path = os.path.join(person_dir, img_file)
                        image_paths.append(img_path)
                        labels.append(
                            person_name
                        )  # Ensure this is a string, not a list
        return image_paths, labels

    def preprocess_image(self, my_dataset) -> np.ndarray:
        """Tiền xử lý ảnh: chuyển thành grayscale, resize, và làm phẳng."""
        # Đọc ảnh và chuyển thành grayscale

        image_vectors = []
        if not my_dataset:
            print("No images found in directory")
            return None
        print("test")

        for file_path in my_dataset:
            try:
                with Image.open(file_path) as img:
                    # Resize to consistent size
                    img_resized = img.resize(
                        self.target_size, Image.Resampling.LANCZOS
                    )  # Better quality resizing

                    # Convert to grayscale
                    img_gray = img_resized.convert("L")

                    # Convert to numpy array and flatten directly
                    vector = (
                        np.array(img_gray, dtype=np.float32).flatten() / 255.0
                    )  # To vector 1D
                    image_vectors.append(vector)
                    for i in list(vector):
                        if i < 0:
                            print(file_path)
                            break
            except (IOError, ValueError) as e:
                print(f"Error processing {file_path}: {e}")
                continue

        if not image_vectors:
            print("No images were successfully processed")
            return None

        # Stack as columns to form V.T => V(n,m)  -> V(m, n)
        # Note:  V đại diện cho đặc trưng của một ảnh, và mỗi hàng tương ứng với một pixel trong không gian đã làm phẳng.
        V = np.array(image_vectors).T
        print(f"V shape: {V.shape}")
        return V
