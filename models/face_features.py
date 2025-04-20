# models/face_features.py
from pydantic import BaseModel
from typing import List

class FaceFeatures(BaseModel):
    features: List[float]