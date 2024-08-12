import numpy as np
from typing import Optional

class Profile:
    "Profile information is saved to database."

    def __init__(self, name: str):
        self.name = name
        self.desc: Optional[np.ndarray] = None

    def mean_desc(self) -> np.ndarray:
        if self.desc is not None:
            return self.desc.mean(axis=0)
        else:
            raise ValueError("No descriptors available.")

