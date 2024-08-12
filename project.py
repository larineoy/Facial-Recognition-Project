import numpy as np
from typing import Optional

class Profile:
    "Profile information is saved to database."

    def __init__(self, name: str):
        self.name = name
        self.desc: Optional[np.ndarray] = None
