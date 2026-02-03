from pathlib import Path
from typing import List, Optional, Tuple
from    PIL import Image
import numpy as np



class ImageLoader:
    """Quan Ly load anh tu thu muc"""

    def __init__(self, root_dir: str):
        self.root_dir: Path = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {root_dir}")
        

    def load_image(self, filename: str) -> np.ndarray:
        "Load anh va  convert sang numpy array"
        file_path = self.root_dir /filename
        with Image.open(file_path) as img:
            return np.array(img.convert('RGB'))
        

        
        