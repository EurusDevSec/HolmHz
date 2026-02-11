# Thuc han OOP va type hints

from typing import List, Tuple, Optional
from pathlib import Path

# ImagePathManager Quan ly duong dan anh

class ImagePathManager:
    """Quan ly duong dan anh trong dataset"""
    def __init__(self, root_dir:str):
        self.root = Path(root_dir)
        self._validate_root()

    def _validate_root(self) -> None:
        """Kiem tra thu muc ton tai"""
        if not self.root.exists():
            raise FileNotFoundError(f"Directory not found: {self.root}")
        
    def get_image_paths(self, extension: str  = "*.jpg") -> List[Path]:
        """Lay tat ca duong dan anh"""
        return list(self.root.rglob(extension))
    
    def split_by_label(self)-> Tuple[List[Path], List[Path]]:
        """Tach anh theo label(real/fake)"""
        real = list (self.root.glob("real/*"))
        fake = list(self.root.glob("fake/*"))
        return real, fake
    

# Test 
if __name__ == "__main__":
    manager = ImagePathManager("data/processed/train")
    print(f"Found{len(manager.get_image_paths())} images")
    