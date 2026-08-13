from __future__ import annotations
import csv, os
from typing import List, Tuple
from PIL import Image
import numpy as np

class LineDataset:
    def __init__(self, root: str, split: str, image_height: int = 48, max_width: int = 1024):
        self.root, self.split = root, split
        self.image_height, self.max_width = image_height, max_width
        csv_path = os.path.join(root, split, "labels.csv")
        if not os.path.isfile(csv_path): raise FileNotFoundError(csv_path)
        self.items: List[Tuple[str,str]] = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f): self.items.append((row['filename'], row['text']))
    def __len__(self): return len(self.items)
    def load(self, index: int):
        filename, text = self.items[index]
        path = os.path.join(self.root, self.split, 'images', filename)
        image = Image.open(path).convert('L')
        w,h = image.size
        new_w=max(8,min(self.max_width,round(w*self.image_height/max(h,1))))
        arr=np.asarray(image.resize((new_w,self.image_height)),dtype=np.float32)/255.0
        return arr[...,None], text
