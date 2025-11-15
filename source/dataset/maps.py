import os
from torchvision.transforms.functional import to_tensor, invert
from torch.utils.data import Dataset
from PIL import Image
from typing import Literal
from random import Random

class MapsDataset(Dataset):
    _split:str
    _ratio:float
    _input_path:str
    _target_path:str
    _indices:list[int]
    def __init__(self, input_path:str, target_path:str, split:Literal["none","train","test"]="none", ratio:float=0.8):
        super().__init__()
        self._input_path = input_path
        self._target_path = target_path
        self._split = split
        self._ratio = ratio
        input_indices  = {int(file.split(".")[-2]) for file in os.listdir(input_path)}
        target_indices = {int(file.split(".")[-2]) for file in os.listdir(target_path)}
        self._indices = sorted(list(input_indices & target_indices))
        Random(60065).shuffle(self._indices)
        if   split == "none": pass
        elif split == "train": self._indices = self._indices[:int(len(self._indices)*ratio)]
        elif split == "test":  self._indices = self._indices[int(len(self._indices)*ratio):]
        else: raise ValueError(f"Unknow split type: {split}")
    def __len__(self):
        return len(self._indices)
    def __getitem__(self, index:int):
        input_image = invert(to_tensor(Image.open(os.path.join(self._input_path, f"{self._indices[index]}.png"))))
        target_image = to_tensor(Image.open(os.path.join(self._target_path, f"{self._indices[index]}.png")))
        return input_image, target_image
