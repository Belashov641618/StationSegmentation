import os.path
import tarfile
import numpy
from io import BytesIO
from torch.utils.data import Dataset
from typing import Literal, BinaryIO, Optional, Type
from random import Random
from tqdm import tqdm
from json import load, dump

class PatchesDataset(Dataset):
    _path:str
    _split:str
    _ratio:float
    _indices:dict[int,tuple[int,int]]
    _order:list[int]
    _file:Optional[BinaryIO]
    def reindex(self, cached_indices:bool=False, pbar:Optional[Type[tqdm]]=tqdm):
        indices:list[tuple[int,int,int]] = []
        index_path = self._path + ".index"
        if os.path.exists(index_path) and cached_indices:
            with open(index_path, 'r') as file:
                indices = load(file)
        else:
            with tarfile.open(self._path,"r") as archive:
                for file in pbar(archive, desc="Indexing") if pbar is not None else archive:
                    indices.append((int(file.name.split(".")[-2]), file.offset_data, file.size))
            indices = sorted(indices)
            Random(60065).shuffle(indices)
            with open(index_path, 'w') as file:
                dump(indices, file)
        if   self._split == "none":  pass
        elif self._split == "train": indices = indices[:int(len(indices)*self._ratio)]
        elif self._split == "test":  indices = indices[int(len(indices)*self._ratio):]
        else: raise ValueError(f"Unknown split type: {self._split}")
        self._order, self._indices = [], {}
        for index, offset, size in indices:
            self._order.append(index)
            self._indices[index] = (offset, size)

    def __init__(self, archive:str, split:Literal["none","train","test"]="none", ratio:float=0.8, cached_indices:bool=True, pbar:Optional[Type[tqdm]]=tqdm):
        self._path = archive
        self._split = split
        self._ratio = ratio
        self._file = None
        self.reindex(cached_indices, pbar)
    def __getstate__(self):
        state:dict = super().__getstate__()    # type: ignore
        state["_file"] = None
        return state
    def __del__(self):
        if self._file is not None and not self._file.closed:
            self._file.close()
            self._file = None
    def __len__(self):
        return len(self._order)
    def __getitem__(self, index:int):
        if self._file is None: self._file = open(self._path, "rb")
        self._file.seek(self._indices[self._order[index]][0])
        data = self._file.read(self._indices[self._order[index]][1])
        patch = numpy.load(BytesIO(data))
        return patch

class PatchesDatasetDual(PatchesDataset):
    def __getitem__(self, index:int):
        patch = super().__getitem__(index)
        return patch, patch
