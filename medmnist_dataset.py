from typing import Optional, Literal, Callable
from medmnist import INFO
from enum import Enum
import medmnist


class DataFlag(Enum):
    PATHMNIST = "pathmnist"
    OCTMNIST = "octmnist"
    PNEUMONIAMNIST = "pneumoniamnist"
    CHESTMNIST = "chestmnist"
    DERMAMNIST = "dermamnist"
    RETINAMNIST = "retinamnist"
    BREASTMNIST = "breastmnist"
    BLOODMNIST = "bloodmnist"
    TISSUEMNIST = "tissuemnist"
    ORGANAMNIST = "organamnist"
    ORGANCNMIST = "organcmnist"
    ORGANSNMIST = "organsmnist"
    ORGANMNIST3D = "organmnist3d"
    NODULEMNIST3D = "nodulemnist3d"
    ADRENALMNIST3D = "adrenalmnist3d"
    FRACTUREMNIST3D = "fracturemnist3d"
    VESSELMNIST3D = "vesselmnist3d"
    SYNAPSEMNIST3D = "synapsemnist3d"


def get_medmnist_dataset(
        data_flag: DataFlag,
        mode: Literal["train", "val", "test"] = "train",
        data_transform: Optional[Callable] = None,
        size: Literal[28, 64, 128, 224] = 28,
        download: bool = False,
        mmap_mode: Optional[Literal["r"]] = None
    ):

    info = INFO[data_flag.value]

    try:
        DataClass = getattr(medmnist, info['python_class'])
        dataset = DataClass(
            split=mode,
            transform=data_transform,
            download=download,
            size=size,
            mmap_mode=mmap_mode
        )
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        dataset = None
    
    return dataset

    