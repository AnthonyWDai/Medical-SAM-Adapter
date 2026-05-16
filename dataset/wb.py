from pathlib import Path
from typing import Optional, Callable, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import random_click


class WholeBody(Dataset):
    """
    Dataset for whole-body volumes stored as:

        Dataset/
            sample_001/
                data.npy       # expected shape: [D, H, W]
                gt_sparse.npy  # expected shape: [D, H, W]

    Returns:
        {
            "image": Tensor [1, image_size, image_size, D],
            "label": Tensor [1, out_size, out_size, D],
            "p_label": int,
            "pt": click point or None,
            "image_meta_dict": {"filename_or_obj": sample_name}
        }
    """

    def __init__(
        self,
        args,
        data_path: str,
        transform: Optional[Callable] = None,
        transform_msk: Optional[Callable] = None,
        mode: str = "Training",
        prompt: str = "click",
        plane: bool = False,
    ):
        self.args = args
        self.root = Path(data_path) / "Dataset"
        self.mode = mode
        self.prompt = prompt
        self.plane = plane

        self.img_size = int(args.image_size)
        self.out_size = int(args.out_size)

        self.transform = transform
        self.transform_msk = transform_msk

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")

        self.samples = sorted(
            p for p in self.root.iterdir()
            if p.is_dir()
            and (p / "data.npy").exists()
            and (p / "gt_sparse.npy").exists()
        )

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_dir = self.samples[index]

        image_ct = np.load(sample_dir / "data_0000.npy")
        image_pet = np.load(sample_dir / "data_0001.npy")
        mask = np.load(sample_dir / "gt_sparse.npy")

        # Convert [H, W, D] numpy arrays to tensors.
        image_ct = torch.from_numpy(np.ascontiguousarray(image_ct)).float()
        image_pet = torch.from_numpy(np.ascontiguousarray(image_pet)).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).float()


        # Treat D as channels and resize only spatial dimensions H, W.
        # Shape: [H, W, D] -> [1, H, W, D]
        image_ct = image_ct.unsqueeze(0)
        mask = mask.unsqueeze(0)

        image_ct = F.interpolate(
            image_ct,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )

        mask = F.interpolate(
            mask,
            size=(self.out_size, self.out_size),
            mode="nearest",
        )

        # Back to [H, W, D]
        image_ct = image_ct.squeeze(0)
        mask = mask.squeeze(0)

        # Binary mask, matching original behavior.
        mask = mask.clamp_(0, 1).to(torch.int64)

        # Binary/integer mask.
        mask = mask.clamp_(0, 1).to(torch.int64)

        # Stack modalities as channels.
        # image: [C, H, W, D] where C=3: CT, PET, CT/PET
        # mask:  [1, H, W, D]
        image = torch.stack([
            image_ct, image_pet, (image_ct + image_pet) / 2
        ], dim=0).contiguous()
        mask = mask.unsqueeze(0).contiguous()

        if self.transform is not None:
            image = self.transform(image)

        if self.transform_msk is not None:
            mask = self.transform_msk(mask)

        point_label = 1
        pt = None
        if self.prompt == "click":
            point_label, pt = random_click(mask.cpu().numpy(), point_label)

        return {
            "image": image,
            "label": mask,
            "p_label": point_label,
            "pt": pt,
            "image_meta_dict": {
                "filename_or_obj": sample_dir.name,
                "modalities": ["ct", "pet"],
            },
        }