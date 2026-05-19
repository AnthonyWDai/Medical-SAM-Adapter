import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from utils import *
from .atlas import Atlas
from .brat import Brat
from .ddti import DDTI
from .isic import ISIC2016
from .kits import KITS
from .lidc import LIDC
from .lnq import LNQ
from .pendal import Pendal
from .refuge import REFUGE
from .segrap import SegRap
from .stare import STARE
from .toothfairy import ToothFairy
from .wbc import WBC
from .wb import WholeBody


def _is_ddp(args):
    return (
        getattr(args, "distributed", "none") in ["ddp", "DDP", "distributed"]
        and dist.is_available()
        and dist.is_initialized()
    )


def _num_workers(args):
    return getattr(args, "num_workers", 8)


def _split_dataset(dataset, val_ratio=0.3, seed=1234):
    """
    Deterministic split. All DDP ranks will produce the same split.
    """
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)

    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    split = int(np.floor(val_ratio * dataset_size))

    val_indices = indices[:split].tolist()
    train_indices = indices[split:].tolist()

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset


def _build_loader(
    dataset,
    args,
    train=True,
    distributed=True,
    drop_last=None,
):
    """
    Builds a normal or DDP DataLoader.

    For DDP training:
        - use DistributedSampler
        - set shuffle=False in DataLoader
        - call loader.sampler.set_epoch(epoch) in training loop

    For validation:
        - by default this returns a non-distributed loader
        - use only on rank 0 unless you implement distributed metric reduction
    """
    ddp = _is_ddp(args) and distributed

    if drop_last is None:
        drop_last = train

    if ddp:
        sampler = DistributedSampler(
            dataset,
            shuffle=train,
            drop_last=drop_last,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = train

    num_workers = _num_workers(args)

    loader = DataLoader(
        dataset,
        batch_size=args.b,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )

    return loader


def get_dataloader(args):
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
    ])

    transform_train_seg = transforms.Compose([
        transforms.Resize((args.out_size, args.out_size)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
    ])

    transform_test_seg = transforms.Compose([
        transforms.Resize((args.out_size, args.out_size)),
        transforms.ToTensor(),
    ])

    seed = getattr(args, "seed", 1234)

    # ------------------------------------------------------------------
    # Datasets with explicit train/test split
    # ------------------------------------------------------------------
    if args.dataset == "isic":
        train_dataset = ISIC2016(
            args,
            args.data_path,
            transform=transform_train,
            transform_msk=transform_train_seg,
            mode="Training",
        )

        test_dataset = ISIC2016(
            args,
            args.data_path,
            transform=transform_test,
            transform_msk=transform_test_seg,
            mode="Test",
        )

    elif args.dataset == "REFUGE":
        train_dataset = REFUGE(
            args,
            args.data_path,
            transform=transform_train,
            transform_msk=transform_train_seg,
            mode="Training",
        )

        test_dataset = REFUGE(
            args,
            args.data_path,
            transform=transform_test,
            transform_msk=transform_test_seg,
            mode="Test",
        )

    elif args.dataset == "DDTI":
        train_dataset = DDTI(
            args,
            args.data_path,
            transform=transform_train,
            transform_msk=transform_train_seg,
            mode="Training",
        )

        test_dataset = DDTI(
            args,
            args.data_path,
            transform=transform_test,
            transform_msk=transform_test_seg,
            mode="Test",
        )

    # ------------------------------------------------------------------
    # Datasets using random internal split
    # ------------------------------------------------------------------
    elif args.dataset == "LIDC":
        dataset = LIDC(
            args,
            data_path=args.data_path,
            transform=transform_train,
            transform_msk=transform_train_seg,
        )
        train_dataset, test_dataset = _split_dataset(dataset, val_ratio=0.2, seed=seed)

    elif args.dataset == "Brat":
        dataset = Brat(
            args,
            data_path=args.data_path,
            transform=transform_train,
            transform_msk=transform_train_seg,
        )
        train_dataset, test_dataset = _split_dataset(dataset, val_ratio=0.3, seed=seed)

    elif args.dataset == "STARE":
        dataset = STARE(
            args,
            data_path=args.data_path,
            transform=transform_train,
            transform_msk=transform_train_seg,
        )
        train_dataset, test_dataset = _split_dataset(dataset, val_ratio=0.2, seed=seed)

    elif args.dataset == "kits":
        dataset = KITS(
            args,
            data_path=args.data_path,
            transform=transform_train,
            transform_msk=transform_train_seg,
        )
        train_dataset, test_dataset = _split_dataset(dataset, val_ratio=0.3, seed=seed)

    elif args.dataset == "WBC":
        dataset = WBC(
            args,
            data_path=args.data_path,
            transform=transform_train,
            transform_msk=transform_train_seg,
        )
        train_dataset, test_dataset = _split_dataset(dataset, val_ratio=0.3, seed=seed)

    elif args.dataset == "segrap":
        dataset = SegRap(
            args,
            data_path=args.data_path,
            transform=transform_train,
            transform_msk=transform_train_seg,
        )
        train_dataset, test_dataset = _split_dataset(dataset, val_ratio=0.3, seed=seed)

    elif args.dataset == "toothfairy":
        dataset = ToothFairy(
            args,
            data_path=args.data_path,
            transform=transform_train,
            transform_msk=transform_train_seg,
        )
        train_dataset, test_dataset = _split_dataset(dataset, val_ratio=0.3, seed=seed)

    elif args.dataset == "wb":
        transform_train = Compose([
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_size, args.roi_size, args.chunk),
                pos=1,
                neg=1,
                num_samples=args.num_sample,
                image_key="image",
                image_threshold=0,
            ),
        ])

        dataset = WholeBody(
            args,
            data_path=args.data_path,
            transform=transform_train,
            transform_msk=transform_train_seg,
        )
        train_dataset, test_dataset = _split_dataset(dataset, val_ratio=0.3, seed=seed)

    elif args.dataset == "atlas":
        dataset = Atlas(
            args,
            data_path=args.data_path,
            transform=transform_train,
            transform_msk=transform_train_seg,
        )
        train_dataset, test_dataset = _split_dataset(dataset, val_ratio=0.3, seed=seed)

    elif args.dataset == "pendal":
        dataset = Pendal(
            args,
            data_path=args.data_path,
            transform=transform_train,
            transform_msk=transform_train_seg,
        )
        train_dataset, test_dataset = _split_dataset(dataset, val_ratio=0.3, seed=seed)

    elif args.dataset == "lnq":
        dataset = LNQ(
            args,
            data_path=args.data_path,
            transform=transform_train,
            transform_msk=transform_train_seg,
        )
        train_dataset, test_dataset = _split_dataset(dataset, val_ratio=0.3, seed=seed)

    else:
        raise ValueError(f"Dataset {args.dataset} is not supported.")

    # ------------------------------------------------------------------
    # Build loaders
    # ------------------------------------------------------------------

    # Training should be distributed under DDP.
    nice_train_loader = _build_loader(
        train_dataset,
        args,
        train=True,
        distributed=True,
        drop_last=True,
    )

    # Validation/test is intentionally non-distributed here.
    # Use this loader only on rank 0.
    nice_test_loader = _build_loader(
        test_dataset,
        args,
        train=False,
        distributed=False,
        drop_last=False,
    )

    return nice_train_loader, nice_test_loader