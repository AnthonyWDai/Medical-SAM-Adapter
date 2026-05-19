import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

import cfg
import function_wb
from conf import settings
from dataset import *
from utils import *


def setup_path_helper(args):
    """
    Only rank 0 creates log/checkpoint/sample directories.
    Then broadcast path_helper to all ranks.
    """
    if is_main_process():
        path_helper = set_log_dir(
            "%s/msadapter/logs" % os.environ["exp"],
            args.exp_name
        )
    else:
        path_helper = None

    if dist.is_available() and dist.is_initialized():
        obj_list = [path_helper]
        dist.broadcast_object_list(obj_list, src=0)
        path_helper = obj_list[0]
        dist.barrier()

    args.path_helper = path_helper
    return args


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend="nccl",
            init_method="env://"
        )

        return True, rank, local_rank, world_size

    return False, 0, 0, 1


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def reduce_scalar(value, device):
    if not dist.is_available() or not dist.is_initialized():
        return value

    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor.item()


class NullWriter:
    def __getattr__(self, name):
        def noop(*args, **kwargs):
            pass
        return noop


def main():
    args = cfg.parse_args()

    ddp_enabled, rank, local_rank, world_size = setup_ddp()

    args.rank = rank
    args.local_rank = local_rank
    args.world_size = world_size

    if ddp_enabled:
        args.distributed = "ddp"

    seed = args.seed + rank
    set_seed(seed)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    # Logging only on rank 0
    # ---------------------------------------------------------
    args = setup_path_helper(args)

    if is_main_process():
        logger = create_logger(args.path_helper["log_path"])
        logger.info(args)
    else:
        logger = None

    if is_main_process():
        logger = create_logger(args.path_helper["log_path"])
        logger.info(args)
    else:
        logger = None

    # ---------------------------------------------------------
    # Build model
    # ---------------------------------------------------------
    # Important:
    # If get_network internally wraps with DataParallel/DDP,
    # disable that and wrap manually here.
    net = get_network(
        args,
        args.net,
        use_gpu=args.gpu,
        gpu_device=device,
        distribution="none"
    )

    net = net.to(device)

    # ---------------------------------------------------------
    # Load pretrain before DDP wrapping
    # ---------------------------------------------------------
    if args.pretrain:
        weights = torch.load(args.pretrain, map_location="cpu")
        net.load_state_dict(weights, strict=False)

    # ---------------------------------------------------------
    # Freeze modules before optimizer and before DDP wrapping
    # ---------------------------------------------------------
    if args.freeze >= 1:
        for p in net.prompt_encoder.parameters():
            p.requires_grad = False

    if args.freeze >= 2:
        for p in net.image_encoder.parameters():
            p.requires_grad = False

    if args.freeze >= 3:
        for p in net.mask_decoder.iou_token.parameters():
            p.requires_grad = False
        for p in net.mask_decoder.mask_tokens.parameters():
            p.requires_grad = False

    if args.freeze >= 4:
        for p in net.mask_decoder.transformer.parameters():
            p.requires_grad = False

    # Optional, useful if you have BatchNorm and want synchronized stats.
    # For SAM-like models this may not be needed.
    # if ddp_enabled:
    #     net = nn.SyncBatchNorm.convert_sync_batchnorm(net)

    # ---------------------------------------------------------
    # Wrap with DDP
    # ---------------------------------------------------------
    if ddp_enabled:
        net = DDP(
            net,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )

    # ---------------------------------------------------------
    # Optimizer should only receive trainable params
    # ---------------------------------------------------------
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
    )

    # ---------------------------------------------------------
    # Resume checkpoint
    # ---------------------------------------------------------
    start_epoch = 0
    best_dice = 0.0
    best_tol = 1e4

    if args.weights != 0:
        if is_main_process():
            print(f"=> resuming from {args.weights}")

        assert os.path.exists(args.weights)
        checkpoint_file = os.path.join(args.weights)
        assert os.path.exists(checkpoint_file)

        checkpoint = torch.load(
            checkpoint_file,
            map_location=device
        )

        start_epoch = checkpoint["epoch"]
        best_tol = checkpoint.get("best_tol", best_tol)

        if ddp_enabled:
            net.module.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            net.load_state_dict(checkpoint["state_dict"], strict=False)

        # Optional:
        # optimizer.load_state_dict(checkpoint["optimizer"])

        if "path_helper" in checkpoint:
            args.path_helper = checkpoint["path_helper"]

        if is_main_process():
            print(f"=> loaded checkpoint {checkpoint_file} epoch {start_epoch}")

    # ---------------------------------------------------------
    # Dataloaders
    # ---------------------------------------------------------
    nice_train_loader, nice_test_loader = get_dataloader(args)

    # ---------------------------------------------------------
    # TensorBoard and checkpoint path only on rank 0
    # ---------------------------------------------------------
    if is_main_process():
        checkpoint_path = "%s/msadapter/%s/%s" % (
            os.environ["exp"],
            args.dataset,
            settings.TIME_NOW
        )

        if not os.path.exists(settings.LOG_DIR):
            os.mkdir(settings.LOG_DIR)

        writer = SummaryWriter(
            log_dir=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW)
        )

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        checkpoint_path = os.path.join(
            checkpoint_path,
            "{net}-{epoch}-{type}.pth"
        )
    else:
        writer = NullWriter()
        checkpoint_path = None

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------
    for epoch in range(start_epoch, settings.EPOCH):

        # Required for DistributedSampler shuffling
        if ddp_enabled and hasattr(nice_train_loader.sampler, "set_epoch"):
            nice_train_loader.sampler.set_epoch(epoch)

        net.train()

        time_start = time.time()

        loss = function_wb.train_sam(
            args,
            net,
            optimizer,
            nice_train_loader,
            epoch,
            writer,
            vis=args.vis if is_main_process() else False
        )

        loss = reduce_scalar(loss, device)

        time_end = time.time()

        if is_main_process():
            logger.info(f"Train loss: {loss} || @ epoch {epoch}.")
            print("time_for_training ", time_end - time_start)

        scheduler.step()

        # -----------------------------------------------------
        # Validation/checkpoint only on rank 0
        # -----------------------------------------------------
        should_validate = (epoch and epoch % args.val_freq == 0) or epoch == settings.EPOCH - 1

        if should_validate:
            if is_main_process():
                net.eval()

                with torch.no_grad():
                    if args.dataset != "REFUGE":
                        tol, (eiou, edice) = function_wb.validation_sam(
                            args,
                            nice_test_loader,
                            epoch,
                            net,
                            writer
                        )

                        logger.info(
                            f"Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}."
                        )

                    else:
                        tol, metrics = function_wb.validation_sam(
                            args,
                            nice_test_loader,
                            epoch,
                            net,
                            writer
                        )

                        eiou_cup, eiou_disc, edice_cup, edice_disc = metrics

                        logger.info(
                            f"Total score: {tol}, "
                            f"IOU_CUP: {eiou_cup}, IOU_DISC: {eiou_disc}, "
                            f"DICE_CUP: {edice_cup}, DICE_DISC: {edice_disc} "
                            f"|| @ epoch {epoch}."
                        )

                        # Choose what "best dice" means for REFUGE.
                        edice = 0.5 * (edice_cup + edice_disc)

                if ddp_enabled:
                    sd = net.module.state_dict()
                else:
                    sd = net.state_dict()

                if edice > best_dice:
                    best_dice = edice
                    best_tol = tol

                    save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "model": args.net,
                            "state_dict": sd,
                            "optimizer": optimizer.state_dict(),
                            "best_tol": best_tol,
                            "best_dice": best_dice,
                            "path_helper": args.path_helper,
                        },
                        True,
                        args.path_helper["ckpt_path"],
                        filename="best_dice_checkpoint.pth"
                    )

            if ddp_enabled:
                dist.barrier()

    if is_main_process():
        writer.close()

    cleanup_ddp()


if __name__ == "__main__":
    main()