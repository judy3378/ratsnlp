import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

def get_trainer(args, return_trainer_only=True):
    ckpt_path = os.path.abspath(args.downstream_model_dir)
    os.makedirs(ckpt_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        save_top_k=args.save_top_k,
        monitor=args.monitor.split()[1],
        mode=args.monitor.split()[0],
        filename='{epoch}-{val_loss:.2f}',
    )

    # Set up the GPU environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = Trainer(
        max_epochs=args.epochs,
        fast_dev_run=args.test_mode,
        num_sanity_val_steps=None if args.test_mode else 0,
        callbacks=[checkpoint_callback],
        default_root_dir=ckpt_path,
        # For GPU Setup
        deterministic=True,  # Deterministic mode for reproducibility
        gpus=-1,  # Use all available GPUs (You can specify specific GPUs using a list [0, 1])
        precision=16 if args.fp16 else 32,
        strategy='dp',  # Use Data Parallelism strategy
    )

    if return_trainer_only:
        return trainer  # No need to use .to(device) for Trainer object
    else:
        return checkpoint_callback, trainer  # Move both callback and trainer to the GPU if available
