from datetime import datetime
import importlib
import os
import random
import time
import sys


from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


PROJECT_ROOT = os.path.dirname(__file__)
CUSTOM_PATH = os.path.join(PROJECT_ROOT, "Scripts", "from_symile")

if CUSTOM_PATH not in sys.path:
    sys.path.insert(0, CUSTOM_PATH)

from args import parse_args_main
import importlib
datasets = importlib.import_module("datasets")


def create_save_directory(args):
    """
    Create a unique save directory using the current timestamp and a random integer
    between 0 and 9999 in order to reduce the chance of directory name collision when
    scripts are run in parallel.
    """
    randint = random.randint(0, 9999)
    save_dir = args.ckpt_save_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{randint:04d}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def get_data_module(args):
    """
    Returns the appropriate DataModule based on the experiment.
    """
    if args.experiment in ["symile_mimic", "mmft_mimic", "stft_mimic"]:
        dm = datasets.SymileMIMICDataModule
    else:
        raise ValueError("Unsupported experiment name specified.")

    return dm(args)


def get_model_module(args):
    """
    Imports and returns the appropriate model module based on the experiment.
    """
    if args.experiment == "mmft_mimic":
        module = importlib.import_module("models.mmft_mimic_model")
        ModelClass = getattr(module, "MMFTModel")
    elif args.experiment == "stft_mimic":
        module = importlib.import_module("models.stft_mimic_model")
        ModelClass = getattr(module, "STFTModel")
    elif args.experiment == "symile_mimic":
        module = importlib.import_module("models.symile_mimic_model")
        ModelClass = getattr(module, "SymileMIMICModel")
    else:
        raise ValueError("Unsupported experiment name specified.")

    return ModelClass(**vars(args))


def main(args):
    if args.wandb:
        logger = WandbLogger(project="symile", log_model=False,
                             save_dir=args.ckpt_save_dir, id=args.wandb_run_id)
    else:
        logger = False


    if args.experiment in ["mmft_mimic", "stft_mimic"]:
        checkpoint_callback = ModelCheckpoint(dirpath=args.save_dir,
                                                filename="{epoch}",
                                                every_n_epochs=args.check_val_every_n_epoch,
                                                save_top_k=-1)
    else:
        checkpoint_callback = ModelCheckpoint(dirpath=args.save_dir,
                                                  filename="{epoch}-{val_loss:.4f}",
                                                  every_n_epochs=args.check_val_every_n_epoch,
                                                  save_top_k=-1)

    trainer = Trainer(
        enable_checkpointing=False,
        # callbacks=checkpoint_callback,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        deterministic=args.use_seed,
        enable_progress_bar=True,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=1,
        logger=logger,
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        profiler=None
    )

    dm = get_data_module(args)
    dm.setup(stage="fit")

    if args.experiment == "symile_m3" and args.missingness:
        setattr(args, "tokenizer_len", dm.tokenizer_len)

    model = get_model_module(args)

    if args.ckpt_path == None:
        print("Training model from scratch!")
        if args.experiment != "mmft_mimic" and args.experiment != "stft_mimic":
            trainer.fit(model, datamodule=dm)
        else:
            trainer.fit(model, train_dataloaders=dm.train_dataloader())
    else:
        print("Loading checkpoint from ", args.ckpt_path)
        if args.experiment != "mmft_mimic" and args.experiment != "stft_mimic":
            trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)
        else:
            trainer.fit(model, train_dataloaders=dm.train_dataloader(), ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    start = time.time()

    args = parse_args_main()

    save_dir = create_save_directory(args)
    setattr(args, "save_dir", save_dir)
    print("\nSaving to: ", save_dir)

    if args.use_seed:
        seed_everything(args.seed, workers=True)

    if args.experiment in ["mmft_mimic", "symile_mimic", "stft_mimic"]:
        main(args)
    else:
        raise ValueError("Unsupported experiment name specified.")

    end = time.time()
    total_time = (end - start)/60
    print(f"Script took {total_time:.4f} minutes")
    print(f"Script took {total_time:.4f} minutes")

