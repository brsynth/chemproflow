from typing import Optional

from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


class WrapModel:
    def __init__(self, model):
        self.model = model
        self.trainer = None
        self.checkpoint_callback = None

    def train(
        self,
        outdir: str,
        train_loader,
        valid_loader,
        logger=None,
        basename: Optional[str] = None,
        early_stopping: float = 0.005,
    ):
        filename = "base_{epoch:02d}"
        if basename:
            filename += basename

        checkpoint_callback = ModelCheckpoint(
            dirpath=outdir,
            monitor=self.model.score_monitor,
            filename=filename,
            mode="max",
        )
        self.checkpoint_callback = checkpoint_callback
        early_stop_callback = EarlyStopping(
            monitor=self.model.score_monitor,
            min_delta=early_stopping,
            patience=5,
            verbose=True,
            mode="max",
        )
        self.trainer = pl.Trainer(
            default_root_dir=outdir,
            enable_checkpointing=True,
            enable_progress_bar=True,
            logger=logger if logger is not None else False, # Conflict Numpy version -> tensorboard
            accelerator="auto",  # "gpu" if str(device).startswith("cuda") else "cpu",
            strategy="auto",
            max_epochs=50,
            callbacks=[checkpoint_callback, early_stop_callback],
            devices="auto",
            num_nodes=1,
            deterministic=True,
        )
        self.trainer.fit(self.model, train_loader, valid_loader)

    def test(self, test_loader):
        results = self.trainer.test(self.model, test_loader)
        return results
