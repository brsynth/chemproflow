from chemproflow.model.dmpnn import ChemEmbedding, ChemReadout
from lightning import pytorch as pl
import torch
from torch import nn
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall


class ModelTransport(pl.LightningModule):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, num_classes, lr=1e-3, pos_prior=0.5, pu_type='elkan-noto'):
        super().__init__()
        if pu_type.lower() != 'elkan-noto':
            raise ValueError("Only the Elkan-Noto PU strategy is supported.")
        self.save_hyperparameters()
        self.embedding = ChemEmbedding(node_feat_dim, edge_feat_dim, hidden_dim)
        self.readout = ChemReadout(hidden_dim, num_classes=num_classes)
        
        self.bce = nn.BCEWithLogitsLoss()
        self.f1 = BinaryF1Score()
        self.test_acc = BinaryAccuracy()
        self.test_f1 = BinaryF1Score()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.register_buffer('elkan_c', torch.tensor(1.0))
        self.elkan_calibration_loader = None
        self.score_monitor = 'val/binary-f1'

    def graph_embedding(self, data):
        return self.embedding(data)

    def forward(self, data):
        graph_emb = self.graph_embedding(data)
        return self.readout(graph_emb)

    def _correct_probs(self, probs: torch.Tensor) -> torch.Tensor:
        c = torch.clamp(self.elkan_c, min=1e-6)
        return torch.clamp(probs / c, max=1.0)

    def set_elkan_calibration_loader(self, loader):
        self.elkan_calibration_loader = loader

    def estimate_elkan_c(self, loader):
        was_training = self.training
        self.eval()
        pos_probs = []
        device = self.elkan_c.device
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                y = batch.y.float()
                if not (y == 1).any():
                    continue
                logits = self(batch).squeeze(-1)
                probs = torch.sigmoid(logits)
                pos_probs.append(probs[y == 1].detach())
        if was_training:
            self.train()
        if not pos_probs:
            raise ValueError("Cannot estimate Elkan-Noto c: calibration split has no positive labels.")
        c_hat = torch.cat(pos_probs).mean().to(device)
        self.elkan_c.copy_(torch.clamp(c_hat, min=1e-6))
        return self.elkan_c

    def training_step(self, batch, batch_idx):
        logits = self(batch).squeeze(-1)
        y = batch.y.float()
        loss = self.bce(logits, y)
        batch_size = y.size(0)
        self.log('train_loss', loss, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch).squeeze(-1)
        y = batch.y.float()
        loss = self.bce(logits, y)
        probs = torch.sigmoid(logits)
        corrected = self._correct_probs(probs)
        self.f1.update(corrected, y.int())
        self.log('valid_loss', loss, prog_bar=False, batch_size=y.size(0))
        return loss

    def on_validation_epoch_start(self):
        if self.trainer is not None and self.trainer.sanity_checking:
            return
        if self.elkan_calibration_loader is None:
            return
        c_hat = self.estimate_elkan_c(self.elkan_calibration_loader)
        self.log('elkan_c', c_hat, prog_bar=True)

    def on_validation_epoch_end(self):
        f1 = self.f1.compute()
        self.log('val/binary-f1', f1, prog_bar=True)
        self.f1.reset()

    def test_step(self, batch, batch_idx):
        logits = self(batch).squeeze(-1)
        probs = torch.sigmoid(logits)
        corrected = self._correct_probs(probs)
        self.test_acc.update(corrected, batch.y.int())
        self.test_f1.update(corrected, batch.y.int())
        self.test_precision.update(corrected, batch.y.int())
        self.test_recall.update(corrected, batch.y.int())

    def on_test_epoch_end(self):
        # acc
        acc = self.test_acc.compute()
        self.log("test/binary-acc", acc)
        self.test_acc.reset()
        # f1
        f1 = self.test_f1.compute()
        self.log("test/binary-f1", f1)
        self.test_f1.reset()
        # precision
        precision = self.test_precision.compute()
        self.log("test/binary-precision", precision)
        self.test_precision.reset()
        # recall
        recall = self.test_recall.compute()
        self.log("test/binary-recall", recall)
        self.test_recall.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def elkan_correct_probs(self, probs: torch.Tensor) -> torch.Tensor:
        c_value = float(self.elkan_c.item())
        if c_value <= 0:
            c_value = 1.0
        return torch.clamp(probs / c_value, max=1.0)
