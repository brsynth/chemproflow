from chemproflow.model.dmpnn import ChemEmbedding, ChemReadout
from lightning import pytorch as pl
import torch
from torch import nn
from torch_geometric.nn import global_add_pool
from torchmetrics.classification import MultilabelF1Score, MultilabelPrecision, MultilabelRecall


class ModelTcid(pl.LightningModule):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, num_classes, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = ChemEmbedding(node_feat_dim, edge_feat_dim, hidden_dim)
        self.readout = ChemReadout(hidden_dim, num_classes=num_classes)

        self.loss_fn = nn.BCEWithLogitsLoss()
        average = "weighted"
        self.valid_metric = MultilabelF1Score(num_labels=num_classes, average=average)
        self.test_f1 = MultilabelF1Score(num_labels=num_classes, average=average)
        self.test_precision = MultilabelPrecision(num_labels=num_classes, average=average)
        self.test_recall = MultilabelRecall(num_labels=num_classes, average=average)
        self.score_monitor = "val/multilabel-f1"

    def graph_embedding(self, data):
        return self.embedding(data)

    def forward(self, data):
        graph_emb = self.graph_embedding(data)
        return self.readout(graph_emb)

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        targets = batch.y.view_as(logits)
        loss = self.loss_fn(logits, targets)
        self.log("train_loss", loss, batch_size=targets.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        targets = batch.y.view_as(logits)
        preds = torch.sigmoid(logits)
        self.valid_metric.update(preds, targets.int())  # y must be int or bool
        loss = self.loss_fn(logits, targets)
        self.log("valid_loss", loss, batch_size=targets.size(0))
        return loss

    def on_validation_epoch_end(self):
        auc = self.valid_metric.compute()
        self.log("val/multilabel-f1", auc)
        self.valid_metric.reset()

    def test_step(self, batch, batch_idx):
        logits = self(batch)
        targets = batch.y.view_as(logits)
        preds = torch.sigmoid(logits)
        self.test_f1.update(preds, targets.int())
        self.test_precision.update(preds, targets.int())
        self.test_recall.update(preds, targets.int())

    def on_test_epoch_end(self):
        # F1
        f1 = self.test_f1.compute()
        self.log("test/multilabel-f1", f1)
        self.test_f1.reset()
        # Precision
        precision = self.test_precision.compute()
        self.log("test/multilabel-precision", precision)
        self.test_precision.reset()
        # Recall
        recall = self.test_recall.compute()
        self.log("test/multilabel-recall", recall)
        self.test_recall.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

