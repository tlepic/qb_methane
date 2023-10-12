from dataset import ImageDataset
from utils import load_train
from methane.models.test import TestModel
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

absolute_path = '/home/octav/Documents/HEC/quantum_black/QB_methane/'

if __name__ == "__main__":
    X, y = load_train(absolute_path+"data")
    ds = ImageDataset(X, y)
    test = DataLoader(ds, 12, shuffle=False)
    print('ok so far')
    model = TestModel()
    model.load_from_checkpoint(absolute_path + 'lightning_logs/version_27/checkpoints/best-model-epoch=17-val_loss=0.33.ckpt')
    model.eval()
    trainer = pl.Trainer()
    
    predictions = trainer.predict(model, test)
    
    for batch_predictions in predictions:
        print(batch_predictions)
