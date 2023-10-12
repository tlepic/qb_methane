import torch
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from methane import Gasnet2
from sklearn.metrics import roc_auc_score

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, targets, base_path, transform=None):
        super().__init__()
        self.image_paths = [base_path + p + ".tif" for p in image_paths]
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        targets = self.targets[index]
        image = Image.open(self.image_paths[index])
        if self.transform:
            image = self.transform(image)
        to_tensor = transforms.ToTensor()
        image = to_tensor(image).float()
        return image, targets
    
    def __len__(self):
        return len(self.image_paths)

class NumericAugmentation():
    """
    Class to apply three pre-trained Vision Models from checkpoints.

    Contains:
        - method to apply voting to the test set.
        - method to augment the metadata table with outputs and voting.
    """
    @staticmethod
    def load_model_from_checkpoint(path, model_type):
        if model_type == "Gasnet":
            model = Gasnet2()
        elif model_type == "Gasnet1":
            model = Gasnet2()
        elif model_type == "Gasnet2":
            model = Gasnet2()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model

    @staticmethod
    def data_loader_and_augmentation(metadata_csv_path, model_info):
        df = pd.read_csv(metadata_csv_path)
        for i, _ in enumerate(model_info):
            df[f'model_{i+1}'] = 0.0

        transform = transforms.Resize((64, 64))
        base_path = '/home/octav/Documents/HEC/quantum_black/QB_methane/data/train_data/'
        dataset = ImageDataset(df['path'].tolist(), df['plume'].tolist(), base_path, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        models = [NumericAugmentation.load_model_from_checkpoint(path, model_type) for path, model_type in model_info]

        for idx, (images, _) in enumerate(dataloader):
            for i, model in enumerate(models):
                with torch.no_grad():
                    output = model(images)
                    probability = torch.sigmoid(output).item()
                    df.at[idx, f'model_{i+1}'] = probability

        df['vote'] = df[['model_1', 'model_2', 'model_3']].mean(axis=1)
 #      df['vote'] = (df['vote'] > 0.2).astype(int)Using 0.5 grossly under predicts true leaks.
        df.to_csv(metadata_csv_path.replace('.csv', '_augmented.csv'), index=False)
        print(NumericAugmentation.vote_accuracy(df))
        return df
    
    @staticmethod
    def vote_accuracy(df):
        """Uses df.vote column to calculate accuracy"""
        df['plume'] = df['plume'].map({'yes': 1, 'no': 0})  
        predicted = (df['vote'] > 0.5).astype(int)
        accuracy = (predicted == df['plume']).mean() * 100
        
        roc_auc = roc_auc_score(df['plume'], df['vote'])
        
        print(f"The model predicts {predicted.mean() * 100:.2f}% of leaks in the dataset. Truth is {(df['plume']==1).sum()/len(df['plume']) * 100:.2f}%.")
        print(f"The ROC AUC score is {roc_auc:.2f}")
        return f"The accuracy of the voting model is {accuracy:.2f}%"

    @staticmethod
    def apply_augmentation():
        absolute_path = '/home/octav/Documents/HEC/quantum_black/QB_methane/'
        model_info = [
            (absolute_path + 'lightning_logs/version_21/checkpoints/best-model-epoch=17-val_loss=0.42.ckpt', 'Gasnet2'), 
            (absolute_path + 'lightning_logs/version_20/checkpoints/best-model-epoch=17-val_loss=0.42.ckpt', 'Gasnet1'), 
            (absolute_path + 'lightning_logs/version_19/checkpoints/best-model-epoch=17-val_loss=0.42.ckpt', 'Gasnet')
        ]
        metadata_csv_path = absolute_path + 'data/train_data/metadata.csv'
        NumericAugmentation.data_loader_and_augmentation(metadata_csv_path, model_info)

if __name__ == "__main__":
    NumericAugmentation.apply_augmentation()

