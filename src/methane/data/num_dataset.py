import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn.metrics import roc_auc_score
from methane import Gasnet2
from dataset import ImageDataset
from utils import load_train


class EnsembleAugmentation:
    def __init__(self, base_path, metadata_csv_path, model_info):
        self.base_path = base_path
        self.metadata_csv_path = metadata_csv_path
        self.model_info = model_info

    def load_metadata(self):
        df = pd.read_csv(self.metadata_csv_path)
        df['plume'] = df['plume'].map({'yes': 1, 'no': 0})  
        for i, _ in enumerate(self.model_info):
            df[f'model_{i+1}'] = 0.0
        return df

    def create_dataloader(self, df):
        X_train, y_train = load_train(os.path.join(self.base_path, 'data'))
        transform_flag = True 
        dataset = ImageDataset(X_train, y_train, transform=transform_flag, extra_feature=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
        return dataloader

    @staticmethod
    def load_model_from_checkpoint(path, model_type):
        """
        Load model from checkpoint based on the given model type.
        """
        if model_type == "Gasnet":
            model = Gasnet2()
        elif model_type == "Gasnet1":
            model = Gasnet2() 
        elif model_type == "Gasnet2":
            model = Gasnet2()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.load_from_checkpoint(path)
        model.eval()
        return model.to(torch.double)

    @staticmethod
    def apply_models_to_data(models, dataloader):
        """
        Apply the given list of models to the data from the dataloader.
        Returns a dataframe with the outputs.
        """
        df_output = pd.DataFrame()
        for idx, (images, _) in enumerate(dataloader):
            for i, model in enumerate(models):
                with torch.no_grad():
                    output = model(images)
                    probability = torch.sigmoid(output).item()
                    df_output.at[idx, f'model_{i+1}'] = probability
        print('THIS')
        print(df_output.head())
        return df_output

    def load_models(self):
        print("Loading models...")
        return [EnsembleAugmentation.load_model_from_checkpoint(path, model_type) for path, model_type in self.model_info]
    
    @staticmethod
    def augment_data_with_model_output(df, model_outputs):
        """
        Augment the given dataframe with the model outputs.
        """
        for column in model_outputs.columns:
            df[column] = model_outputs[column]
        df['vote'] = model_outputs.mean(axis=1)
        return df

    @staticmethod
    def calculate_accuracy(df):
        """
        Calculate the accuracy of the ensemble voting model.
        """
        predicted = (df['vote'] > 0.5).astype(int)
        accuracy = (predicted == df['plume']).mean() * 100
        roc_auc = roc_auc_score(df['plume'], df['vote'])
        print(f"The model predicts {predicted.mean() * 100:.2f}% of the data as leaks. Actual leak percentage is {(df['plume']==1).sum()/len(df['plume']) * 100:.2f}%.")
        print(f"The ROC AUC score is {roc_auc:.2f}")
        return f"The accuracy of the voting model is {accuracy:.2f}%"
    
    def save_and_print_results(self, df_augmented):
        output_file = self.metadata_csv_path.replace('.csv', '_augmented.csv')
        df_augmented.to_csv(output_file, index=False)
        print(EnsembleAugmentation.calculate_accuracy(df_augmented))
    
    def data_loader_and_augmentation(self):
        print("Starting data augmentation...")
        try:
            df = self.load_metadata()
            dataloader = self.create_dataloader(df)
            models = self.load_models()
            model_outputs = EnsembleAugmentation.apply_models_to_data(models, dataloader)
            df_augmented = EnsembleAugmentation.augment_data_with_model_output(df, model_outputs)
            self.save_and_print_results(df_augmented)
        except Exception as e:
            print(f"Error: {e}")
        return df_augmented

if __name__ == "__main__":
    absolute_path = '/home/octav/Documents/HEC/quantum_black/QB_methane/'
    model_info = [
        (absolute_path + 'lightning_logs/version_26/checkpoints/best-model-epoch=05-val_loss=0.39.ckpt', 'Gasnet2'), 
        (absolute_path + 'lightning_logs/version_26/checkpoints/best-model-epoch=05-val_loss=0.39.ckpt', 'Gasnet1'), 
        (absolute_path + 'lightning_logs/version_26/checkpoints/best-model-epoch=05-val_loss=0.39.ckpt', 'Gasnet')
    ]
    metadata_csv_path = absolute_path + 'data/train_data/metadata.csv'
    
    ensemble = EnsembleAugmentation(absolute_path, metadata_csv_path, model_info)
    ensemble.data_loader_and_augmentation()
