from data_utils import load_train
from num_dataset import NumericAugmentation

def test():
    NumericAugmentation.apply_augmentation()
    
    df = load_train('data/train_data/metadata_augmented.csv')
    print(df.head())  # Display the first few rows to check

if __name__ == "__main__":
    test()


# def train_model():
#     NumericAugmentation.apply_augmentation(models=[('lightning_logs/version_21/checkpoints/model.pt', 'Gasnet2'), 
#                     ('lightning_logs/version_20/checkpoints/model.pt', 'Gasnet1'), 
#                     ('lightning_logs/version_19/checkpoints/model.pt', 'Gasnet')])
    
#     X_train, y_train = load_train('data/train_data/metadata_augmented.csv')

#     # 2. Load them as pandas dataframes and print them to check successful loading
    
#     # 3. Preprocess and split dataset
    
#     # 4. Train your model on the dataset
    
#     # 5. Save or evaluate your model

# if __name__ == "__main__":
#     train_model()
