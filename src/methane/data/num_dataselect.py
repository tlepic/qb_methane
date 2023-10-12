import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, roc_auc_score

class NumDataSelector():

    def __init__(self, df):
        super().__init__()
        self.df = df
        self.world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    def get_country(self, lat, lon):
        point = Point(lon, lat)
        country = self.world[self.world.geometry.contains(point)]
        if country.empty:
            return None
        else:
            return country.iloc[0]['name']

    def parse_dataset(self, train=True):
        self.df['country'] = self.df.apply(lambda row: self.get_country(row['lat'], row['lon']), axis=1)

        self.df['month'] = pd.to_datetime(self.df['date'], format='%Y%m%d').dt.month
        self.df.drop(columns=['date', 'model_1', 'model_2', 'model_3', 'vote'], inplace=True)

        self.df['coord_product'] = self.df['coord_x'] * self.df['coord_y']

        columns_to_normalize = ['coord_product']
        for col in columns_to_normalize:
            self.df[col] = (self.df[col] - self.df[col].min()) / (self.df[col].max() - self.df[col].min())

        # One-hot encode 'country' and 'continent' (if it exists)
        if 'country' in self.df.columns:
            self.df = pd.get_dummies(self.df, columns=['country'])
        if 'continent' in self.df.columns:
            self.df = pd.get_dummies(self.df, columns=['continent'])

        # Ensure only numerical columns are included in X

        if train:
            y = self.df['plume']
        
        X = self.df.drop(columns=['plume'])
        X = X.select_dtypes(include=[float, int])
        
        if train:
            return X, y
        else:
            return X


if __name__ == '__main__':
    absolute_path = '/home/octav/Documents/HEC/quantum_black/QB_methane'

    df = pd.read_csv(absolute_path + '/data/train_data/metadata_augmented.csv')
    selector = NumDataSelector(df)
    X, y = selector.parse_dataset()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train a Random Forest Classifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)
    # Predict on the test set

    y_pred_proba = clf.predict_proba(X_test)[:, 1]  # Probability of positive class

    # Print the accuracy and f1 score
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, y_pred_proba))

    # Print the and f1
    print("Accuracy:", accuracy_score(y_test, y_pred))
