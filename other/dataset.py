import numpy as np
import csv
from sklearn.datasets.base import Bunch
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_my_fancy_dataset():
    with open('../dataset/Blooms_Taxonomy_keywords.csv') as csv_file:
        data_file = csv.reader(csv_file)
        # df = pd.get_dummies(data_file)
        df = OneHotEncoder().fit_transform(data_file)
        temp = next(df)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, sample in enumerate(df):
            data[i] = np.asarray(sample[:-1], dtype=np.float64)
            target[i] = np.asarray(sample[-1], dtype=np.int)

    return Bunch(data=data, target=target)