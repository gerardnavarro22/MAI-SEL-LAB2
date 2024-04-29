from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

menu_txt = \
    """1. SMALL DATASET - IRIS"""

print(menu_txt)

option = int(input("Select option number: "))

if option == 1:
    # IRIS DATASET
    # Import the dataset
    dataset, meta = arff.loadarff('../Data/arff/iris.arff')
    df = pd.DataFrame(dataset)

    # Storing the clean dataset in a .csv file
    df.to_csv('../Data/csv/iris.csv', index=False)

    # Creating the train and test datasets and storing them in .csv files
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv('../Data/csv/iris_train.csv', index=False)
    test.to_csv('../Data/csv/iris_test.csv', index=False)

else:
    print("Number not valid")
