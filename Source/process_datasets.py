from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split

menu_txt = \
    """1. SMALL DATASET - IRIS
2. MEDIUM DATASET - WINE
3. BIG DATASET - SEGMENT"""

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

elif option == 2:
    # WINE DATASET
    # Import the dataset
    dataset, meta = arff.loadarff('../Data/arff/wine.arff')
    df = pd.DataFrame(dataset)
    df = df.rename(columns={"Class": "class"})

    # Storing the clean dataset in a .csv file
    df.to_csv('../Data/csv/wine.csv', index=False)

    # Creating the train and test datasets and storing them in .csv files
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv('../Data/csv/wine_train.csv', index=False)
    test.to_csv('../Data/csv/wine_test.csv', index=False)

elif option == 3:
    # SEGMENT DATASET
    # Import the dataset
    dataset, meta = arff.loadarff('../Data/arff/segment.arff')
    df = pd.DataFrame(dataset)

    # Storing the clean dataset in a .csv file
    df.to_csv('../Data/csv/segment.csv', index=False)

    # Creating the train and test datasets and storing them in .csv files
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv('../Data/csv/segment_train.csv', index=False)
    test.to_csv('../Data/csv/segment_test.csv', index=False)

else:
    print("Number not valid")
