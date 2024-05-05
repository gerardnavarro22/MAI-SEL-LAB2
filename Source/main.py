import time
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
from RandomForest import RandomForest
from DecisionForest import DecisionForest


def process_dataset(name: str, train_file: str, test_file: str, algorithm: str, f: int = 0, nt: int = 100,
                    n_jobs: int = -1):
    """
    This function processes a dataset for training and testing purposes.
    :param n_jobs: Cores to train the classifier. -1 means all available
    :param nt: Number of desired trees
    :param f: Number of random features used in the splitting of the nodes in RF or in each tree in DF
    :param name: The name or identifier of the dataset.
    :param train_file: The filename of the training dataset in CSV format. It must be located in the '../Data/csv/'
    directory.
    :param test_file: The filename of the testing dataset in CSV format. It must be located in the '../Data/csv/'
    directory.
    :param algorithm: The ensemble algorithm to use: random-forest or decision-forest
    """
    outfile = open(fr'../Data/output/{name}_{algorithm}_nt={nt}_f={f}output.txt', 'w')

    print('############', file=outfile)
    print(f'DATASET - {name.upper()}', file=outfile)
    print('############', file=outfile)

    data_train = pd.read_csv(fr'../Data/csv/{train_file}')
    data_test = pd.read_csv(fr'../Data/csv/{test_file}')

    feature_names = data_train.columns.to_list()

    X_train = data_train.drop('class', axis=1).to_numpy()
    y_train = data_train['class'].to_numpy()
    X_test = data_test.drop('class', axis=1).to_numpy()
    y_test = data_test['class'].to_numpy()

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.fit_transform(y_test)

    train_start = time.time()

    if algorithm == 'random-forest':
        classifier = RandomForest(nt, f, n_jobs=n_jobs)
    elif algorithm == 'decision-forest':
        classifier = DecisionForest(nt, f, n_jobs=n_jobs)
    else:
        raise NotImplementedError("Type random-forest or decision-forest")

    classifier.fit(X_train, y_train_encoded)
    train_end = time.time()

    print(f'Training time: {(train_end - train_start):.2f}s', file=outfile)

    y_predict = classifier.predict(X_test)

    print(f'Accuracy: {100 * accuracy_score(y_test_encoded, y_predict):.2f}%\n', file=outfile)

    print(f'Feature importances:', file=outfile)
    print("{:<15} {:<10}".format('Feature', 'Frequency'), file=outfile)
    for feature_idx, freq in classifier.feature_importances.items():
        print("{:<15} {:<10}".format(feature_names[feature_idx], freq), file=outfile)

    outfile.close()

    outfile = open(fr'../Data/output/{name}_{algorithm}_nt={nt}_f={f}output.txt', 'r')
    print(outfile.read())
    outfile.close()


if __name__ == "__main__":

    menu_txt = \
        """1. RANDOM FOREST - SMALL DATASET - IRIS
2. DECISION FOREST - SMALL DATASET - IRIS
3. RANDOM FOREST - MEDIUM DATASET - WINE
4. DECISION FOREST - MEDIUM DATASET - WINE
5. RANDOM FOREST - BIG DATASET - SEGMENT
6. DECISION FOREST - BIG DATASET - SEGMENT"""

    print(menu_txt)

    option = int(input("Select option number: "))

    if option == 1:
        process_dataset('iris', 'iris_train.csv', 'iris_test.csv', 'random-forest', n_jobs=1)
    elif option == 2:
        process_dataset('iris', 'iris_train.csv', 'iris_test.csv', 'decision-forest', n_jobs=1)
    elif option == 3:
        process_dataset('wine', 'wine_train.csv', 'wine_test.csv', 'random-forest')
    elif option == 4:
        process_dataset('wine', 'wine_train.csv', 'wine_test.csv', 'decision-forest')
    elif option == 5:
        process_dataset('segment', 'segment_train.csv', 'segment_test.csv', 'random-forest')
    elif option == 6:
        process_dataset('segment', 'segment_train.csv', 'segment_test.csv', 'decision-forest')
    else:
        print("Number not valid")
