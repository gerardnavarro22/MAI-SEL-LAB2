from RandomForest import process_dataset

if __name__ == "__main__":

    menu_txt = \
        """1. SMALL DATASET - IRIS"""

    print(menu_txt)

    option = int(input("Select option number: "))

    if option == 1:
        process_dataset('iris', 'iris_train.csv', 'iris_test.csv')
    else:
        print("Number not valid")
