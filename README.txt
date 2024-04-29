Supervised and Experiential Learning (SEL) from Master in Artificial Intelligence at UPC
Practical Work 1: CN2 Rule-based classifier
Student: Gerard Navarro

The structure and contents of the projest are:
- Documentation/: Directory with the report of the project
- Data/: Directory with the datasets and outputs
	- arff/: Raw datasets in arff format
	- csv/: Preprocessed datasets in csv format (whole/train/test)
	- output/: Processing outputs (rules+performance+general output with times, etc.)
- Source/: Directory with source code
	- process_datasets.py: This script executes a menu to decide which dataset to preprocess
	- CN2.py: This file includes the CN2 class with all the functionalities and a function to process a dataset (train+test+save)
	- main.py: This script is the main to process a dataset. It shows a menu to choose a dataset to process.
	- requirements.txt: List of Python packages required to run the code.

In order to safely execute the project it is recommended to create a virtual Python environment with the packages from the requirements.txt.
To install the packages just run 'pip install -r requirements.txt' within a clean virtual environment.
