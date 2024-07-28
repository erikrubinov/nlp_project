import pandas as pd
from csv import QUOTE_NONE

""" Function to load data into a pandas DataFrame without treating any character as quotes"""

def load_data_to_dataframe(file_path):
    # Read the file line-by-line into a list
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Strip any leading/trailing whitespace characters from each line
    lines = [line.strip() for line in lines]

    # Create a DataFrame with a single column 'text'
    df = pd.DataFrame(lines, columns=['text'])
    return df

def load_datasets(fraction):
    # Define the paths to your files
    english_file_path = "fr-en/europarl-v7.fr-en.en"
    french_file_path = "fr-en/europarl-v7.fr-en.fr"
    # Load the data
    english_data = load_data_to_dataframe(english_file_path)
    french_data = load_data_to_dataframe(french_file_path)

    ## Take a fraction of data randomly sampled but keep order
    english_data = english_data.sample(frac=fraction, random_state=42).sort_index()
    # for all the indices in the english data that do not exist in the french data, drop them
    english_data = english_data[english_data.index.isin(french_data.index)]
    french_data = french_data[french_data.index.isin(english_data.index)]

    return english_data, french_data