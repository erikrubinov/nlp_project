import pandas as pd

""" Function to load data into a pandas DataFrame without treating any character as quotes"""

def load_data_to_dataframe(file_path):
    # Read the entire file as a single column DataFrame, ignoring any quoting
    return pd.read_csv(file_path, header=None, names=['text'], encoding='utf-8', sep='\t', quoting=QUOTE_NONE, engine='python')

def load_datasets():
    # Define the paths to your files
    english_file_path = "fr-en/europarl-v7.fr-en.en"
    french_file_path = "fr-en/europarl-v7.fr-en.fr"
    # Load the data
    english_data = load_data_to_dataframe(english_file_path)
    french_data = load_data_to_dataframe(french_file_path)

    ## Take 10% fraction of data randomly sampled
    english_data = english_data.sample(frac=0.1, random_state=42)
    french_data = french_data.sample(frac=0.1, random_state=42)

    return english_data, french_data