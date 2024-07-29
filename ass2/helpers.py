import pandas as pd

def load_data_to_dataframe(file_path) -> pd.DataFrame:
    # Read the file line-by-line into a list
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Strip any leading/trailing whitespace characters from each line
    lines = [line.strip() for line in lines]

    # Create a DataFrame with a single column 'text'
    df = pd.DataFrame(lines, columns=['text'])
    return df


def load_datasets(fraction, source_file_path="fr-en/europarl-v7.fr-en.en",
                  target_file_path="fr-en/europarl-v7.fr-en.fr"):
    # Load the data
    source_data = load_data_to_dataframe(source_file_path)
    target_data = load_data_to_dataframe(target_file_path)

    # Take a fraction of data randomly sampled but keep order
    source_data = source_data.sample(frac=fraction, random_state=42).sort_index()
    # for all the indices in the source data that do not exist in the target data, drop them
    source_data = source_data[source_data.index.isin(target_data.index)]
    target_data = target_data[target_data.index.isin(source_data.index)]

    return source_data, target_data
