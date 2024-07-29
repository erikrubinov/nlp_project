import re
from helpers import load_datasets


def preprocess_data(source_data, target_data):
    # Lowercase the text
    """
    Normalizing case helps reduce the complexity of the language model by treating words like “The” and “the” as the same word, which can be particularly helpful in languages like English where capitalization is more stylistic than semantic.
    """
    source_data['text'] = source_data['text'].str.lower()
    target_data['text'] = target_data['text'].str.lower()

    # Remove XML tags
    """
    Lines containing XML-tags are likely not actual conversational or formal text but rather formatting or metadata which is irrelevant for translation purposes
    """
    source_data['text'] = source_data['text'].apply(lambda x: '' if x.strip().startswith('<') else x)
    target_data['text'] = target_data['text'].apply(lambda x: '' if x.strip().startswith('<') else x)

    # Strip empty lines and remove their correspondences
    """
     Empty lines or lines that do not contain any meaningful content should be removed because they do not provide valuable information for training the model. It is also important to remove the corresponding line in the other language to maintain alignment.
    """
    mask = (source_data['text'].str.strip().astype(bool) & target_data['text'].str.strip().astype(bool))
    source_data = source_data[mask]
    target_data = target_data[mask]

    # Define the characters to remove
    noisy_characters = re.escape('@#$%^&*~<>|\\{}[]+=_/')

    # Regex to match any noisy character
    regex_pattern = f'[{noisy_characters}]'

    # Remove noisy characters using regex substitution
    source_data['text'] = source_data['text'].apply(lambda x: re.sub(regex_pattern, '', x))
    target_data['text'] = target_data['text'].apply(lambda x: re.sub(regex_pattern, '', x))

    return source_data, target_data


def prepare_data(fraction, source_file_path="fr-en/europarl-v7.fr-en.en",
                 target_file_path="fr-en/europarl-v7.fr-en.fr"):
    source_data, target_data = load_datasets(fraction, source_file_path, target_file_path)
    preprocessed_source, preprocessed_target = preprocess_data(source_data, target_data)

    return preprocessed_source, preprocessed_target
