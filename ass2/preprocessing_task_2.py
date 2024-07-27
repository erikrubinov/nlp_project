import re
from helpers import load_datasets


def preprocess_data(data_en, data_fr):
    # Lowercase the text
    """
    Normalizing case helps reduce the complexity of the language model by treating words like “The” and “the” as the same word, which can be particularly helpful in languages like English where capitalization is more stylistic than semantic.
    """
    data_en['text'] = data_en['text'].str.lower()
    data_fr['text'] = data_fr['text'].str.lower()

    # Remove XML tags
    """
    Lines containing XML-tags are likely not actual conversational or formal text but rather formatting or metadata which is irrelevant for translation purposes
    """
    data_en['text'] = data_en['text'].apply(lambda x: '' if x.strip().startswith('<') else x)
    data_fr['text'] = data_fr['text'].apply(lambda x: '' if x.strip().startswith('<') else x)

    # Strip empty lines and remove their correspondences
    """
     Empty lines or lines that do not contain any meaningful content should be removed because they do not provide valuable information for training the model. It is also important to remove the corresponding line in the other language to maintain alignment.
    """
    mask = (data_en['text'].str.strip().astype(bool) & data_fr['text'].str.strip().astype(bool))
    data_en = data_en[mask]
    data_fr = data_fr[mask]

    # Define the characters to remove
    noisy_characters = re.escape('@#$%^&*~<>|\\{}[]+=_/')

    # Regex to match any noisy character
    regex_pattern = f'[{noisy_characters}]'

    # Remove noisy characters using regex substitution
    data_en['text'] = data_en['text'].apply(lambda x: re.sub(regex_pattern, '', x))
    data_fr['text'] = data_fr['text'].apply(lambda x: re.sub(regex_pattern, '', x))


    return data_en, data_fr



def prepare_data():
    english_data, french_data = load_datasets()
    preprocessed_en, preprocessed_fr = preprocess_data(english_data, french_data)

    return preprocessed_en, preprocessed_fr
