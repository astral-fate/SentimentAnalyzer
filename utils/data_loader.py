from datasets import load_dataset

def load_imdb_dataset():
    """
    Load the IMDB dataset from HuggingFace datasets
    """
    try:
        dataset = load_dataset("stanfordnlp/imdb")
        return dataset
    except Exception as e:
        raise Exception(f"Error loading IMDB dataset: {str(e)}")
