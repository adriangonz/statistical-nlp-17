"""
Methods and tools to work with the pre-processed CSVs.
"""
from collections import defaultdict

from torchtext.data import Field, TabularDataset
from torchtext.vocab import Vocab


def simple_tokenizer(text):
    """
    Simple tokenizer which splits the token by the space
    character. The CSVs have already been pre-processed with
    spaCy, therefore this should be enough.

    Parameters
    ---
    text : str
        Input text to tokenize.

    Returns
    ---
    iterator
        Iterator over token text.
    """
    return text.split(' ')


def get_vocab_state(vocab):
    """
    Returns a serializable state of a vocab.

    Inspired by https://github.com/pytorch/text/issues/253#issue-305929871

    Parameters
    ---
    vocab : torchtext.Vocab
        Vocabulary with our corpus.

    Returns
    ---
    state : dict
        Serializable state with the string-to-int mappings.
    """
    return dict(vocab.__dict__, stoi=dict(vocab.stoi))


def create_vocab_from_state(vocab_state):
    """
    Generates a vocab from its previously stored state.

    Inspired by https://github.com/pytorch/text/issues/253#issue-305929871

    Parameters
    ---
    vocab_state : dict
        State with the string-to-int mappings.

    Returns
    ---
    vocab : torchtext.Vocab
        Vocabulary created.
    """
    vocab = Vocab()
    vocab.__dict__.update(vocab_state)
    vocab.stoi = defaultdict(lambda: 0, vocab.stoi)
    return vocab


def generate_vocabulary(file_path):
    """
    Generate the vocabulary from one of the pre-processed CSVs composed
    of columns `label` and `sentence`.

    Parameters
    ---
    file_path : str
        Path to the CSV file.

    Returns
    ---
    vocab : torchtext.Vocab
        Vocabulary generated from the file.
    """
    text = Field(sequential=True, tokenize=simple_tokenizer)

    data_set = TabularDataset(
        path=file_path,
        format='csv',
        fields=[('label', text), ('sentence', text)])

    text.build_vocab(data_set.label, data_set.sentence)
    return text.vocab
