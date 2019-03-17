import json
import torch
import numpy as np
import pandas as pd

from collections import defaultdict, Counter

from torchtext.vocab import Vocab
from torchtext.data import Field, TabularDataset

from pytorch_pretrained_bert import BertTokenizer


class AbstractVocab(object):
    """
    Abstract interface for the Vocab classes which allows to map between text
    and numbers.
    """

    padding_token_index = 0

    def __len__(self):
        raise NotImplementedError()

    def to_tensors(self, file_path):
        raise NotImplementedError()

    def to_text(self, X):
        raise NotImplementedError()


class VanillaVocab(AbstractVocab):
    """
    Allows to map between text and numbers using a simple tokenizer.
    """

    padding_token_index = 1

    def __init__(self, file_path):
        """
        Initialise the vocabulary by reading it from a file path.

        Parameters
        ---
        file_path : str
            Path to the vocab file.
        """
        super().__init__()

        self.vocab = self._read_vocab(file_path)

    def _read_vocab(self, file_path):
        """
        Reads a vocab from its previously stored state.

        Inspired by https://github.com/pytorch/text/issues/253#issue-305929871

        Parameters
        ---
        file_path : str
            Path to the JSON file with the vocab info.

        Returns
        ---
        vocab : torchtext.Vocab
            Vocabulary created.
        """
        vocab_state = {}
        with open(file_path) as file:
            vocab_state = json.load(file)

        vocab = Vocab(Counter())
        vocab.__dict__.update(vocab_state)
        vocab.stoi = defaultdict(lambda: 0, vocab.stoi)
        return vocab

    def __len__(self):
        """
        Returns the size of the vocabulary.

        Returns
        ---
        int
            Number of tokens in the vocabulary.
        """
        return len(self.vocab)

    def save(self, file_path):
        """
        Stores a vocab in a JSON file.

        Inspired by https://github.com/pytorch/text/issues/253#issue-305929871

        Parameters
        ---
        file_path : str
            Path to the vocab state to write.

        """
        vocab_state = dict(self.vocab.__dict__, stoi=dict(self.vocab.stoi))
        with open(file_path, 'w') as file:
            json.dump(vocab_state, file)

    def to_tensors(self, file_path):
        """
        Reads the data set from one of the pre-processed CSVs composed
        of columns `label` and `sentence`.

        Parameters
        ---
        file_path : str
            Path to the CSV file.

        Returns
        ---
        X : torch.Tensor[num_labels x num_examples x sen_length]
            Sentences on the dataset grouped by labels.
        y : torch.Tensor[num_labels]
            Labels for each group of sentences.
        """
        sentence = Field(
            batch_first=True, sequential=True, tokenize=self._tokenizer)
        sentence.vocab = self.vocab

        label = Field(is_target=True)
        label.vocab = self.vocab

        data_set = TabularDataset(
            path=file_path,
            format='csv',
            skip_header=True,
            fields=[('label', label), ('sentence', sentence)])

        sentences_tensor = sentence.process(data_set.sentence)
        labels_tensor = label.process(data_set.label).squeeze()

        return _reshape_tensors(sentences_tensor, labels_tensor)

    @classmethod
    def _tokenizer(cls, text):
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

    def to_text(self, X):
        """
        Reverses some numericalised tensor into text.

        Parameters
        ----
        X : torch.Tensor[num_elements x sen_length]
            Sentences on the tensor.

        Returns
        ----
        sentences : np.array[num_elements]
            Array of strings.
        """
        sentences = []
        for sentence_tensor in X:
            if len(sentence_tensor.shape) == 0:
                # 0-D tensor
                sentences.append(self.vocab.itos[sentence_tensor])
                continue

            sentence = [
                self.vocab.itos[token] for token in sentence_tensor
                if token != self.padding_token_index
            ]
            sentences.append(' '.join(sentence))

        return np.array(sentences)

    @classmethod
    def generate_vocab(cls, file_path):
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
        text = Field(sequential=True, tokenize=cls._tokenizer)

        data_set = TabularDataset(
            path=file_path,
            format='csv',
            skip_header=True,
            fields=[('label', text), ('sentence', text)])

        text.build_vocab(data_set.label, data_set.sentence)
        return text.vocab


class BertVocab(AbstractVocab):
    """
    Implementation of mappings between text and tensors using Bert.
    """

    padding_token_index = 0

    def __init__(self, *args, **kwargs):
        """
        Initialise Bert's tokenizer.
        """
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        """
        Returns the length of Bert's vocabulary.

        Returns
        ---
        int
            Length of the vocabulary.
        """
        return len(self.tokenizer.vocab)

    def to_tensors(self, file_path):
        """
        Reads the data set from one of the pre-processed CSVs composed
        of columns `label` and `sentence`.

        Parameters
        ---
        file_path : str
            Path to the CSV file.

        Returns
        ---
        X : torch.Tensor[num_labels x num_examples x sen_length]
            Sentences on the dataset grouped by labels.
        y : torch.Tensor[num_labels]
            Labels for each group of sentences.
        """
        data_set = pd.read_csv(file_path)

        # Convert into tokens and find max sen length
        sentences_tokens, labels_tokens, sen_length = self._to_tokens(data_set)

        # Convert into tensors
        num_elems = len(sentences_tokens)
        sentences_tensor = torch.zeros((num_elems, sen_length))
        labels_tensor = torch.zeros(num_elems)

        for idx in range(num_elems):
            tensor_sentence = self.tokenizer.convert_tokens_to_ids(
                sentences_tokens[idx])
            tensor_label = self.tokenizer.convert_tokens_to_ids(
                labels_tokens[idx])

            sentences_tensor[idx, :len(tensor_sentence)] = torch.Tensor(
                tensor_sentence)
            labels_tensor[idx] = tensor_label[0]

        return _reshape_tensors(sentences_tensor, labels_tensor)

    def _to_tokens(self, data_set):
        """
        Tokenize the dataset.

        Parameters
        ---
        data_set : pd.DataFrame[label, sentence]
            Dataset with two columns.

        Returns
        ---
        sentences_tokens : list
            List of tokenized sentences.
        labels_tokens : list
            List of tokenized labels.
        sen_length : int
            Maximum sentence length.
        """
        sentences_tokens = []
        labels_tokens = []
        sen_length = 0
        for idx, row in data_set.iterrows():
            token_sentence = self._tokenize(row['sentence'])
            token_label = self._tokenize(row['label'])

            if len(token_label) > 1:
                continue
                #  raise ValueError(f"Label '{row['label']}' was split "
                #  f"into more than one tokens: "
                #  f"{token_label}")

            length = len(token_sentence)
            if length > sen_length:
                sen_length = length

            sentences_tokens.append(token_sentence)
            labels_tokens.append(token_label)

        return sentences_tokens, labels_tokens, sen_length

    def _tokenize(self, text):
        """
        Tokenize a text using Bert's tokenizer but processing it first to
        replace:

            - <unk> => [UNK]
            - <blank_token> => [MASK]

        Parameters
        ---
        text : str
            Input string.

        Returns
        ---
        list
            List of tokens.
        """
        with_unk = text.replace('<unk>', '[UNK]')
        with_mask = with_unk.replace('<blank_token>', '[MASK]')

        return self.tokenizer.tokenize(with_mask)

    def to_text(self, X):
        """
        Reverses some numericalised tensor into text.

        Parameters
        ----
        X : torch.Tensor[num_elements x sen_length]
            Sentences on the tensor.

        Returns
        ----
        sentences : np.array[num_elements]
            Array of strings.
        """
        sentences = []
        for sentence_tensor in X:
            if len(sentence_tensor.shape) == 0:
                # 0-D tensor
                sentences.append(self.tokenizer.ids_to_tokens[sentence_tensor])
                continue

            sentence = [
                self.tokenizer.ids_to_tokens[token_id]
                for token_id in sentence_tensor
                if token_id != self.padding_token_index
            ]
            sentences.append(' '.join(sentence))

        return np.array(sentences)


def _reshape_tensors(sentences_tensor, labels_tensor):
    """
    Reshape tensors to the [N x k x sen_lenth] structure.

    Parameters
    ---
    sentences_tensor : torch.Tensor[num_elems x sen_length]
        Flat tensor with all the sentences.
    labels_tensor : torch.Tensor[num_elems]
        Flat tensor with all the labels.

    Returns
    ---
    X : torch.Tensor[num_labels x num_examples x sen_length]
        Sentences on the dataset grouped by labels.
    y : torch.Tensor[num_labels]
        Labels for each group of sentences.
    """
    # Infer num_labels and num_examples by label
    num_labels = labels_tensor.unique().shape[0]
    num_examples = labels_tensor.shape[0] // num_labels
    y = labels_tensor[::num_examples]

    # More robust to potentially duplicated labels
    num_labels = y.shape[0]

    sen_length = sentences_tensor.shape[-1]
    X = sentences_tensor.view(num_labels, num_examples, sen_length)

    return X, y


VOCABS = {'vanilla': VanillaVocab, 'bert': BertVocab}


def get_vocab(embeddings, *args, **kwargs):
    """
    Returns an initialised vocab, forwarding the extra args and kwargs.

    Parameters
    ---
    embeddings : str
        Embeddings to use. Can be one of the VOCABS keys.

    Returns
    ---
    AbstractVocab
    """
    return VOCABS[embeddings](*args, **kwargs)
