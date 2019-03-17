import json
import numpy as np

from collections import defaultdict, Counter

from torchtext.vocab import Vocab
from torchtext.data import Field, TabularDataset


class AbstractVocab(object):
    """
    Abstract interface for the Vocab classes which allows to map between text
    and numbers.
    """

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
        self.padding_token_index = 1

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
        vocab : torchtext.Vocab
            Vocabulary with our corpus.
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
        vocab : torchtext.Vocab
            Vocabulary to use.

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

        # Infer num_labels and group sentences by label
        num_labels = labels_tensor.unique().shape[0]
        num_examples = labels_tensor.shape[0] // num_labels
        y = labels_tensor[::num_examples]
        sen_length = sentences_tensor.shape[-1]
        X = sentences_tensor.view(num_labels, num_examples, sen_length)

        return X, y

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
        vocab : torchtext.Vocab
            Vocabulary to use.

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
