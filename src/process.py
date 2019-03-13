import re
import spacy

from spacy.tokens import Token

from .utils import sample_elements, getattrd

# A title starts and ends with one or more '='
# e.g. '= = Gameplay = ='
TITLE_REGEX = re.compile(r'^ (= )+.+( =)+ \n$')

BLANK_TOKEN = "<blank_token>"

VOCAB_SIZE = 27465

PADDING_TOKEN_INDEX = 1


def read_wikitext_corpus(file_path):
    """
    Reads the WT2 corpus, ignoring titles.

    Parameters
    ---
    file_path : string
        Path to the corpus file.

    Returns
    ---
    iterator
        Iterator over blocks of text, where each chunk corresponds to one of
        the documents.
    """
    with open(file_path, 'r') as file:
        line = None
        while line != "":
            line = file.readline()

            if TITLE_REGEX.match(line):
                # Ignore title
                continue

            if line == " \n":
                # If it's just a new line,
                # ignore
                continue

            if line != "":
                # Don't return empty line
                # from last iteration
                yield line


class PTBTransformer(object):
    """
    Pipeline component which processes the input
    to make it look like the PTB dataset, by:

        - Lowercase all input.
        - Replace numbers for N.
        - Remove punctuation.

    The processed output gets stored on a custom `as_ptb` extension.
    """

    name = "ptb_transformer"

    def __init__(self):
        """
        Initialise some custom attributes.
        """
        Token.set_extension("as_ptb", default=None, force=True)

    def __call__(self, doc):
        """
        Transforms a sentence to look like the PTB dataset.

        Parameters
        ----
        doc : spacy.Doc
            Sentence object with some text.

        Returns
        ----
        spacy.Doc
            Document with processed sentences.
        """
        for (index, token) in enumerate(doc):
            if token.is_punct:
                continue

            # Lowercase
            token._.as_ptb = token.lower_

            if token.is_digit:
                # Transform to N
                token._.as_ptb = u'N'

        return doc


class EpisodesSampler(object):
    """
    Pipeline component which counts the distribution
    of tokens over sentences, and then allows to sample words
    and examples.
    """

    name = "episodes_sampler"

    def __init__(self, attr_name, is_label):
        """
        Initialises internal data structures to count frequency
        of tokens per sentence.

        Parameters
        ---
        attr_name : string
            Attribute name to where to get the actual text from.
        is_label : func
            Function to determine if a token works as label.
        """
        # Sentences will be a dictionary with pointers
        # to sentences is present.
        self._sentences = {}

        # Sentences count is computed over self.sentences, with
        # the counts of sentences of each entry
        self._sentences_count = {}

        self.attr_name = attr_name
        self.is_label = is_label

    def __call__(self, doc):
        """
        Processes doc's sentences and increases count
        of each word.

        Parameters
        ----
        doc : spacy.Doc
            Document to process.

        Parameters
        ----
        doc : spacy.Doc
            Unmodified document.
        """
        for sentence in doc.sents:
            seen = set()
            for token in sentence:
                if not self.is_label(token):
                    continue

                # Extract actual text
                text = self._get_text(token)

                # If text has already been seen
                # during this sentence
                if text in seen:
                    continue

                # If text has never been seen before,
                # initialise array
                if text not in self._sentences:
                    self._sentences[text] = []
                    self._sentences_count[text] = 0

                # Store pointer to sentence and increase count
                self._sentences[text].append(sentence)
                self._sentences_count[text] += 1

    def _get_text(self, token):
        """
        Returns the text content of a token
        following the specified attr_name
        """
        return getattrd(token, self.attr_name)

    def sample(self, N, k):
        """
        Samples N labels with k example sentences each. On the sentences,
        the word will get replaced with the token <blank_token>.

        Parameters
        ---
        N : int
            Number of labels to sample.
        k : int
            Number of examples to sample per label.

        Returns
        ---
        iterator
            Iterator yielding pairs of label and sentences sampled randomly.
        """
        # Find words present on more than k sentences
        labels = [
            label for label, count in self._sentences_count.items()
            if count >= k
        ]

        # For each sampled label...
        for label in sample_elements(labels, size=N):
            # Fetch all sentences for the given label
            sentences = self._sentences[label]

            # For each sampled sentence...
            for sentence in sample_elements(sentences, size=k):
                sentence_text = self._get_sentence_text(label, sentence)
                yield label, sentence_text

    def _get_sentence_text(self, label, sentence):
        """
        Generates a sentence's text, replacing the given
        label for <blank_token>.

        Parameters
        ----
        label : str
            Given label, present in the sentence.
        sentence : spacy.Span
            Span of sentence.

        Returns
        ----
        str
            Assembled sentence.
        """
        tokens = []
        label_indices = []
        for token in sentence:
            token_text = self._get_text(token)
            if token_text is None:
                # Punctuation marks can be None
                continue

            if token_text == label:
                label_indices.append(len(tokens))

            tokens.append(token_text)

        # Replace label with <blank_token>
        if len(label_indices) == 0:
            raise ValueError(f"Sentence doesn't contain label '{label}'")

        label_index = label_indices[0]
        if len(label_indices) > 1:
            # If the label is present on more than one
            # location, decide randomly which to replace
            label_index = sample_elements(label_indices, size=1)[0]

        tokens[label_index] = BLANK_TOKEN

        return ' '.join(tokens)


def is_label(token):
    """
    Logic to decide if a token is or not a valid
    label, that is if it's not a SYMbol, a NUMber or
    a stop word.

    Parameters
    ---
    token : spacy.Token
        Input token.

    Returns
    ---
    bool
    """
    # We only look at tokens which are not
    # SYM, NUM or stop words
    invalid_pos = ['NUM', 'PUNCT', 'SYM', 'X', "SPACE"]

    return token.pos_ not in invalid_pos and not token.is_stop


def process_wikitext_corpus(file_path):
    """
    Processes the wikitext corpus in chunks, using spacy.

    Parameters
    ----
    file_path : string
        Path to the corpus file.

    Returns
    ---
    sampler : EpisodesSampler
        Sampler which can be used to sample sentences.
    """
    nlp = spacy.load('en', disable=['lemmatizer', 'matcher', 'parser', 'ner'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'), last=True)
    nlp.add_pipe(PTBTransformer(), last=True)

    sampler = EpisodesSampler(attr_name="_.as_ptb", is_label=is_label)
    nlp.add_pipe(sampler, last=True)

    line_iterator = read_wikitext_corpus(file_path)
    nlp_pipe = nlp.pipe(line_iterator, batch_size=50, n_threads=2)

    # Process all chunks
    [doc for doc in nlp_pipe]

    return sampler
