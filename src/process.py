import re
import spacy

from spacy.tokens import Token

from .utils import getattrd

# A title starts and ends with one or more '='
# e.g. '= = Gameplay = ='
TITLE_REGEX = re.compile(r'^ (= )+.+( =)+ \n$')


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
        Token.set_extension("as_ptb", default=None)

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

    def __init__(self, attr_name="as_ptb"):
        """
        Initialises internal data structures to count frequency
        of tokens per sentence.

        Parameters
        ---
        attr_name : string
            Attribute name to where to get the actual text from.
        """
        # Sentences will be a dictionary with pointers
        # to sentences is present.
        self._sentences = {}

        # Sentences count is computed over self.sentences, with
        # the counts of sentences of each entry
        self._sentences_count = {}

        self.attr_name = attr_name

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
                # Extract actual text
                text = getattrd(token, self.attr_name)

                # If token has already been processed,
                # ignore
                if token in seen:
                    continue

                # If text has never been seen before,
                # initialise array
                if text not in self._sentences:
                    self._sentences[text] = []
                    self._sentences_count[text] = 0

                # Store pointer to sentence and increase count
                self._sentences[text].append(sentence)
                self._sentences_count[text] += 1


def process_wikitext_corpus(file_path):
    """
    Processes the wikitext corpus in chunks, using spacy.

    Parameters
    ----
    file_path : string
        Path to the corpus file.

    Returns
    ---
    list
        List of processed sentences.
    """
    nlp = spacy.load(
        'en', disable=['lemmatizer', 'matcher', 'parser', 'tagger', 'ner'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'), last=True)

    line_iterator = read_wikitext_corpus(file_path)
    nlp_pipe = nlp.pipe(line_iterator, batch_size=50, n_threads=2)
    for line in nlp_pipe:
        import ipdb
        ipdb.set_trace()
