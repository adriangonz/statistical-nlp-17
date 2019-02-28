import os
import pytest
import spacy

FIXTURES_PATH = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def wiki_file_path():
    return os.path.join(FIXTURES_PATH, "wiki.fixture.tokens")


@pytest.fixture
def wiki_doc(wiki_file_path):
    nlp = spacy.load(
        'en', disable=['lemmatizer', 'matcher', 'parser', 'tagger', 'ner'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'), last=True)
    with open(wiki_file_path, 'r') as file:
        return nlp(file.read())
