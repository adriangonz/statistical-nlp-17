from ..process import (read_wikitext_corpus, PTBTransformer, EpisodesSampler,
                       process_wikitext_corpus)


def test_read_wikitext_corpus(wiki_file_path):
    lines = list(read_wikitext_corpus(wiki_file_path))
    assert len(lines) == 39


def test_ptb_transformer(wiki_doc):
    transformer = PTBTransformer()
    transformed_doc = transformer(wiki_doc)

    for token in transformed_doc:
        if token.is_punct:
            assert token._.as_ptb is None
        elif token.is_digit:
            assert token._.as_ptb == u'N'
        else:
            assert token._.as_ptb == token.text.lower()


def test_episodes_sampler(wiki_doc):
    sampler = EpisodesSampler(attr_name="text")
    sampler(wiki_doc)

    assert len(sampler._sentences) == 1179
    assert len(sampler._sentences_count) == 1179


def test_process_wikitext_corpus(wiki_file_path):
    process_wikitext_corpus(wiki_file_path)
