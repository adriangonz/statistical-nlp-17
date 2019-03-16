import collections

from ..process import (read_wikitext_corpus, PTBTransformer, EpisodesSampler,
                       process_wikitext_corpus, BLANK_TOKEN)


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
    sampler = EpisodesSampler(attr_name="text", is_label=lambda _: True)
    sampler(wiki_doc)

    assert len(sampler._sentences) == 1179
    assert len(sampler._sentences_count) == 1179


def test_episodes_sample(wiki_doc):
    sampler = EpisodesSampler(attr_name="text", is_label=lambda _: True)
    sampler(wiki_doc)

    N = 9
    k = 3

    # re-assemble dict
    pairs = collections.defaultdict(list)
    for (label, sentence) in sampler.sample(N, k):
        pairs[label].append(sentence)

    assert len(pairs) == N
    for label, sentences in pairs.items():
        assert len(sentences) == k
        assert label not in sentences
        for sentence in sentences:
            assert BLANK_TOKEN in sentence


def test_process_wikitext_corpus(wiki_file_path):
    sampler = process_wikitext_corpus(wiki_file_path)

    assert len(sampler._sentences) == 852
