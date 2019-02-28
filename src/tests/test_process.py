from ..process import read_wikitext_corpus


def test_read_wikitext_corpus(wiki_file_path):
    lines = list(read_wikitext_corpus(wiki_file_path))
    assert len(lines) == 39
