import re

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
        Iterator over blocks of text, where each
        chunk corresponds to one of the documents.
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
