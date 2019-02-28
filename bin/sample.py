"""
Script which pre-processes the WT2 dataset and samples N labels with K example
sentences each.

Sampling is performed without replacement.
"""
import os

from argparse import ArgumentParser

from src.process import process_wikitext_corpus

parser = ArgumentParser()
parser.add_argument(
    "-N",
    "--number-of-labels",
    action="store",
    dest="N",
    help="Number of labels to sample")
parser.add_argument(
    "-k",
    "--number-of-examples-per-label",
    action="store",
    dest="k",
    help="Number of examples to sample per label")
parser.add_argument("file_path", help="Path to the input WT2 file")


def main(args):
    # This should avoid 100% CPU usage and
    # should make it faster according to
    # https://spacy.io/api/language#pipe
    os.environ['OPENBLAS_NUM_THREADS'] = "1"
    sampler = process_wikitext_corpus(args.file_path)

    for pair in sampler.sample(N=args.N, k=args.k):
        print(pair)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
