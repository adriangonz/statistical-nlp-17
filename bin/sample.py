"""
Script which pre-processes the WT2 dataset and samples N labels with K example
sentences each.

Sampling is performed without replacement.
"""
import os
import csv

from argparse import ArgumentParser

from src.process import process_wikitext_corpus

parser = ArgumentParser()
parser.add_argument(
    "-N",
    "--number-of-labels",
    action="store",
    dest="N",
    type=int,
    help="Number of labels to sample")
parser.add_argument(
    "-k",
    "--number-of-examples-per-label",
    action="store",
    dest="k",
    type=int,
    help="Number of examples to sample per label")
parser.add_argument("input", help="Path to the input WT2 file")
parser.add_argument("output", help="Path to the output CSV file")


def main(args):
    # This should avoid 100% CPU usage and
    # should make it faster according to
    # https://spacy.io/api/language#pipe
    os.environ['OPENBLAS_NUM_THREADS'] = "1"

    print(f"Processing input file {args.input}...")
    sampler = process_wikitext_corpus(args.input)

    with open(args.output, 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["label", "sentence"])

        print(f"Sampling {args.N} labels with {args.k} "
              f"examples from processed corpus...")
        writer.writerows(sampler.sample(N=args.N, k=args.k))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
