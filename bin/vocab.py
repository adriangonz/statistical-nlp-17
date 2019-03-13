"""
Tool to generate the vocab.
"""

from argparse import ArgumentParser

from src.data import generate_vocab, store_vocab

parser = ArgumentParser()
parser.add_argument("input", help="Path to the input CSV data set")
parser.add_argument("output", help="Path to the output JSON vocab")


def main(args):
    print("Generating vocab...")
    vocab = generate_vocab(args.input)

    print("Storing vocab...")
    store_vocab(vocab, args.output)

    print(f"Stored vocab of size {len(vocab)} at {args.output}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
