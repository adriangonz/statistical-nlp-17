"""
Tool to generate the vocabulary.
"""
import json

from argparse import ArgumentParser

from src.data import generate_vocabulary, get_vocab_state

parser = ArgumentParser()
parser.add_argument("input", help="Path to the input CSV data set")
parser.add_argument("output", help="Path to the output JSON vocabulary")


def main(args):
    print("Generating vocabulary...")
    vocab = generate_vocabulary(args.input)
    vocab_state = get_vocab_state(vocab)

    print("Storing vocabulary...")
    with open(args.output, 'w') as file:
        json.dump(vocab_state, file)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
