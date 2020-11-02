import logging
import sys
import argparse


class ParserWithUsage(argparse.ArgumentParser):
    """ A custom parser that writes error messages followed by command line usage documentation."""

    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def construct_parameters():
    """ Constructs command line parameters for the script."""
    parser = ParserWithUsage()
    parser.description = "Extracts keywords from prompts."
    parser.add_argument("--input", help="File containing the prompts (one prompt per line)", required=True)
    parser.add_argument("--input_stopwords", help="File containing the list of stop words", required=True)
    parser.add_argument("--output", help="Output file where to store the keywords (one set of keywords per line)",
                        required=True)

    args = parser.parse_args()
    return args


def read_stopwords(path):
    """
    Loads stop words.
    :param path: path to file containing the smart stop words
    :return: set containing all stop words
    """
    stopwords = set()
    with open(path, "r") as file_stopwords:
        for line in file_stopwords:
            line = line.rstrip()
            if line[0] == "#":
                continue
            else:
                stopwords.add(line)
    return stopwords


def read_prompt(file_object):
    """
    Uses a generator to read one prompt at a time.
    :file_object self-describing, a file object
    :returns one prompt at a time via a generator
    """
    idx = -1
    while True:
        data = file_object.readline()
        if not data:
            break
        data = data.strip().split(" ")
        # get rid of the prompt type
        data = data[3:]
        data_to_return = ""
        for token in data:
            data_to_return += token.lower() + " "
        idx += 1
        yield (data_to_return, idx)


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO,
                        datefmt='%m/%d/%Y %H:%M:%S')
    args = construct_parameters()
    args_input_prompts = args.input
    args_input_stopwords = args.input_stopwords
    args_output = args.output
    logging.info("STARTED")

    logging.info("Will read stop words from {0}".format(args_input_stopwords))
    stop_words = read_stopwords(args_input_stopwords)

    logging.info("Will prompts from {0}".format(args_input_prompts))
    logging.info("Will write keywords to {0}".format(args_output))
    logging.info("Processing prompts...")
    with open(args_output, "w") as file_output:
        with open(args_input_prompts, "r") as file_input:
            for prompt, idx_prompt in read_prompt(file_input):
                first_sentence = prompt.split(" ")
                if first_sentence is not None:
                    keywords = []
                    for token in first_sentence:
                        if len(token) > 1 and token not in stop_words:
                            keywords.append(token)
                    for idx, k in enumerate(keywords):
                        file_output.write(k)
                        if idx < len(keywords) - 1:
                            file_output.write(" ")
                    file_output.write("\n")

    logging.info("DONE")


if __name__ == "__main__":
    main()
