import logging
import sys
import argparse


class ParserWithUsage(argparse.ArgumentParser):
    """ A custom parser that writes error messages followed by command line usage documentation."""

    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def create_arguments():
    parser = ParserWithUsage()
    parser.description = "Removes stories that have no keywords"
    parser.add_argument("--input_keywords", help="Input file containing keywords", required=True)
    parser.add_argument("--input_storylines", help="Input file containing story lines", required=True)
    parser.add_argument("--input_stories", help="Input file containing stories", required=True)
    parser.add_argument("--output", help="Output directory (will create clean files here)", required=True)

    return parser.parse_args()


def get_file_name(file_path: str) -> str:
    """
    Returns the name of a file from a file path.
    :param file_path: file path
    :return: name of file
    """
    from pathlib import Path
    p = Path(file_path)
    return str(p.name)


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO,
                        datefmt='%m/%d/%Y %H:%M:%S')

    args = create_arguments()
    args_input_keywords = args.input_keywords
    args_input_storylines = args.input_storylines
    args_input_stories = args.input_stories
    args_output_dir = args.output
    logging.info("STARTED")

    logging.info("Identifying empty keyword sets")
    lines_to_skip_keywords = get_lines_with_empty_keywords(args_input_keywords)
    lines_to_skip_train = get_lines_with_empty_keywords(args_input_storylines)
    lines_to_skip = lines_to_skip_keywords.union(lines_to_skip_train)
    logging.info("Writing new files")
    write_lines_to_keep(args_input_keywords, args_output_dir + "/" + get_file_name(args_input_keywords), lines_to_skip)
    write_lines_to_keep(args_input_storylines, args_output_dir + "/" + get_file_name(args_input_storylines), lines_to_skip)
    write_lines_to_keep(args_input_stories, args_output_dir + "/" + get_file_name(args_input_stories), lines_to_skip)
    logging.info("DONE")


def write_lines_to_keep(file_path: str, output_path: str, lines_to_skip: set) -> None:
    """
    Copies only lines from the input file that are not listed in the set of lines to skip.
    :param file_path: where to read the input from
    :param output_path: where to write the output
    :param lines_to_skip: which lines to skip
    :return: nothing
    """
    with open(output_path, "w") as file_out:
        with open(file_path, "r") as file_input:
            line_number = -1
            for line in file_input:
                line_number += 1
                if line_number not in lines_to_skip:
                    file_out.write(line)


def get_lines_with_empty_keywords(file_path: str) -> set:
    """
    Identifies lines in the file that are empty.
    :param file_path: path to keywords file
    :return: set of lines that are empty
    """
    empty_keywords = set()
    with open(file_path, "r") as file_obj:
        line_number = -1
        for line in file_obj:
            line = line.rstrip()
            line_number += 1
            if len(line) == 0:
                empty_keywords.add(line_number)
    return empty_keywords


if __name__ == "__main__":
    main()
