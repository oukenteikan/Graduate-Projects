'''
A script for creating training files from keywords.
Note that it doesn't throw out any data so for instance, if there are blank lines where keywords
weren't generated, it just makes blanks
'''

import argparse
import os
import numpy as np
import re


def read_story_file(filename):
    """read story file, return three lists, one of titles, one of keywords, one of stories"""
    title_list, kw_list, story_list = [], [], []
    with open(filename, 'r') as infile:
        for line in infile:
            title, rest = line.strip().split('<EOT>')
            kw, story = rest.split('<EOL>')
            title_list.append(title)
            kw_list.append(kw)
            story_list.append(story)
    return title_list, kw_list, story_list

def read_kw_file(filename):
    """read keywords file. Aligned with story file per line, kw # delimited. Return list of kw"""
    kw_list = []
    with open(filename, 'r') as infile:
        for line in infile:
            kw_list.append(line.strip())
    return kw_list

def recover_hyperparams(filename):
    """hacky. gets the hyperparams for RAKE from the filename to use in generating new filenames"""
    h_params = re.findall('\d', filename)
    return h_params


def write_training_files(titles, new_kw, stories, lines, splits, h_params, target_dir):
    """
    Takes a bunch of stuff, writes files for training
    :param titles: a list of strings representing titles
    :param new_kw: a list of strings of keywords, coindexed with titles
    :param stories: a list of story strings, coindexed with kw and titles
    :param lines: total number of lines (used for test-valid-train split)
    :param splits: the desired test-valid-train split, a string, entered as an optional arg
    :param h_params: the hyperparams used to generate the keyword, used in writing titles
    :param target_dir: the directory to write the titles
    :return: None
    """
    base_filename = 'ROCStories_all_merge_tokenize'
    # get num lines per split
    split_numbers = np.array(list(map(float, splits.split('-')))) * lines/10
    num_train, num_valid, num_test = list(map(int,map(round, split_numbers)))
    # write patterns
    title_kw_pattern = '{0} <EOT> {1} <EOL>'
    all_pattern =  '{0} <EOT> {1} <EOL> {2}'
    # make lists to store strings
    title_kw_list_train, title_kw_list_valid, title_kw_list_test = [], [], []
    all_train, all_valid, all_test = [], [], []
    for index in range(num_train):
        title_kw_list_train.append(title_kw_pattern.format(titles[index], new_kw[index]))
        all_train.append(all_pattern.format(titles[index], new_kw[index], stories[index]))
    for index in range(num_train, num_train+num_valid):
        title_kw_list_valid.append(title_kw_pattern.format(titles[index], new_kw[index]))
        all_valid.append(all_pattern.format(titles[index], new_kw[index], stories[index]))
    for index in range(num_train+num_valid, lines):
        title_kw_list_test.append(title_kw_pattern.format(titles[index], new_kw[index]))
        all_test.append(all_pattern.format(titles[index], new_kw[index], stories[index]))

    with open('{2}/{0}.titlesepkey.train.{1}'.format(base_filename, h_params, target_dir), 'w') as outfile:
        outfile.write('\n'.join(title_kw_list_train))
    with open('{2}/{0}.titlesepkeysepstory.train.{1}'.format(base_filename, h_params, target_dir), 'w') as outfile:
        outfile.write('\n'.join(all_train))
    with open('{2}/{0}.titlesepkey.dev.{1}'.format(base_filename, h_params, target_dir), 'w') as outfile:
        outfile.write('\n'.join(title_kw_list_valid))
    with open('{2}/{0}.titlesepkeysepstory.dev.{1}'.format(base_filename, h_params, target_dir), 'w') as outfile:
        outfile.write('\n'.join(all_valid))
    with open('{2}/{0}.titlesepkey.test.{1}'.format(base_filename, h_params, target_dir), 'w') as outfile:
        outfile.write('\n'.join(title_kw_list_test))
    with open('{2}/{0}.titlesepkeysepstory.test.{1}'.format(base_filename, h_params, target_dir), 'w') as outfile:
        outfile.write('\n'.join(all_test))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='../rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory',
                        help='file with all titles, keywords, and stories to read in')
    p.add_argument('--kw-dir', type=str, default='../Datasets/ROCstory_keywords/experiments/',
                   help='dir of files with alternative keywords in them, aligned with data-file. One line per set, # delimited between keywords (as they are multi-word)')
    p.add_argument('--split', type=str, default='8-1-1', help='test-valid-train split')
    p.add_argument('--outdir', type=str,
                   default='../rocstory_plan_write/keyword_experiments/',
                   help='target director for writing files to')
    args = p.parse_args()

    # read in overall file and store titles, keywords, stories
    titles, keywords, stories = read_story_file(args.data)
    total_lines = len(titles)

    # read in keywords file from a dir with all of them
    kw_filenames = sorted(os.listdir(args.kw_dir))
    print('Processing {} files in directory {}'.format(len(kw_filenames), args.kw_dir))

    for file in kw_filenames:
        new_kw = read_kw_file('{0}{1}'.format(args.kw_dir, file))
        hyperparams = '.'.join(recover_hyperparams(file))

        # generate all training files
        write_training_files(titles, new_kw, stories, total_lines, args.split, hyperparams, args.outdir)
        print('Finished writing all files.')
