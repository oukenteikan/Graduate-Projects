"""A script to measure the keyword incorporation metrics of stories and storylines"""

import argparse
from itertools import combinations

import numpy as np
import re
import sys

def read_w2v(w2v_path, word2index, n_dims=300, unk_token="unk"):
    """takes tokens from files and returns word vectors
    :param w2v_path: path to pretrained embedding file
    :param word2index: Counter of tokens from processed files
    :param n_dims: embedding dimensions
    :param unk_token: this is the unk token for glove 840B 300d. Ideally we make this less hardcode-y
    :return numpy array of word vectors
    """
    print('Getting Word Vectors...', file=sys.stderr)
    vocab = set()
    # hacky thing to deal with making sure to incorporate unk tokens in the form they are in for a given embedding type
    if unk_token not in word2index:
        word2index[unk_token] = 0 # hardcoded, this would be better if it was a method of a class

    word_vectors = np.zeros((len(word2index), n_dims))  # length of vocab x embedding dimensions
    with open(w2v_path) as file:
        lc = 0
        for line in file:
            lc += 1
            line = line.strip()
            if line:
                row = line.split()
                token = row[0]
                if token in word2index or token == unk_token:
                    vocab.add(token)
                    try:
                        vec_data = [float(x) for x in row[1:]]
                        word_vectors[word2index[token]] = np.asarray(vec_data)
                        if lc == 1:
                            if len(vec_data) != n_dims:
                                raise RuntimeError("wrong number of dimensions")
                    except:
                        print('Error on line {}'.format(lc), file=sys.stderr)
                    # puts data for a given embedding at an index based on the word2index dict
                    # end up with a matrix of the entire vocab
    tokens_without_embeddings = set(word2index) - vocab
    print('Word Vectors ready!', file=sys.stderr)
    print('{} tokens from text ({:.2f}%) have no embeddings'.format(
        len(tokens_without_embeddings), len(tokens_without_embeddings)*100/len(word2index)), file=sys.stderr)
    print('Tokens without embeddings: {}'.format(tokens_without_embeddings), file=sys.stderr)
    print('Setting those tokens to unk embedding', file=sys.stderr)
    for token in tokens_without_embeddings:
        word_vectors[word2index[token]] = word_vectors[word2index[unk_token]]
    return word_vectors


def get_tokens(files):
    """take a list of filepaths, returns word2idx dict"""
    print('Getting tokens ... ...', file=sys.stderr)
    all_tokens = set()
    for path in files:
        with open(path, 'r') as infile:
            all_tokens.update(set(infile.read().strip().split()))
    word2index = dict(map(reversed, enumerate(list(all_tokens))))
    return word2index


def cos_sim(v1, v2):
    return v1.dot(v2) / (np.sqrt(v1.dot(v1)) * np.sqrt(v2.dot(v2)))


def cos_sim_array(vec, vec_array):
    """
    take dot product of 2 vectors. which reduces dimensionality and gives me an array of results.
    IMPORTANT that vec_array is first arg as a result
    :param vec: a vector
    :param vec_array: an array of vectors
    :return: cosine_sim_array of the cosine similarity between the vector and each vector in the array
    """
    dot_prod_array = np.dot(vec_array, vec)
    len_vec_array, len_x_d = (vec_array**2).sum(axis=1) ** .5, (vec ** 2).sum() ** .5
    cosine_sim_array = np.divide(dot_prod_array, len_vec_array*len_x_d)
    return cosine_sim_array


def remove_chars(text: str, remove='#') -> str:
    """take a string and optional chars to remove and returns string without them"""
    return re.sub(r'[{}]'.format(remove), '', text)


def make_vec_array(word_list: list, word_vectors, word2index: dict, drop_set={'#', '<EOL>', '<EOT>', '<\s>'}):
    """take a list of strings, an array of word vectors, return a numpy array of word vectors"""
    vecs = [np.array(word_vectors[word2index.get(word, 0)])
            for word in word_list if word not in drop_set]
    return np.array(vecs)


def calc_similarity(storyline_path, story_path, word2index, word_vectors):
    """calculates cosine similarity between keywords in storyline and between keywords in storyline
    and corresponding sentence in story. Averaged over all """
    keyword_relatedness = 0
    keyword_incorporation_rate = 0

    storylines, stories = [], []
    with open(storyline_path, 'r') as infile:
        for line in infile:
            processed_line = remove_chars(line).strip().split()[:-1]  # remove the <EOL> at the end
            storylines.append(processed_line)
    with open(story_path, 'r') as infile:
        for line in infile:
            processed_line = remove_chars(line).strip().split()
            stories.append(processed_line)

    num_storylines = len(storylines)
    assert(num_storylines == len(stories)), "Mismatch between number of storylines and number of stories"
        
    # loop through stories and storylines and calc similarities
    for i in range(num_storylines):
        storyline_word_array = make_vec_array(storylines[i], word_vectors, word2index)  # all storyline vectors
        story_word_array = make_vec_array(stories[i], word_vectors, word2index)  # all story word vectors
        # calculate the similarities between the storyline words
        # this is the cumulative cosine similarity between each word and all the other words then averaged
        num_words_in_storyline = len(storyline_word_array)
        storyline_idx_combinations = list(combinations(range(num_words_in_storyline), 2))
        this_storyline_relatedness = 0
        for kw1, kw2 in storyline_idx_combinations:
                this_storyline_relatedness += cos_sim(storyline_word_array[kw1], storyline_word_array[kw2])
        #print("KW Relatedness", this_storyline_relatedness/len(storyline_idx_combinations))  # to debug individual lines
        keyword_relatedness += this_storyline_relatedness/len(storyline_idx_combinations)  # since end up with 2x comparisons as words

        # calculate the similarities between the word and the sentence
        # this is the maximum cosine sim between each keyword and any other word in the sentence, summed over keywords then averaged
        this_incorporation_rate = 0
        for kw_vec in storyline_word_array:
            cosine_max = np.nanmax(cos_sim_array(kw_vec, story_word_array))
            this_incorporation_rate += cosine_max
        #print("KW Incorporation", this_incorporation_rate/num_words_in_storyline)  # to debug individual lines
        keyword_incorporation_rate += this_incorporation_rate/num_words_in_storyline

    # report average over all in set
    keyword_relatedness /= num_storylines
    keyword_incorporation_rate /= num_storylines
    print('Metrics for {} samples'.format(num_storylines))
    print('dynamic relatedness : {:.2f}'.format(keyword_relatedness))
    print('dynamic keyword_incorporation_rate : {:.2f}'.format(keyword_incorporation_rate))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('storyline_file', type=str,
                        help='location of file with storylines')
    parser.add_argument('story_file', type=str, help='location of story file')
    parser.add_argument('wordvec_file', type=str, help='path to wordvec file' )
    args = parser.parse_args()

    word2idx = get_tokens([args.storyline_file, args.story_file]) # takes list of arbitrarily many files
    word_vectors = read_w2v(args.wordvec_file, word2idx)
    calc_similarity(args.storyline_file, args.story_file, word2idx, word_vectors)


