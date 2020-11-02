from interactive_generate import load_model, load_pickle, constrained_generate, generate

keyword_model = '../models/ROCstory_title_keywords_e500_h1000_edr0.4_hdr0.1.pt'
story_model = '../models/ROCstory_titlesepkey_story_e1000_h1500_edr0.2_hdr0.1.pt'
keyword_vocab = '../models/keyword_dict.pkl'
story_vocab = '../models/story_dict.pkl'
title_file = '../rocstory_plan_write/ROCStories_all_merge_tokenize.title.test'
cuda = False

kw_model = load_model(keyword_model, cuda)
st_model = load_model(story_model, cuda)
kw_vocab = load_pickle(keyword_vocab)
st_vocab = load_pickle(story_vocab)
kw_vocab_size = len(kw_vocab)
st_vocab_size = len(st_vocab)
with open(title_file, 'r') as fin:
    titles = fin.readlines()
max_kw_len = 15
max_st_len = 1000
kw_dedup = True
st_dedup = False
st_temperature = 0.3
kw_temperature = 0.15
st_eos_id = st_vocab.word2idx['<eos>']
kw_eos_id = kw_vocab.word2idx['<EOL>']

def generate_story(title, keywords, K, extendable):
    modified_keywords = constrained_generate(kw_model, kw_vocab, keywords, title, kw_eos_id, max_kw_len, kw_dedup, cuda, kw_temperature, K, extendable)

    prefix = modified_keywords
    tokens = generate(st_model, st_vocab, prefix, st_eos_id, max_st_len, st_dedup, cuda, st_temperature)

    return modified_keywords, tokens
