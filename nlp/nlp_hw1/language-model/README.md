# Plan-And-Write Automatic Storytelling

This repo contains the code and models for the AAAI19 paper: "
Plan-And-Write: Towards Better Automatic Storytelling" [[arxiv]](https://arxiv.org/abs/1811.05701) [[paper]]() [[bib]]()

# Citation

If you find this repo useful, please cite our paper.

@inproceedings{yao2018plan, 
  title={Plan-And-Write: Towards Better Automatic Storytelling}, 
  author={Yao, Lili and Peng, Nanyun and Weischedel, Ralph and Knight, Kevin and Zhao, Dongyan and Yan, Rui}, 
  booktitle={Proceedings of the Thirty-Third AAAI Conference on Artificial Intelligence (AAAI)}, 
  year={2019} 
}

----------

# Run the language model:
Story generation code based on a pipeline that generates a set of keywords from a title and then generates a story based on the title and the pre-planning step. 

## Train
Training takes in train, test, and validation data, as well as many possible hyperparameters that you are able to specify (see `main.py`). It outputs a trained model You will need this for future generation steps.

### For the title to storyline model in the paper: 

```python pytorch_src/main.py --batch_size 20 --train-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkey.train --valid-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkey.dev --test-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkey.test --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 500 --emsize 200 --nhid 300 --save model.pt --vocab-file vocab.pkl```

### For the title to title-storyline to story model in the paper: 

```python pytorch_src/main.py --batch_size 20 --train-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.train --valid-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.dev --test-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.test --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 500 --emsize 1000 --nhid 1000 --save model.pt --vocab-file vocab.pkl```

### For generic training of a language model (for more general usage): 

```python pytorch_src/main.py --batch_size 20 --train-data rocstory_data/train.txt --valid-data rocstory_data/valid.txt --test-data rocstory_data/test.txt --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 500 --save model.pt --vocab-file vocab.pkl```

Notes:

* `--save` takes the path of the location you want the pytorch models to be saved to.
* what your train and validation and test data are will vary depending that you're trying to do. 
Data we use to train is under `data/rocstory_plan_write` for ROC.

This will immediately print the values of the args being used in the run, which gives visibility into the defaults being used as well as the flags you specified. 
(These defaults can also be seen in `pytorch_src/main.py`). It looks like this: 
```
Args: Namespace(alpha=2, batch_size=20, beta=1, bptt=70, clip=0.25, cuda=True, dropout=0.4, 
dropoute=0.1, dropouth=0.25, dropouti=0.4, emsize=400, epochs=500, log_interval=200, lr=30, model='LSTM', 
nhid=1150, nlayers=3, nonmono=5, optimizer='sgd', resume='models/test_model.pt', 
save='models/test_model_resume.pt', seed=141, test_data='rocstory_data/test.txt', tied=True, 
train_data='rocstory_data/train_all.txt', valid_data='rocstory_data/valid.txt', wdecay=1.2e-06, wdrop=0.5, when=[-1])
```

Status will be printed at regular intervals to stdout. The interval is configurable but defaults to every 200 batches.
The model will be saved after every epoch. The maximum epochs is also configurable and defaults to 8k.
There is a `--resume` flag which takes a path to a pretrained model instead of starting from scratch each time.

## Generate

There is a `--task` flag specifying the type of generation, but it defaults to just generating text.

Note that the `--temperature` hyperparameter controls the conservatism of the output text. The example temperature results in fluent text that is less creative. 

```python pytorch_src/generate.py --vocab vocab.pkl --checkpoint model.pt --outf OUTPUT-FILE  --temperature 0.5```

## Conditional generate (generate storyline from title)

```
python pytorch_src/generate.py  
--checkpoint PATH_TO_PRETRAINED_MODEL  --vocab vocab.pkl --task cond_generate 
--conditional-data data/rocstory/ROCStories_all_merge_tokenize.title.test --cuda 
--temperature 0.5 --sents 100 --dedup --outf OUTPUT_FILE
```
Note that this will require a model trained/validated/tested with the `titlesepkey` data, which is different than 
the language model trained in the previous steps. 
* The format of this training data is `the bike accident <EOT> learned # sister # crashed # hill # nervous <EOL>`.
* The `--conditional-data` flag takes the titles from which to generate the storylines. That is, a file
(in this case `ROCStories_all_merge_tokenize.title.test`) which contains titles and <EOT> tags.


## Run the current storyline model to generate storylines:
 ```
python pytorch_src/generate.py --vocab vocab.pkl --checkpoint models/ROCstory_title_keywords_e500_h1000_edr0.4_hdr0.1.pt  --task cond_generate --conditional-data rocstory_plan_write/ROCStories_all_merge_tokenize.title.test --cuda --temperature 0.15 --sents 100 --dedup --outf generation_results/cond_generated_keywords_test_e500_h1000_edr0.4_hdr0.1_t0.15.txt
 ```

## Run title-storyline to story model:
 ```
 python pytorch_src/generate.py --vocab vocab.pkl --checkpoint models/ROCstory_titlesepkey_story_e1000_h1500_edr0.2_hdr0.1.pt  --task cond_generate --conditional-data  generation_results/cond_generated_keywords_test_e500_h1000_edr0.4_hdr0.1_t0.15.txt --cuda --temperature 0.3 --sents 100 --outf generation_results/cond_generated_keywords_test_e500_h1000_edr0.4_hdr0.1_t0.15.txt_lm_e1000_h1500_edr0.2_hdr0.1_t0.3.txt
 ```

## Run interactive story generation
### Make vocabulary dictionary

The vocab pickle generated has indices representing vocab words that correspond to those in the model.

```
	python pytorch_src/make_vocab.py \
		--train-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.train --valid-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.dev --test-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkeysepstory.test \
		--output models/story_dict.pkl
```
```
	python pytorch_src/make_vocab.py \
		--train-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkey.train --valid-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkey.dev --test-data rocstory_plan_write/ROCStories_all_merge_tokenize.titlesepkey.test \
		--output models/keyword_dict.pkl
```
### Run iteractive generation

Interactive story generation allows a user to interactively get results from entering any title.
For example, "secret notes" or "the bike accident" or "jury duty", etc.

User can enter a title, and the model will suggest a complete storyline, 
which looks something like this:
`mandy museum # excited painting # concern teacher # wanted # luckily enjoy` (from the title 
`the art trip`) where storyline keywords and phrases are separated by `#` symbols (as in the 
training data).
You can then accept or reject the storyline. If you reject it you provide a full alternate one 
yourself, from which a story is generated.

```
	python pytorch_src/interactive_generate.py \
		--keyword-model models/ROCstory_title_keywords_e500_h1000_edr0.4_hdr0.1.pt \
		--story-model models/ROCstory_titlesepkey_story_e1000_h1500_edr0.2_hdr0.1.pt \
		--keyword-vocab models/keyword_dict.pkl \
		--story-vocab models/story_dict.pkl \
		--titles rocstory_plan_write/ROCStories_all_merge_tokenize.title.test \
		--cuda --temperature 0.3
```
