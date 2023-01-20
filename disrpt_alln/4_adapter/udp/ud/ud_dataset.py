import glob
import logging
import os
import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from filelock import FileLock
from transformers import (
    # language_aware_data_collator,
    PreTrainedTokenizer,
)
from transformers import default_data_collator

import torch
from torch import nn
from torch.utils.data.dataset import Dataset, IterableDataset

logger = logging.getLogger(__name__)

class language_aware_data_collator(default_data_collator):
    def __init__(self, language='en'):
        super().__init__()
        self.language = 'en'

@dataclass
class InputExample:
    """
    A single training/test example for universal dependency parsing.

    Args:
        words: list. The words of the sequence.
        head_labels: (Optional) list. The labels for each word's dependency head. This should be
        specified for train and dev examples, but not for test examples.
        rel_labels: (Optional) list. The labels for the relations between each word and its respective head. This should be
        specified for train and dev examples, but not for test examples.
    """

    words: List[str]
    head_labels: Optional[List[int]]
    rel_labels: Optional[List[str]]


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a BertForBiaffineParsing model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    word_starts: List[int]
    labels_arcs: List[int]
    labels_rels: List[int]


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class UDDataset(Dataset):
    """
    Pytorch Dataset for universal dependency parsing.
    """

    features: List[InputFeatures]

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        labels: List[str],
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        mode: Union[Split, str] = Split.train,
    ):
        if isinstance(mode, Split):
            mode = mode.value
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            data_dir, "udp_cached_{}_{}_{}".format(mode, tokenizer.__class__.__name__, str(max_seq_length)),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                examples = read_examples_from_file(data_dir, mode)
                self.features = convert_examples_to_features(
                    examples=examples, label_list=labels, max_seq_length=max_seq_length, tokenizer=tokenizer
                )
                logger.info(f"Saving features into cached file {cached_features_file}")
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


class _MultiSourceUDDatasetIterator:

    def __init__(self, tc_datasets):
        datasets = sorted(list(tc_datasets.datasets.items()))
        self.generators = [dataset.generator() for _, dataset in datasets]
        self.n_steps = tc_datasets.n_steps
        self.step_count = 0 

    def __next__(self):
        if self.step_count >= self.n_steps:
            raise StopIteration()

        dataset = np.random.choice(self.generators)
        batch = next(dataset)
        self.step_count += 1
        return batch


class SingleSourceUDDataset(UDDataset):

    def __init__(
            self,
            language: str,
            batch_size: int,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Union[Split, str] = Split.train,
    ):
        super().__init__(
            data_dir, tokenizer, labels,
            max_seq_length=max_seq_length,
            overwrite_cache=overwrite_cache,
            mode=mode,
        )
        logging.info('Initialised %s dataset with %d examples' % (language, len(self)))
        self.language = language
        self.batch_size = batch_size
        self.data_collator = language_aware_data_collator(language)
        self.n_examples = len(self)

    def set_max_examples(self, n_examples):
        assert n_examples <= len(self)
        self.n_examples = n_examples

    def generator(self):
        indices = list(range(self.n_examples))
        while True:
            random.shuffle(indices)
            for batch_begin in range(0, self.n_examples - self.batch_size, self.batch_size):
                batch_end = batch_begin + self.batch_size
                batch = [self.features[indices[i]]
                         for i in range(batch_begin, batch_end)]
                batch = self.data_collator(batch)
                yield batch


class MultiSourceUDDataset(IterableDataset):

    def __init__(
        self,
        data_dir: str,
        languages_file: str,
        batch_size: int,
        n_steps: int,
        tokenizer: PreTrainedTokenizer,
        labels: List[str],
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        mode: Union[Split, str] = Split.train,
        max_examples: int = None,
    ):
        language_df = pd.read_csv(languages_file, na_filter=False)
        self.languages = []
        self.datasets = {}
        for i in range(language_df.shape[0]):
            language = language_df['iso_code'][i]
            self.languages.append(language)
            path = os.path.join(data_dir, language_df['path'][i])
            dataset = SingleSourceUDDataset(
                language=language,
                batch_size=batch_size,
                data_dir=path,
                tokenizer=tokenizer,
                labels=labels,
                max_seq_length=max_seq_length,
                overwrite_cache=overwrite_cache,
                mode=mode
            )
            self.datasets[language] = dataset

        if max_examples:
            n_examples = [(len(self.datasets[lang]), lang) for lang in self.languages]
            n_examples.sort()
            examples_left = max_examples
            for i, (dataset_size, language) in enumerate(n_examples):
                examples_per_language = examples_left // (len(n_examples) - i)
                examples_for_this_language = min(dataset_size, examples_per_language)
                logging.info(f'Dataset contains {examples_for_this_language} {language} examples')
                self.datasets[language].set_max_examples(examples_for_this_language)
                examples_left -= examples_for_this_language

        self.n_steps = n_steps

    def __len__(self):
        return self.n_steps

    def __iter__(self):
        return _MultiSourceUDDatasetIterator(self)


def get_file(data_dir: str, mode: Union[Split, str]) -> Optional[str]:
    if isinstance(mode, Split):
        mode = mode.value

    fp = os.path.join(data_dir, f"*-ud-{mode}.conllu")
    _fp = glob.glob(fp)
    if len(_fp) == 1:
        return _fp[0]
    elif len(_fp) == 0:
        return None
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[InputExample]:

    file_path = get_file(data_dir, mode)
    examples = []

    with open(file_path, "r", encoding="utf-8") as f:
        words: List[str] = []
        head_labels: List[int] = []
        rel_labels: List[str] = []
        for line in f.readlines():
            tok = line.strip().split("\t")
            if len(tok) < 2 or line[0] == "#":
                if words:
                    examples.append(InputExample(words=words, head_labels=head_labels, rel_labels=rel_labels))
                    words = []
                    head_labels = []
                    rel_labels = []
            if tok[0].isdigit():
                word, head, label = tok[1], tok[6], tok[7]
                words.append(word)
                head_labels.append(int(head))
                rel_labels.append(label.split(":")[0])
        if words:
            examples.append(InputExample(words=words, head_labels=head_labels, rel_labels=rel_labels))
    return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token=-1,
) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = [tokenizer.tokenize(w) for w in example.words]
        word_lengths = [len(w) for w in tokens]
        tokens_merged = []
        list(map(tokens_merged.extend, tokens))

        if 0 in word_lengths:
            logger.info("Invalid sequence with word length 0 filtered: %s", example.words)
            continue
        # Filter out sequences that are too long
        if len(tokens_merged) >= (max_seq_length - 2):
            logger.info("Sequence of len %d filtered: %s", len(tokens_merged), tokens_merged)
            continue

        encoding = tokenizer.encode_plus(
            tokens_merged,
            add_special_tokens=True,
            pad_to_max_length=True,
            max_length=max_seq_length,
            is_split_into_words=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True,
        )

        input_ids = encoding["input_ids"]
        token_type_ids = encoding["token_type_ids"]
        attention_mask = encoding["attention_mask"]

        pad_item = [pad_token]

        # pad or truncate arc labels
        labels_arcs = example.head_labels
        labels_arcs = labels_arcs + (max_seq_length - len(labels_arcs)) * pad_item

        # convert rel labels from map, pad or truncate if necessary
        labels_rels = [label_map[i] for i in example.rel_labels]
        labels_rels = labels_rels + (max_seq_length - len(labels_rels)) * pad_item

        # determine start indices of words, pad or truncate if necessary
        word_starts = np.cumsum([1] + word_lengths).tolist()
        word_starts = word_starts + (max_seq_length + 1 - len(word_starts)) * pad_item

        # sanity check lengths
        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(labels_arcs) == max_seq_length
        assert len(labels_rels) == max_seq_length
        assert len(word_starts) == max_seq_length + 1

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("tokens: %s", " ".join([str(x) for x in tokens_merged]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s", " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s", " ".join([str(x) for x in token_type_ids]))
            logger.info("labels_arcs: %s", " ".join([str(x) for x in labels_arcs]))
            logger.info("labels_rels: %s", " ".join([str(x) for x in labels_rels]))
            logger.info("word_starts: %s", " ".join([str(x) for x in word_starts]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                word_starts=word_starts,
                labels_arcs=labels_arcs,
                labels_rels=labels_rels,
            )
        )
    return features
