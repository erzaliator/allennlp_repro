# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from torch import nn

from transformers import (
    AdapterConfig,
    AdapterType,
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    language_aware_data_collator,
    MultiLingAdapterArguments,
    TrainingArguments,
    set_seed,
)

from ud.modeling_biaffine import TransformerForBiaffineParsing
from ud.ud_dataset import Split, UDDataset
from ud.utils_udp import UD_HEAD_LABELS, DependencyParsingTrainer, UDTrainingArguments

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    languages_file: str = field(
        metadata={"help": "CSV file containing languages to evaluate on."}
    )
    force_adapter: Optional[str] = field(
        default=None,
        metadata={'help': 'Forces the adapter for a particular language to be used regardless of the target language.'}
    )
    use_closest_adapter: bool = field(
        default=False,
        metadata={'help': 'If an adapter for the language is not available, use the adapter for the most similar language.'}
    )
    eval_split: Optional[str] = field(
        default=Split.dev,
        metadata={"help": "dev or test."},
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, UDTrainingArguments, MultiLingAdapterArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            f"Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s.%(msecs)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    labels = UD_HEAD_LABELS
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
        pad_token_id=-1,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )
    model = TransformerForBiaffineParsing.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        name_or_path=model_args.model_name_or_path,
    )

    task_name = 'udp'

    # Setup adapters
    adapter_names = None
    if adapter_args.train_adapter:
        # check if adapter already exists, otherwise add it
        if task_name not in model.config.adapters.adapter_list(AdapterType.text_task):
            # resolve the adapter config
            if adapter_args.adapter_omit_final_layer:
                leave_out = [model.config.num_hidden_layers - 1]
            else:
                leave_out = []
            adapter_config = AdapterConfig.load(
                adapter_args.adapter_config,
                non_linearity=adapter_args.adapter_non_linearity,
                reduction_factor=adapter_args.adapter_reduction_factor,
                leave_out=leave_out,
            )
            # load a pre-trained from Hub if specified
            if adapter_args.load_adapter:
                model.load_adapter(
                    adapter_args.load_adapter, AdapterType.text_task, config=adapter_config, load_as=task_name,
                )
            # otherwise, add a fresh adapter
            else:
                model.add_adapter(task_name, AdapterType.text_task, config=adapter_config)
        # optionally load a pre-trained language adapter
        if adapter_args.load_lang_adapter:
            # resolve the language adapter config
            if adapter_args.lang_adapter_omit_final_layer:
                leave_out = [model.config.num_hidden_layers - 1]
            else:
                leave_out = []
            if adapter_args.load_lang_adapter.endswith('.csv'):
                adapter_df = pd.read_csv(adapter_args.load_lang_adapter, na_filter=False)
                adapter_dir = os.path.dirname(adapter_args.load_lang_adapter)
                adapter_list = [
                        (lang, os.path.join(adapter_dir, lang))
                        for lang in adapter_df['iso_code']
                ]
                adapter_names = [[None], [task_name]]
            else:
                adapter_list = [(adapter_args.language, adapter_args.load_lang_adapter)]
                adapter_names = [[adapter_args.language], [task_name]]
            for language, adapter_dir in adapter_list:
                lang_adapter_config = AdapterConfig.load(
                    adapter_args.lang_adapter_config,
                    non_linearity=adapter_args.lang_adapter_non_linearity,
                    reduction_factor=adapter_args.lang_adapter_reduction_factor,
                    leave_out=leave_out,
                )
                # load the language adapter from Hub
                lang_adapter_name = model.load_adapter(
                    adapter_dir,
                    AdapterType.text_lang,
                    config=lang_adapter_config,
                    load_as=language,
                    with_embeddings=True,
                )
        # Freeze all model weights except of those of this adapter
        model.train_adapter([task_name])

    for name, param in model.named_parameters():
        logging.info('%s: %s' % (name, str(param.requires_grad)))

    df = pd.read_csv(data_args.languages_file, na_filter=False)
    for i in range(df.shape[0]):
        logger.info('Evaluating on ' + df['language'][i])
        language = df['iso_code'][i]
        treebank = df['treebank'][i]
        if data_args.force_adapter:
            language_adapter = data_args.force_adapter
        else:
            if data_args.use_closest_adapter:
                language_adapter = model.config.adapters.get_closest_language_adapter(language)
                logging.info(f'Using {language_adapter} adapter for {language}.')
            else:
                language_adapter = language

        eval_dataset = (
            UDDataset(
                data_dir=os.path.join(data_args.data_dir, treebank),
                tokenizer=tokenizer,
                labels=labels,
                max_seq_length=data_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
                mode=data_args.eval_split,
            )
            if training_args.do_eval
            else None
        )

        data_collator = language_aware_data_collator(language_adapter)

        # Initialize our Trainer
        trainer = DependencyParsingTrainer(
            model=model,
            args=training_args,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            adapter_names=adapter_names
        )

        # Evaluation
        results = {}
        if training_args.do_eval:
            logger.info("*** Evaluate ***")

            result = trainer.evaluate()

            output_eval_dir = os.path.join(training_args.output_dir, language)
            if not os.path.exists(output_eval_dir):
                os.mkdir(output_eval_dir)
            output_eval_file = os.path.join(
                    output_eval_dir,
                    "%s_eval_results.txt" % task_name)
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results *****")
                    for key, value in result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

                results.update(result)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
