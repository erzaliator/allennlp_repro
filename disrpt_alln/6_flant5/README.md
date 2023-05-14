---
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- samsum
metrics:
- rouge
model-index:
- name: google/flan-t5-small
  results:
  - task:
      name: Sequence-to-sequence Language Modeling
      type: text2text-generation
    dataset:
      name: samsum
      type: samsum
      config: samsum
      split: test
      args: samsum
    metrics:
    - name: Rouge1
      type: rouge
      value: 42.5796
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# google/flan-t5-small

This model is a fine-tuned version of [google/flan-t5-small](https://huggingface.co/google/flan-t5-small) on the samsum dataset.
It achieves the following results on the evaluation set:
- Loss: 1.6719
- Rouge1: 42.5796
- Rouge2: 18.3044
- Rougel: 35.2446
- Rougelsum: 38.8443
- Gen Len: 16.9048

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 1

### Training results

| Training Loss | Epoch | Step | Validation Loss | Rouge1  | Rouge2  | Rougel  | Rougelsum | Gen Len |
|:-------------:|:-----:|:----:|:---------------:|:-------:|:-------:|:-------:|:---------:|:-------:|
| 1.8163        | 1.0   | 1842 | 1.6719          | 42.5796 | 18.3044 | 35.2446 | 38.8443   | 16.9048 |


### Framework versions

- Transformers 4.28.0
- Pytorch 1.12.1+cu102
- Datasets 2.12.0
- Tokenizers 0.13.3
