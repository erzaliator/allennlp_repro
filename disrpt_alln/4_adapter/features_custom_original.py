import math
from typing import Tuple, Dict, List, Any, Union

import torch
from allennlp.common import FromParams, Registrable
import scipy.stats as stats


from allennlp.data import Vocabulary, Field
from allennlp.data.fields import TextField, TensorField, SequenceLabelField, LabelField

def get_vocab_feature_name(feature_name):
    if 'func' in feature_name: return 'func'
    if 'depdir' in feature_name: return 'depdir'
    return feature_name

def get_combined_feature_tensor_2(features, feature_list, feature_modules):
    output_tensors = []
    i = 0
    for module_key in feature_list:
        module = feature_modules[module_key]
        try:
            feature = features[module_key].squeeze()
        except:
            feature = features[:, i:i+1].squeeze()
        if module_key in ['sat_children', 'nuc_children', 'length_ratio', 'doclen']:
            feature = feature.unsqueeze(-1)
        try:
            output_tensor = module(feature)
        except:
            print(module_key, module, feature)
            raise ValueError()
        output_tensors.append(output_tensor)
        i += 1
    if len(output_tensors)==0:
        output_tensors = torch.empty(0,0)
    else:
        output_tensors = torch.cat(output_tensors, dim=-1)
    if len(output_tensors.shape)==1: 
        output_tensors = torch.unsqueeze(output_tensors, 0)
    return output_tensors

def get_feature_modules(feature_list, vocab: Vocabulary) -> Tuple[torch.nn.ModuleDict, int]:
    """
    Returns a PyTorch `ModuleDict` containing a module for each feature in `token_features`.
    This function tries to be smart: if the feature is numeric, it will not do anything, but
    if it is categorical (as indicated by the presence of a `label_namespace`), then the module
    will be a `torch.nn.Embedding` with size equal to the ceiling of the square root of the
    categorical feature's vocabulary size. We could be a lot smarter of course, but this will
    get us going.

    Args:
        features: a dict of `TokenFeatures` describing all the categorical features to be used
        vocab: the initialized vocabulary for the model

    Returns:
        A 2-tuple: the ModuleDict, and the summed output dimensions of every module, for convenience.
    """
    
    modules: Dict[str, torch.nn.Module] = {}
    total_dims = 0
    # if isinstance(features, FeatureBundle):
    #     keys = features.corpus_keys
    #     config_dict = features.features
    # else:
    #     keys = features.keys()
    #     config_dict = features

    keys = feature_list
    # config_dict = vocab#.get_token_to_index_vocabulary(vocab_feature_name)
    for key in keys:
        vocab_feature_name = get_vocab_feature_name(key)
        # config = vocab.get_token_to_index_vocabulary(key)
        # ns = config.label_namespace
        if key in ['sat_children', 'nuc_children', 'length_ratio', 'doclen']:
            modules[key] = torch.nn.Identity()
            total_dims += 1
        else:
            size = vocab.get_vocab_size(vocab_feature_name)
            # if size <= 5:
            #    modules[key] = torch.nn.Identity()
            #    total_dims += size
            # else:
            edims = math.ceil(math.sqrt(size))
            total_dims += edims
            modules[key] = torch.nn.Embedding(size, edims, padding_idx=(0))

    return torch.nn.ModuleDict(modules), total_dims