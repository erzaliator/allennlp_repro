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
        if module_key in ['sat_children', 'nuc_children']:
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

def make_custom_dims():
    size_dict={'distance':5, 'u1_depdir':5, 'u2_depdir':5, 'u2_func':46, 'u1_position':12, 'u2_position':12}
    dim_dict={'distance':3, 'u1_depdir':3, 'u2_depdir':3, 'u2_func':6, 'u1_position':4, 'u2_position':4}
    return size_dict, dim_dict

def get_feature_modules(feature_list, vocab: Vocabulary, use_allennlp_dims=True,  
                            ) -> Tuple[torch.nn.ModuleDict, int]:
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
        if key in ['sat_children', 'nuc_children']:
            modules[key] = torch.nn.Identity()
            total_dims += 1
        else:
            if use_allennlp_dims:
                size = vocab.get_vocab_size(vocab_feature_name)
                edims = math.ceil(math.sqrt(size))
            else:
                size_dict, dim_dict = make_custom_dims()
                size = size_dict[key]
                edims = dim_dict[key]
            total_dims += edims
            modules[key] = torch.nn.Embedding(size, edims, padding_idx=(0))

    return torch.nn.ModuleDict(modules), total_dims


def get_mapping_from_bin(column_name, dict_val):
    bins = {
      'distance': [[-1e9, -8], [-8, -2], [-2, 0], [0, 2], [2, 8], [8, 1e9]],
      'u1_position': [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1.0], [1.0, 1e9]],
      'u2_position': [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1.0], [1.0, 1e9]],
      'lex_overlap_length': [[0, 2], [2, 7], [7, 1e9]]
    }   
    bins = bins[column_name]
    for b,i in zip(bins, range(len(bins))):
      left = b[0]
      right = b[1]
      if left<=dict_val and right>=dict_val: return i


def apply_bins(train_df, test_df, val_df):
    bins = {
      'distance': [[-1e9, -8], [-8, -2], [-2, 0], [0, 2], [2, 8], [8, 1e9]],
      'u1_position': [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1.0], [1.0, 1e9]],
      'u2_position': [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1.0], [1.0, 1e9]],
      'lex_overlap_length': [[0, 2], [2, 7], [7, 1e9]]
    }   
    for df in [train_df, test_df, val_df]:
      for feature_name in bins.keys():
        if feature_name=='u2_func':
          print(df[feature_name].unique())
          raise ValueError()
        df[feature_name] = df[feature_name].apply(lambda x: get_mapping_from_bin(feature_name, float(x)))
        print(feature_name, df[feature_name].unique())

    return train_df, test_df, val_df