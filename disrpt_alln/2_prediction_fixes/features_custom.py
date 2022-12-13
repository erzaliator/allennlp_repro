import torch

def get_combined_feature_tensor_2(features, feature_list, feature_modules):
    output_tensors = []
    i = 0
    for module_key in feature_list:
        module = feature_modules[module_key]
        feature = features[module_key]
        if module_key in ['sat_children', 'nuc_children']:
            feature = feature.unsqueeze(-1)
        output_tensor = module(feature)
        output_tensors.append(output_tensor)
        i += 1
    output_tensors = torch.cat(output_tensors, dim=-1)
    if len(output_tensors.shape)==1: 
        output_tensors = torch.unsqueeze(output_tensors, 0)
    return output_tensors