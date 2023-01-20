from typing import Dict

import torch
from torch import nn
from allennlp.data import Vocabulary
from transformers import XLMRobertaConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaPooler, XLMRobertaEncoder, XLMRobertaEmbeddings, XLMRobertaPreTrainedModel


# from gucorpling_models.features import get_feature_modules, get_combined_feature_tensor, FeatureBundle
from features_custom2 import get_combined_feature_tensor_2, get_feature_modules


def insert_into_sequence(batched_sequence, batched_sequence_item, sequence_position):
    """Given a sequence [b, seqlen, d] and an item [b, 1, d], insert the item at the given position"""
    device = batched_sequence.device
    chunks = [
        batched_sequence[:, :sequence_position],
        batched_sequence_item,
        batched_sequence[:, sequence_position:]
    ]
    return torch.cat(chunks, dim=1).to(device)


def zero_pad(batched_sequence_item, d):
    """Pad an item to d in the 3rd dimension"""
    batched_sequence_item = batched_sequence_item.squeeze(-1)
    assert batched_sequence_item.shape[1] == 1
    if len(batched_sequence_item.shape)==2: # handle for no features
        batched_sequence_item = batched_sequence_item.unsqueeze(1)
    if batched_sequence_item.shape[2] > d:
        raise Exception("Too many feature dimensions! " + str(batched_sequence_item.shape[2]) + " > " + f"{d}")
    diff = d - batched_sequence_item.shape[2]
    zeros = torch.zeros((batched_sequence_item.shape[0], 1, diff)).to(batched_sequence_item.device)
    return torch.cat((batched_sequence_item, zeros), dim=2).to(batched_sequence_item.device)


class FeaturefulBert(XLMRobertaPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    # def get_combined_feature_tensor_2(self, feats, kwargs):
    #     output_tensors = []
    #     for module_key, module in self.feature_modules.items():
    #         output_tensor = module(kwargs[module_key])
    #         if len(output_tensor.shape) == 1:
    #             output_tensor = output_tensor.unsqueeze(-1)
    #         output_tensors.append(output_tensor)

    #     combined_feature_tensor = torch.cat(output_tensors, dim=1)
    #     print('combined feature tenor')
    #     print(combined_feature_tensor)
    #     return combined_feature_tensor

    def __init__(self, config: XLMRobertaConfig, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = XLMRobertaEmbeddings(config)
        self.encoder = XLMRobertaEncoder(config)
        self.pooler = XLMRobertaPooler(config) if add_pooling_layer else None

        self.init_weights()

        self.features = None
        self.feature_modules = None
        self.feature_dims = None
        self.feature_projector = None

    # def init_feature_modules(self):
        # self.features =  ['distance', 'u1_depdir', 'u2_depdir', 'u2_func', 'u1_position', 'u2_position', 'sat_children', 'nuc_children']
        # self.feature_modules = nn.ModuleDict()
        # self.dims = 0
        # for feature in self.features:
        #     print(feature)
        #     if feature=='distance':
        #         self.feature_modules[feature] =nn.Embedding(5, 3, padding_idx=0) #6,3
        #         self.dims += 3
        #     elif feature=='u1_depdir':
        #         self.feature_modules[feature] = nn.Embedding(5, 3, padding_idx=0)
        #         self.dims += 3
        #     elif feature=='u2_depdir':
        #         self.feature_modules[feature] = nn.Embedding(5, 3, padding_idx=0)
        #         self.dims += 3
        #     elif feature=='u2_func':
        #         self.feature_modules[feature] = nn.Embedding(23, 5, padding_idx=0)
        #         self.dims += 5
        #     elif feature=='u1_position':
        #         self.feature_modules[feature] = nn.Embedding(12, 4, padding_idx=0)
        #         self.dims += 4
        #     elif feature=='u2_position':
        #         self.feature_modules[feature] = nn.Embedding(12, 4, padding_idx=0)
        #         self.dims += 4
        #     elif 'sat_children' in self.features:        
        #         self.feature_modules[feature] = nn.Identity()
        #         self.dims += 1
        #     elif 'nuc_children' in self.features:
        #         self.feature_modules[feature] = nn.Identity()
        #         self.dims += 1
        #     else: raise ValueError()
        # self.feature_dims = 24
        # self.feature_projector = torch.nn.Linear(self.feature_dims + 1, self.config.hidden_size)

    def init_feature_modules(self, feature_list, vocab):
        self.features = feature_list
        if feature_list is not None and len(feature_list) > 0:
            feature_modules, feature_dims = get_feature_modules(feature_list, vocab)
            self.feature_modules = feature_modules
        else:
            self.feature_modules = None
            feature_dims = 0
        self.feature_dims = feature_dims
        self.feature_projector = torch.nn.Linear(feature_dims + 1, self.config.hidden_size)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        direction_tensor = None,
        feature_list = None,
        feature_values = None,
        **kwargs
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        segments = []
        if len(feature_list) > 0:
            feature_tensor = get_combined_feature_tensor_2(feature_values, feature_list, self.feature_modules)
            segments.append(feature_tensor)
        # direction_tensor = kwargs.get('direction')
        segments.append(direction_tensor.unsqueeze(-1))
        feature_tensor = torch.cat(segments, dim=1).to(embedding_output.device)
        projected_feature_tensor = zero_pad(feature_tensor.unsqueeze(1), embedding_output.shape[2])
        #projected_feature_tensor = self.feature_projector(feature_tensor).unsqueeze(1)
        # Add the feature tensor at the 2nd position in the sequence, i.e. after CLS
        modified_embedding_output = insert_into_sequence(embedding_output, projected_feature_tensor, 1)
        # Modify the attention mask by taking the value at 0 and repeating it at the front: the value at 0 will
        # always be a non-masked value, which will get us the proper mask for the extended sequence
        modified_extended_attention_mask = torch.cat(
            (extended_attention_mask[:, :, :, :1], extended_attention_mask),
            dim=3
        )

        encoder_outputs = self.encoder(
            modified_embedding_output,
            attention_mask=modified_extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0] #4, 513, 768 v allen 4, 50, 568
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


    
