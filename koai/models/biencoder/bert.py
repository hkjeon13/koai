import os
from torch import nn
from typing import Optional, Union
from transformers import (
    BertModel,
    PreTrainedModel,
    BertConfig,
    load_tf_weights_in_bert,
    AutoConfig
)


class BiBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bibert"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertForBiEncoder):
            module.gradient_checkpointing = value


class BertForBiEncoder(BiBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.query_bert = BertModel(config)

        self.key_bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.init_weights()

    def forward(
            self,
            query_input_ids=None,
            query_attention_mask=None,
            query_token_type_ids=None,
            query_position_ids=None,
            key_input_ids=None,
            key_attention_mask=None,
            key_token_type_ids=None,
            key_position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
    ) -> None:
        pass

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
            *model_args, **kwargs
    ):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        if "config" in kwargs:
            kwargs["config"] = config
        model, log = super(BertForBiEncoder, cls).from_pretrained(pretrained_model_name_or_path, output_loading_info=True, *model_args, **kwargs)

        if (k.startswith(("query_bert.encoder.layer", "key_bert.encoder.layer")) for k in log["missing_keys"]):
            model = cls(config)
            model.query_bert.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            model.key_bert.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        return model