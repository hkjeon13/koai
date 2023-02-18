from collections import OrderedDict
from typing import Union, Optional, Tuple

import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.auto.auto_factory import _LazyAutoMapping, _BaseAutoModelClass
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.bert.modeling_bert import BertPooler


# TODO:  not yet - Now it is just concept!!
class BertForRelationExtraction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.subject_pooler = BertPooler(config)
        self.object_pooler = BertPooler(config)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def set_subject_object_maps(self, mapping_dict: dict):
        self.subject_maps = mapping_dict.get("subject")
        self.object_maps = mapping_dict.get("object")

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            subject_mask: Optional[torch.Tensor] = None,
            object_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sub_embeddings = subject_mask * sequence_output
        obj_embeddings = object_mask * sequence_output

        pooled_output = outputs[1]  # "pooled_output" is the meaninig of sentence
        subject_pooled_output = None
        object_pooled_output = None
        pooled_output = self.dropout(pooled_output)
        subject_pooled_output = self.dropout(subject_pooled_output)
        object_pooled_output = self.dropout(object_pooled_output)
        logits = self.classifier(pooled_output)


RE_MODEL = OrderedDict([
    ("bert", BertForRelationExtraction),
])

MODEL_CONFIG = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, RE_MODEL
)


class AutoModelForRelationExtraction(_BaseAutoModelClass):
    _model_mapping = MODEL_CONFIG


if __name__ == "__main__":
    model = BertForRelationExtraction.from_pretrained("klue/bert-base")
