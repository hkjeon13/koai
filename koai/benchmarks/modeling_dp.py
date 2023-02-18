import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel, PretrainedConfig, AutoConfig
from transformers.models.bert.modeling_bert import ModelOutput

logger = logging.getLogger(__file__)


@dataclass
class BertDependencyParsingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    head_prediction_logits: torch.FloatTensor = None
    dp_prediction_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertDependenceParsingHeads(nn.Module):
    def __init__(self, config, max_seq_length, num_relations):
        super().__init__()
        self.head_classifier = nn.Linear(config.hidden_size, max_seq_length)
        self.rel_classifier = nn.Linear(config.hidden_size, num_relations)

    def forward(self, sequence_output):
        head_scores = self.head_classifier(sequence_output)
        rel_scores = self.rel_classifier(sequence_output)
        return head_scores, rel_scores


class BertModelForDependencyParsing(BertPreTrainedModel):
    def __init__(self, config, max_seq_length, num_relations):
        super().__init__(config)
        self.max_seq_length = max_seq_length
        self.num_relations = num_relations
        self.bert = BertModel(config)
        self.cls = BertDependenceParsingHeads(config, max_seq_length, num_relations)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            head_labels: Optional[torch.Tensor] = None,
            dp_labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BertDependencyParsingOutput]:

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

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, relationship_scores = self.cls(sequence_output)

        total_loss = None
        if head_labels is not None and dp_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.max_seq_length), head_labels.view(-1))
            next_sentence_loss = loss_fct(relationship_scores.view(-1, self.num_relations), dp_labels.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, relationship_scores) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertDependencyParsingOutput(
            loss=total_loss,
            head_prediction_logits=prediction_scores,
            dp_prediction_logits=relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


MODEL_FOR_DEPENDENCY_PARSING_MAPPING = OrderedDict(
    [
        ("bert", BertModelForDependencyParsing)
    ]
)


class AutoModelForDependencyParsing:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, max_seq_length: Optional[int] = None,
                        num_relations: Optional[int] = None, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        kwargs["_from_auto"] = True
        hub_kwargs_names = [
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "resume_download",
            "revision",
            "subfolder",
            "use_auth_token",
        ]
        hub_kwargs = {name: kwargs.pop(name) for name in hub_kwargs_names if name in kwargs}
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                trust_remote_code=trust_remote_code,
                **hub_kwargs,
                **kwargs,
            )
        model_class = MODEL_FOR_DEPENDENCY_PARSING_MAPPING.get(config.model_type)
        return model_class.from_pretrained(
            pretrained_model_name_or_path,
            max_seq_length=max_seq_length, num_relations=num_relations, config=config,
            *model_args, **hub_kwargs, **kwargs
        )


if __name__ == "__main__":
    print(AutoModelForDependencyParsing.from_pretrained("klue/bert-base", 10, 20))
