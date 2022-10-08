from dataclasses import dataclass
from typing import Tuple, Union, Optional, Callable
from collections import OrderedDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MODEL_CONFIG = OrderedDict([
    ('sequence-classification', AutoModelForSequenceClassification),
    ('token-classification', AutoModelForTokenClassification),
    ('conditional-generation', AutoModelForCausalLM),
    ('question-answering', AutoModelForQuestionAnswering),
    ('masked-language-modeling', AutoModelForMaskedLM),
    ('causal-language-modeling', AutoModelForCausalLM),
    ('conditional-generation', AutoModelForCausalLM),
    ('sequence-to-sequence', AutoModelForSeq2SeqLM)
])

TASK_ATTRS = ["task", "task_type", "text_column", "text_pair_column", "label_column",
              "preprocess_function", "train_split", "eval_split"]


@dataclass
class TaskInfo:
    task: Tuple[str, str]
    task_type: str
    text_column: str
    text_pair_column: str
    label_column: str
    train_split: str = "train"
    eval_split: str = "validation"
    preprocess_function: Optional[Callable] = None
    def from_dict(self, data: dict) -> None:
        return TaskInfo(**{k: data.get(k) for k in TASK_ATTRS})


def klue_sts_function(examples):
    examples['labels'] = [label["binary-label"] for label in examples['labels']]
    return examples


def get_model(model_name_or_path: str, task_type:str) -> PreTrainedModel:
    model = MODEL_CONFIG.get(task_type).from_pretrained(model_name_or_path)
    if model is None:
        raise FileExistsError(f"Can't find any model matching '{model_name_or_path}' on huggingface hub or local directory.")
    return model


def get_task_info(task_name: str):
    return [
        TaskInfo(
            task=("klue", "sts"),
            task_type="sequence-classification",
            text_column="sentence1",
            text_pair_column="sentence2",
            label_column="labels",
            preprocess_function=klue_sts_function,
            train_split="train",
            eval_split="validation",
        )
    ]


def get_example_function(
        info: TaskInfo,
        tokenizer: Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast],
        max_source_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        padding: str = "longest",
        truncation: bool = True
):
    if info.task_type == "sequence-classification":
        def example_function(examples):
            tokenized = tokenizer(
                examples.get(info.text_column),
                text_pair=examples.get(info.text_pair_column),
                max_length=max_source_length,
                truncation=truncation,
                padding=padding
            )
            if info.label_column in examples:
                tokenized['labels'] = [label for label in examples[info.label_column]]
            return tokenized

    return example_function


def get_trainer(task_type: str):
    if task_type == "sequence-to-sequence":
        return Seq2SeqTrainingArguments, Seq2SeqTrainer
    else:
        return TrainingArguments, Trainer

def get_data_collator(task_type:str):
    get_trainer