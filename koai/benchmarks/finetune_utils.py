from dataclasses import dataclass
from typing import Tuple, Union, Optional
from collections import OrderedDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast
)

MODEL_CONFIG = OrderedDict([
    ('sequence-classification', AutoModelForSequenceClassification),
    ('token-classification', AutoModelForTokenClassification),
])


@dataclass
class TaskInfo:
    task: Tuple[str, str]
    task_type: str


def get_task_info(task_name: str):
    return [
        TaskInfo(task=("klue", "sts"), task_type="sequence-classification")
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
                text_pair=examples.get(info.text_pair),
                max_length=max_source_length,
                truncation=truncation,
                padding=padding
            )
            return examples
