from dataclasses import dataclass
from typing import Tuple, Union, Optional
from collections import OrderedDict
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
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
                examples[info.text_column],
                max_length=max_source_length,
                truncation=truncation,
                padding=padding
            )
            return examples


def finetune(task_name: str, model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    infolist = get_task_info(task_name=task_name)
    for info in infolist:
        dataset = load_dataset(*info.task)
        get_example_function(info.task_type, tokenizer)
    return None


if __name__ == "__main__":
    finetune("klue-sts")
