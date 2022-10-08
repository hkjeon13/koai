from collections import OrderedDict
from datasets import load_dataset
from finetune_utils import get_task_info, get_example_function
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

MODEL_CONFIG = OrderedDict([
    ('sequence-classification', AutoModelForSequenceClassification),
    ('token-classification', AutoModelForTokenClassification),
])


def finetune(task_name: str, model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    infolist = get_task_info(task_name=task_name)
    for info in infolist:
        dataset = load_dataset(*info.task)
        example_function = get_example_function(info.task_type, tokenizer)

    return None


if __name__ == "__main__":
    finetune("klue-sts")
