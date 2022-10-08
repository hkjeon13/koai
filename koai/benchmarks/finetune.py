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


def finetune(task_name: str, model_name_or_path: str, remove_columns: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    infolist = get_task_info(task_name=task_name)
    for info in infolist:
        print(info)
        dataset = load_dataset(*info.task)
        example_function = get_example_function(info, tokenizer)
        _rm_columns = []
        if remove_columns:
            _rm_columns+=list(dataset.column_names.values())[0]
        dataset = dataset.map(example_function, batched=True, remove_columns=_rm_columns)
        print(dataset)

    return None


if __name__ == "__main__":
    finetune("klue-sts", "klue/bert-base")
