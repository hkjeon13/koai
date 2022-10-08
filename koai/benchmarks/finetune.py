from collections import OrderedDict
from typing import List, Optional
from datasets import load_dataset
from finetune_utils import get_task_info, get_example_function, TaskInfo, get_model
from transformers import AutoTokenizer, PreTrainedModel

def finetune(
        task_name: str,
        model_name_or_path: str,
        remove_columns: bool = True,
        custom_task_infolist: Optional[List[TaskInfo]] = None,
        max_source_length: int = 512,
        max_target_length: Optional[int] = None,
        save_model: bool = True,
        finetune_model_across_the_tasks: bool = False) -> PreTrainedModel:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    infolist = custom_task_infolist
    if infolist is None:
        infolist += get_task_info(task_name=task_name)

    for info in infolist:
        dataset = load_dataset(*info.task)
        example_function = get_example_function(
            info,
            tokenizer=tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
        )
        _rm_columns = []
        if remove_columns:
            _rm_columns += list(dataset.column_names.values())[0]
        dataset = dataset.map(example_function, batched=True, remove_columns=_rm_columns)


    return None


if __name__ == "__main__":
    finetune("klue-sts", "klue/bert-base")
