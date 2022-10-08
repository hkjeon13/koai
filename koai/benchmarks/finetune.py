from collections import OrderedDict
from inspect import signature
from typing import List, Optional
from datasets import load_dataset
from finetune_utils import get_task_info, get_example_function, TaskInfo, get_model, get_trainer, get_data_collator, trim_task_name
from transformers import AutoTokenizer, PreTrainedModel
import os

def finetune(
        task_name: str,
        model_name_or_path: str,
        remove_columns: bool = True,
        custom_task_infolist: Optional[List[TaskInfo]] = None,
        max_source_length: int = 512,
        max_target_length: Optional[int] = None,
        padding: str = "longest",
        save_model: bool = False,
        output_dir: str = "runs/",
        finetune_model_across_the_tasks: bool = False,
        *args, **kwargs) -> PreTrainedModel:

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    infolist = custom_task_infolist
    if infolist is None:
        infolist = get_task_info(task_name=task_name)

    for info in infolist:
        dataset = load_dataset(*info.task)
        example_function = get_example_function(
            info,
            tokenizer=tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
        )
        _rm_columns = []
        if remove_columns:
            _rm_columns += list(dataset.column_names.values())[0]

        dataset = dataset.map(example_function, batched=True, remove_columns=_rm_columns)

        model = get_model(model_name_or_path, info.task_type)
        traininig_args, trainer = get_trainer(info.task_type)
        traininig_args_params = list(signature(traininig_args).parameters.keys())
        traininig_args_params = {arg: kwargs[args] for arg in traininig_args_params if arg in kwargs}

        traininig_args = traininig_args(
            output_dir=output_dir,
            **traininig_args_params,
        )

        trainer = trainer(
            model=model
        )
        if save_model:
            _path = os.path.join(output_dir, trim_task_name(task_name))
            trainer.save_model(output_dir=_path)

        print(traininig_args_params)


    return None


if __name__ == "__main__":

    finetune("klue-sts", "klue/bert-base")
