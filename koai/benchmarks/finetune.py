from collections import OrderedDict
from inspect import signature
from typing import List, Optional
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedModel
from finetune_utils import (
    TaskInfo,
    get_task_info,
    get_example_function,
    get_model,
    get_trainer,
    get_data_collator,
    trim_task_name,
)

import os

def get_dataset_columns(dataset:DatasetDict):
    columns = []
    columns += list(dataset.column_names.values())[0]
    return columns

def finetune(
        task_name: str,
        model_name_or_path: str,
        remove_columns: bool = True,
        custom_task_infolist: Optional[List[TaskInfo]] = None,
        max_source_length: int = 512,
        max_target_length: Optional[int] = None,
        padding: str = "longest",
        save_model: bool = False,
        return_models: bool = False,
        output_dir: str = "runs/",
        finetune_model_across_the_tasks: bool = False,
        *args, **kwargs) -> PreTrainedModel:

    infolist = custom_task_infolist
    if infolist is None:
        infolist = get_task_info(task_name=task_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    models_for_return = []
    for info in infolist:
        dataset = load_dataset(*info.task)
        dataset = dataset.map(info.preprocess_function, batched=True)
        example_function = get_example_function(
            info,
            tokenizer=tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
        )

        _rm_columns = get_dataset_columns(dataset)
        dataset = dataset.map(example_function, batched=True, remove_columns=_rm_columns)

        data_collator = get_data_collator(task_type=info.task_type)

        collator_params = list(signature(data_collator).parameters.keys())
        params = {arg: kwargs[arg] for arg in collator_params if arg in kwargs}
        if "tokenizer" in collator_params:
            params['tokenizer'] = tokenizer

        data_collator = data_collator(**params)
        model = get_model(model_name_or_path, info)

        traininig_args, trainer = get_trainer(info.task_type)

        traininig_args_params = list(signature(traininig_args).parameters.keys())
        traininig_args_params = {arg: kwargs[arg] for arg in traininig_args_params if arg in kwargs}

        traininig_args = traininig_args(
            output_dir=output_dir,
            **traininig_args_params,
        )
        print(next(iter(dataset['train'])))
        trainer = trainer(
            model=model,
            args=traininig_args,
            data_collator=data_collator,
            train_dataset=dataset.get(info.train_split),
            eval_dataset=dataset.get(info.eval_split),
        )

        if kwargs.get("do_train", False):
            trainer.train()

        elif kwargs.get("do_eval"):
            trainer.evaluate()

        if save_model:
            _path = os.path.join(output_dir, trim_task_name(task_name))
            trainer.save_model(output_dir=_path)

        if return_models:
            models_for_return.append(trainer.model)

    if return_models:
        return models_for_return

    return None


if __name__ == "__main__":
    finetune("klue-ner", "klue/bert-base", do_train=True)
