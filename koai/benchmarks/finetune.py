import json
import os
import re
from inspect import signature
from typing import List, Optional, Union

from datasets import load_dataset, DatasetDict, IterableDatasetDict
from transformers import AutoTokenizer, PreTrainedModel, logging
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from .evaluation import get_metrics
from .finetune_utils import (
    TaskInfo,
    get_task_info,
    get_example_function,
    get_model,
    get_trainer,
    get_data_collator,
    trim_task_name,
)
from .postprocess import get_mrc_post_processing_function

logger = logging.get_logger(__file__)


def load_json(path: str, encoding: str = 'utf-8') -> Union[dict, list]:
    with open(path, 'r', encoding=encoding) as r:
        return json.load(r)


def write_json(path: str, content: Union[dict, list], encoding: str = 'utf-8') -> None:
    with open(path, 'w', encoding=encoding) as w:
        json.dump(content, w)


def write_text(path: str, content: str, encoding: str = 'utf-8') -> None:
    with open(path, 'w', encoding=encoding) as w:
        w.write(content)


_UNUSED = re.compile(r"\[unused[0-9]+\]")


def add_special_tokens_to_unused(
        tokenizer: Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast],
        special_tokens: List[str],
        save_path: str = '.cache/') -> Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast]:
    unused_tokens = sorted([(k, v) for k, v in tokenizer.vocab.items() if _UNUSED.match(k)], key=lambda x: x[1])
    vocab = tokenizer.vocab.copy()
    tokenizer.save_pretrained(save_path)
    vocab_json = load_json(os.path.join(save_path, 'tokenizer.json'))
    if unused_tokens:
        for spt in special_tokens:
            unu, num = unused_tokens.pop(0)
            del vocab[unu]
            del vocab_json["model"]["vocab"][unu]
            vocab[spt] = num
            vocab_json["model"]["vocab"][spt] = num

    ordered_vocab = [k for k, v in sorted(list(vocab.items()), key=lambda x: x[1])]

    write_text(os.path.join(save_path, "vocab.txt"), "\n".join(ordered_vocab))
    write_json(os.path.join(save_path, "tokenizer.json"), vocab_json)

    tokenizer = tokenizer.from_pretrained(save_path)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    tokenizer.save_pretrained(save_path)
    return tokenizer


def get_dataset_columns(dataset: Union[DatasetDict, IterableDatasetDict]):
    columns = []
    if isinstance(dataset, DatasetDict):
        columns += list(dataset.column_names.values())[0]
    elif isinstance(dataset, IterableDatasetDict):
        sample = next(iter(list(dataset.values())[0]))
        columns += list(sample.keys())
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
        train_samples: Optional[int] = None,
        eval_samples: Optional[int] = None,
        finetune_model_across_the_tasks: bool = False,
        add_sp_tokens_to_unused: bool = True,
        *args, **kwargs) -> Optional[PreTrainedModel]:
    """

    Args:
        task_name: Task name on the Huggingface-Hub.
        model_name_or_path: Model name on the Huggingface-Hub.
        remove_columns: Column names to remove.
        custom_task_infolist: A list of the custom tasks was wrapped with the 'TaskInfo' class.
        max_source_length: A maximum length of the input sequences.
        max_target_length: A maximum length of the label sequences (Optional).
        padding: padding strategy for the language tokenizer (familiar with the 'transformers.PretrainedTokenizerBase')
        save_model: Whether the fine-tuned model is saved on the local path (at the end of training/evaluating).
        return_models: Whether the function returns the model or not
        output_dir: Output directory for saving model.
        train_samples: The number of samples to train (Optional).
        eval_samples: The number of samples to evaluate (Optional).
        finetune_model_across_the_tasks: Whether the model train across the tasks or not.
        add_sp_tokens_to_unused: Whether the tokenizer replaces its unused tokens with the special tokens or not (Optional).
        *args:
        **kwargs:
    Returns:
        Pretrained model or None

    >>> from koai import finetune

    >>> finetune(
    ...    task_name="klue-sts",
    ...    model_name_or_path="klue/bert-base",
    ...    do_train=True,
    ...    do_eval=True,
    ...    num_train_epochs=5,
    ...    evaluation_strategy="epoch",
    ...    save_strategy="no",
    ...    logging_strategy="epoch"
    ... )
    """
    infolist = custom_task_infolist
    if infolist is None:
        infolist = get_task_info(task_name=task_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = None
    models_for_return = []
    for info in infolist:
        print("-".join(info.task))
        _path = os.path.join(output_dir, trim_task_name(task_name))
        has_sp_tokens = info.extra_options.get("has_special_tokens")
        if has_sp_tokens:
            if add_sp_tokens_to_unused:
                tokenizer = add_special_tokens_to_unused(
                    tokenizer, info.extra_options["additional_special_tokens"]
                )
            else:
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": info.extra_options["additional_special_tokens"]}
                )

        custom_dataset = {}
        if info.custom_train_dataset is not None:
            custom_dataset[info.train_split] = info.custom_train_dataset

        if info.custom_eval_dataset is not None:
            custom_dataset[info.eval_split] = info.custom_eval_dataset

        if not custom_dataset:
            dataset = load_dataset(*info.task)
        else:
            if len(custom_dataset) == 1:
                candidates = [info.train_split, info.eval_split]
                custom_column = candidates.pop(next(iter(custom_dataset.keys())))
                remain_column = candidates[0]
                dataset = DatasetDict({
                    custom_column: custom_dataset[custom_column],
                    remain_column: load_dataset(*info.task, split=remain_column)
                })
            elif len(custom_dataset) == 2:
                dataset = DatasetDict(custom_dataset)
            else:
                raise ValueError(
                    f"Not Supported Replacement on the split(not '{info.train_split}', '{info.eval_split}')"
                )

        if info.train_split in dataset and train_samples is not None:
            dataset[info.train_split] = dataset[info.train_split].select(range(train_samples))

        if info.eval_split in dataset and eval_samples is not None:
            dataset[info.eval_split] = dataset[info.eval_split].select(range(eval_samples))

        eval_examples = dataset.get(info.eval_split)

        if isinstance(info.preprocess_function, dict):
            preprocess_train = info.preprocess_function.get(info.train_split)
            if preprocess_train is not None:
                dataset[info.train_split] = dataset[info.train_split].map(
                    preprocess_train, batched=True
                )

            preprocess_eval = info.preprocess_function.get(info.eval_split)
            if preprocess_eval is not None:
                dataset[info.eval_split] = dataset[info.eval_split].map(
                    preprocess_eval, batched=True
                )
        else:
            dataset = dataset.map(info.preprocess_function, batched=True)

        example_function = get_example_function(
            info,
            tokenizer=tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
        )

        _rm_columns = get_dataset_columns(dataset)

        if isinstance(example_function, tuple):
            train_function, eval_function = example_function
            if info.train_split in dataset:
                dataset[info.train_split] = dataset[info.train_split].map(
                    train_function,
                    batched=True,
                    remove_columns=dataset[info.train_split].column_names if remove_columns else None
                )

            if info.eval_split in dataset:
                dataset[info.eval_split] = dataset[info.eval_split].map(
                    eval_function,
                    batched=True,
                    remove_columns=dataset[info.eval_split].column_names if remove_columns else None
                )
        else:
            dataset = dataset.map(example_function, batched=True, remove_columns=_rm_columns)

        data_collator = get_data_collator(task_type=info.task_type)

        collator_params = list(signature(data_collator).parameters.keys())
        params = {arg: kwargs[arg] for arg in collator_params if arg in kwargs}
        if "tokenizer" in collator_params:
            params['tokenizer'] = tokenizer

        data_collator = data_collator(**params)
        if finetune_model_across_the_tasks and model is not None:
            model_name_or_path = _path

        model = get_model(model_name_or_path, info, max_source_length)
        if has_sp_tokens:
            model.resize_token_embeddings(len(tokenizer))

        compute_metrics = get_metrics(
            task_type=info.task_type,
            id2label=model.config.id2label,
            metric_name=info.metric_name,
            tokenizer=tokenizer
        )

        traininig_args, trainer = get_trainer(info.task_type)

        traininig_args_params = list(signature(traininig_args).parameters.keys())
        traininig_args_params = {arg: kwargs[arg] for arg in traininig_args_params if arg in kwargs}

        if "optim" not in traininig_args_params:
            traininig_args_params["optim"] = "adamw_torch"

        train_dataset, eval_dataset = dataset.get(info.train_split), dataset.get(info.eval_split)

        traininig_args = traininig_args(
            output_dir=output_dir,
            label_names=["head_labels", "dp_labels"] if info.task_type == "dependency-parsing" else None,
            **traininig_args_params,
        )

        params = list(signature(trainer.__init__).parameters.keys())
        other_params = {}
        if "post_process_function" in params and info.task_type == "question-answering":
            other_params["post_process_function"] = get_mrc_post_processing_function(info, output_dir=output_dir)
            other_params["eval_examples"] = eval_examples if kwargs.get("do_eval") else None

        trainer = trainer(
            model=model,
            args=traininig_args,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **other_params
        )

        if kwargs.get("do_train", False):
            trainer.train()

        elif kwargs.get("do_eval"):
            eval_result = trainer.evaluate()
            print(eval_result)

        if save_model or finetune_model_across_the_tasks:
            trainer.save_model(output_dir=_path)

        if return_models:
            models_for_return.append(trainer.model)

    if return_models:
        return models_for_return

    return None


if __name__ == "__main__":
    finetune("klue-re", "klue/bert-base", do_train=True, do_eval=True)
