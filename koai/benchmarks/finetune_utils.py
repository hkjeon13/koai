import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from inspect import signature
from typing import Tuple, Union, Optional, Callable, Dict, List

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForMultipleChoice,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForSOP,
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
    DataCollatorForSeq2Seq,
    DataCollatorForPermutationLanguageModeling,
    DataCollatorForWholeWordMask,
)

from .modeling_dp import AutoModelForDependencyParsing
from .postprocess import *
from .preprocess import *
from .trainer_qa import QuestionAnsweringTrainer

DATA_COLLATOR = OrderedDict([
    ("sop", DataCollatorForSOP),
    ("language-modeling", DataCollatorForLanguageModeling),
    ("token-classification", DataCollatorForTokenClassification),
    ("sequence-to-sequence", DataCollatorForSeq2Seq),
    ("permutation-language-modeling", DataCollatorForPermutationLanguageModeling),
    ("whole-word-mask", DataCollatorForWholeWordMask),
])

MODEL_CONFIG = OrderedDict([
    ('sequence-classification', AutoModelForSequenceClassification),
    ('multiple-choice', AutoModelForMultipleChoice),
    ('token-classification', AutoModelForTokenClassification),
    ('conditional-generation', AutoModelForCausalLM),
    ('question-answering', AutoModelForQuestionAnswering),
    ('masked-language-modeling', AutoModelForMaskedLM),
    ('causal-language-modeling', AutoModelForCausalLM),
    ('sequence-to-sequence', AutoModelForSeq2SeqLM),
    ('dependency-parsing', AutoModelForDependencyParsing),

])

TASK_ATTRS = [
    "task", "task_type", "text_column", "text_pair_column", "label_column", "metric_name", "extra_options",
    "preprocess_function", "train_split", "eval_split", "num_labels", "is_split_into_words", "id_column",
    "postprocess_function", "custom_train_dataset", "custom_eval_dataset"
]

_task_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks.json")
with open(_task_path, "r", encoding='utf-8') as f:
    TASKS = json.load(f)

PREPROCESS_FUNCTIONS_MAP = OrderedDict([
    ("klue-sts", klue_sts_preprocess_function),
    ("klue-re", klue_re_preprocess_function),
    ("klue-mrc", {"train": default_preprocess_function, "validation": None})
])

POSTPROCESS_FUNCTIONS_MAP = OrderedDict([
    ("klue-mrc", get_mrc_post_processing_function),
])

TASKS = {k: dict(v, **{"preprocess_function": PREPROCESS_FUNCTIONS_MAP.get(k, default_preprocess_function),
                       "postprocess_function": POSTPROCESS_FUNCTIONS_MAP.get(k, default_preprocess_function)})
         for k, v in TASKS.items()}


@dataclass
class TaskInfo:
    """
    A Wrapper Class for the Benchmark(or Custom) Task

    Args:
        task: Task name on the Huggingface-Hub or Custom name.
        task_type: Type of task
            support tasks: [
                'sequence-classification', 'multiple-choice', 'token-classification', 'conditional-generation',
                'question-answering', 'masked-language-modeling', 'causal-language-modeling', 'sequence-to-sequence',
                'dependency-parsing'
            ]
        text_column: A column name for the text; text means the (first) input sequence.
        label_column: A column name for the label
        num_labels: The number of classification labels(if you do the non-sequence classification task(e.g., Text Generation), don't need to be filled).
        id_column: A column name for the id (it is used for the evaluation with Question Answering Model(except Generation Model)).
        text_pair_column: A column name for the second text column name. It is useful when the task has distinguishable Token ID(e.g. sts; Input = SeqA + [SEP] + SeqB).
        train_split: A split name of the Dataset to train(under the 'datasets.DatasetDict' structure).
        eval_split: A split name of the Dataset to evaluate(under the 'datasets.DatasetDict' structure).
        custom_train_dataset: custom dataset to train (Optional, 'transformers.Dataset').
        custom_eval_dataset: custom dataset to evaluate (Optional, 'transformers.Dataset').
        metric_name: A metric name for the evaluation(registered in the Huggingface-Hub).
        extra_options: A Extra-Options to apply(It should be more sophisticated in the future version).
        is_split_into_words: Whether the tokenizer tokenize a pre-tokenized sequences or not(e.g. if 'true', the type of sequence is 'List[List[str]]').
        preprocess_function: pre-process functions for the dataset.
        postprocess_function: post-process functions for the dataset.
    """
    task: Tuple[str, str]
    task_type: str
    text_column: str
    label_column: Union[str, Dict[str, str]]
    num_labels: int = 2
    id_column: Optional[str] = None
    text_pair_column: Optional[str] = None
    train_split: str = "train"
    eval_split: str = "validation"
    custom_train_dataset: Optional[Dataset] = None
    custom_eval_dataset: Optional[Dataset] = None
    metric_name: Optional[str] = None
    extra_options: dict = field(default_factory=dict)
    is_split_into_words: bool = False
    preprocess_function: Optional[Callable] = None
    postprocess_function: Optional[Callable] = None

    @classmethod
    def from_dict(cls, data: dict) -> None:
        info = {k: data.get(k) for k in TASK_ATTRS}
        info = {k: v for k, v in info.items() if v}
        info['task'] = tuple(info['task'].split("-"))
        return TaskInfo(**info)


def get_model(model_name_or_path: str, info: TaskInfo, max_seq_length: int) -> PreTrainedModel:
    _model = MODEL_CONFIG.get(info.task_type)
    if _model is None:
        raise ValueError(
            f"Model type '{info.task_type}' is not defined! The model type should be in {list(MODEL_CONFIG.keys())}")
    _params = list(signature(_model.from_pretrained).parameters.keys())
    params = {k: v for k, v in info.extra_options.items() if k in _params}

    if "max_seq_length" in _params:
        params["max_seq_length"] = max_seq_length
    if "num_relations" in _params:
        params["num_relations"] = info.num_labels

    _model = _model.from_pretrained(
        model_name_or_path,
        num_labels=info.num_labels,
        **params
    )

    if _model is None:
        raise FileExistsError(
            f"Can't find any model matching '{model_name_or_path}' on huggingface hub or local directory."
        )

    label_names = info.extra_options.get("label_names")
    if info.task_type in ("token-classification", "dependency-parsing") and label_names:
        _model.config.id2label = {i: l for i, l in enumerate(label_names)}
        _model.config.label2id = {v: k for k, v in _model.config.id2label.items()}

    return _model


def get_task_info(task_name: str) -> List[TaskInfo]:
    tasks = TASKS.get(task_name, [])
    if not tasks:
        tasks = [TASKS.get(key) for key in TASKS.keys() if key.split("-")[0] == task_name]

    if not isinstance(tasks, list):
        tasks = [tasks]

    return [TaskInfo.from_dict(t) for t in tasks]


def get_example_function(
        info: TaskInfo,
        tokenizer: Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast],
        max_source_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        padding: str = "longest",
        truncation: bool = True
):
    extra_options = info.extra_options
    label_all_tokens = extra_options.get("label_all_tokens", False)
    b_to_i_label = extra_options.get("extra_options", [])
    label2id = {name: i for i, name in enumerate(extra_options.get("label_names", []))}
    prefix = extra_options.get("prefix", "")
    prefix = [prefix] if info.is_split_into_words else prefix

    if info.task_type == "token-classification":
        def example_function(examples):
            tokenized_inputs = tokenizer(
                [prefix + t for t in examples.get(info.text_column)],
                text_pair=examples.get(info.text_pair_column),
                max_length=max_source_length,
                truncation=truncation,
                padding=padding,
                is_split_into_words=info.is_split_into_words,
            )
            labels = []
            for i, label in enumerate(examples[info.label_column]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    else:
                        if label_all_tokens:
                            label_ids.append(b_to_i_label[label[word_idx]])
                        else:
                            label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

    elif info.task_type == "dependency-parsing":
        if not isinstance(info.label_column, dict):
            raise ValueError(
                "For Dependency Parsing, 'label_colmn' should be constructed as {'head':<head_name>, 'dependency':<dependency_relations>}")

        deprels, heads = info.label_column["dependency"], info.label_column["head"]

        def example_function(examples):
            tokenized_inputs = tokenizer(
                [prefix + t for t in examples[info.text_column]],
                padding="max_length",
                truncation=True,
                max_length=max_source_length,
                is_split_into_words=True,
            )

            labels_head, labels_dp = [], []
            for i, (label_dp, label_head) in enumerate(zip(examples[deprels], examples[heads])):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                _label_dp, _label_head = [], []
                for word_idx in word_ids:
                    if word_idx is None:
                        _label_dp.append(-100)
                        _label_head.append(-100)
                    elif word_idx != previous_word_idx:
                        _label_dp.append(label2id[label_dp[word_idx]])
                        _label_head.append(label_head[word_idx])
                    else:
                        _label_dp.append(-100)
                        _label_head.append(-100)
                    previous_word_idx = word_idx
                labels_head.append(_label_head)
                labels_dp.append(_label_dp)
            tokenized_inputs["head_labels"] = labels_head
            tokenized_inputs["dp_labels"] = labels_dp
            return tokenized_inputs

    elif info.task_type == "question-answering":
        pad_on_right = tokenizer.padding_side == "right"
        doc_stride = info.extra_options.get("doc_stride", 0)

        def train_example_function(examples):
            tokenized_inputs = tokenizer(
                examples[info.text_pair_column if pad_on_right else info.text_column],
                examples[info.text_column if pad_on_right else info.text_pair_column],
                truncation=True,
                stride=doc_stride,
                max_length=max_source_length,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length"
            )

            sample_mapping = tokenized_inputs.pop("overflow_to_sample_mapping")
            tokenized_inputs["start_positions"] = []
            tokenized_inputs["end_positions"] = []

            for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
                input_ids = tokenized_inputs["input_ids"][i]
                cls_index = input_ids.index(tokenizer.cls_token_id)
                sequence_ids = tokenized_inputs.sequence_ids(i)
                sample_index = sample_mapping[i]
                answers = examples[info.label_column][sample_index]

                if len(answers["answer_start"]) == 0:
                    tokenized_inputs["start_positions"].append(cls_index)
                    tokenized_inputs["end_positions"].append(cls_index)
                else:
                    # Start/end character index of the answer in the text.
                    start_char = answers["answer_start"][0]
                    end_char = start_char + len(answers["text"][0])

                    # Start token index of the current span in the text.
                    token_start_index = 0
                    while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                        token_start_index += 1

                    # End token index of the current span in the text.
                    token_end_index = len(input_ids) - 1
                    while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                        token_end_index -= 1

                    if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                        tokenized_inputs["start_positions"].append(cls_index)
                        tokenized_inputs["end_positions"].append(cls_index)
                    else:
                        # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                        # Note: we could go after the last offset if the answer is the last word (edge case).
                        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                            token_start_index += 1
                        tokenized_inputs["start_positions"].append(token_start_index - 1)

                        while offsets[token_end_index][1] >= end_char:
                            token_end_index -= 1
                        tokenized_inputs["end_positions"].append(token_end_index + 1)

            tokenized_inputs["example_id"] = []
            for i in range(len(tokenized_inputs["input_ids"])):
                sequence_ids = tokenized_inputs.sequence_ids(i)
                context_index = 0 if pad_on_right else 1
                sample_index = sample_mapping[i]
                tokenized_inputs["example_id"].append(
                    examples[info.id_column][sample_index]
                )
                tokenized_inputs["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_inputs["offset_mapping"][i])
                ]
            return tokenized_inputs

        def eval_example_function(examples):
            tokenized_inputs = tokenizer(
                [prefix + t for t in examples[info.text_pair_column if pad_on_right else info.text_column]],
                examples[info.text_column if pad_on_right else info.text_pair_column],
                truncation=True,
                stride=doc_stride,
                max_length=max_source_length,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length"
            )

            sample_mapping = tokenized_inputs.pop("overflow_to_sample_mapping")
            tokenized_inputs["example_id"] = []

            for i in range(len(tokenized_inputs["input_ids"])):
                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_inputs.sequence_ids(i)
                context_index = 1 if pad_on_right else 0

                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                tokenized_inputs["example_id"].append(examples[info.id_column][sample_index])

                # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
                # position is part of the context or not.
                tokenized_inputs["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_inputs["offset_mapping"][i])
                ]

            return tokenized_inputs

        return train_example_function, eval_example_function

    elif info.task_type == "sequence-to-sequence":
        def example_function(examples):
            tokenized_inputs = tokenizer(
                [prefix + t for t in examples.get(info.text_column)],
                text_pair=examples.get(info.text_pair_column),
                max_length=max_source_length,
                truncation=truncation,
                padding=padding,
                is_split_into_words=info.is_split_into_words
            )

            if info.label_column in examples:
                tokenized_labels = tokenizer(
                    text=[example for example in examples[info.label_column]],
                    padding=True,
                    max_length=max_source_length,
                    truncation=True
                )

                tokenized_inputs['labels'] = [
                    [l if l != tokenizer.pad_token_id else -100 for l in label]
                    for label in tokenized_labels['input_ids']
                ]

            return tokenized_inputs


    else:
        def example_function(examples):
            tokenized_inputs = tokenizer(
                [prefix + t for t in examples.get(info.text_column)],
                text_pair=examples.get(info.text_pair_column),
                max_length=max_source_length,
                truncation=truncation,
                padding=padding,
                is_split_into_words=info.is_split_into_words
            )

            if info.label_column in examples:
                tokenized_inputs['labels'] = [label for label in examples[info.label_column]]
            return tokenized_inputs
    return example_function


def get_trainer(task_type: str):
    if task_type == "sequence-to-sequence":
        return Seq2SeqTrainingArguments, Seq2SeqTrainer
    elif task_type == "question-answering":
        return TrainingArguments, QuestionAnsweringTrainer
    else:
        return TrainingArguments, Trainer


def get_data_collator(task_type: str):
    return DATA_COLLATOR.get(task_type, DataCollatorWithPadding)


def trim_task_name(name: str):
    name = name.replace(" ", "_").replace(".", "_")
    name = re.sub("[^a-zA-Z가-힣0-9\-_]+", "", name)
    return name


if __name__ == "__main__":
    print(PREPROCESS_FUNCTIONS_MAP.get("klue-sts", default_preprocess_function))
