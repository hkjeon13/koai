import os
import re
import json
from dataclasses import dataclass, field
from typing import Tuple, Union, Optional, Callable
from .evaluation import get_metrics
from .preprocess import *
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
    DataCollatorWithPadding,
    DataCollatorForSOP,
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
    DataCollatorForSeq2Seq,
    DataCollatorForPermutationLanguageModeling,
    DataCollatorForWholeWordMask,
)

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
    ('token-classification', AutoModelForTokenClassification),
    ('conditional-generation', AutoModelForCausalLM),
    ('question-answering', AutoModelForQuestionAnswering),
    ('masked-language-modeling', AutoModelForMaskedLM),
    ('causal-language-modeling', AutoModelForCausalLM),
    ('sequence-to-sequence', AutoModelForSeq2SeqLM)
])


TASK_ATTRS = ["task", "task_type", "text_column", "text_pair_column", "label_column", "metric_name", "extra_options",
              "preprocess_function", "train_split", "eval_split", "num_labels", "is_split_into_words"]


_task_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks.json")
with open(_task_path, "r", encoding='utf-8') as f:
    TASKS = json.load(f)

PROCESS_FUNCTIONS_MAP = OrderedDict([
    ("klue-sts", klue_sts_preprocess_function),
    ("klue-re", klue_re_preprocess_function)
])

TASKS = {k: dict(v, **{"preprocess_function": PROCESS_FUNCTIONS_MAP.get(k)}) for k, v in TASKS.items()}


@dataclass
class TaskInfo:
    task: Tuple[str, str]
    task_type: str
    text_column: str
    label_column: str
    num_labels: int = 2
    text_pair_column: Optional[str] = None
    train_split: str = "train"
    eval_split: str = "validation"
    metric_name: Optional[str] = None
    extra_options: dict = field(default_factory=dict)
    is_split_into_words: bool = False
    preprocess_function: Optional[Callable] = None

    @classmethod
    def from_dict(cls, data: dict) -> None:
        info = {k: data.get(k) for k in TASK_ATTRS}
        info = {k: v for k, v in info.items() if v}
        info['task'] = tuple(info['task'].split("-"))
        return TaskInfo(**info)


def get_model(model_name_or_path: str, info: TaskInfo) -> PreTrainedModel:
    _model = MODEL_CONFIG.get(info.task_type).from_pretrained(model_name_or_path, num_labels=info.num_labels)
    if _model is None:
        raise FileExistsError(f"Can't find any model matching '{model_name_or_path}' on huggingface hub or local directory.")
    label_names = info.extra_options.get("label_names")
    if info.task_type == "token-classification" and label_names:
        _model.config.id2label = {i: l for i, l in enumerate(label_names)}
        _model.config.label2id = {v: k for k, v in _model.config.id2label.items()}
    return _model


def get_task_info(task_name: str):
    tasks = TASKS.get(task_name, [])
    if not tasks:
        tasks = [TASKS.get(key) for key in TASKS.keys() if key.split("-")[0] == task_name]

    if not isinstance(tasks, list):
        tasks = [tasks]
    output = [TaskInfo.from_dict(t) for t in tasks]
    return output


def get_example_function(
        info: TaskInfo,
        tokenizer: Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast],
        max_source_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        padding: str = "longest",
        truncation: bool = True
):
    if info.task_type == "token-classification":
        extra_options = info.extra_options
        label_all_tokens = extra_options.get("label_all_tokens", False)
        b_to_i_label = extra_options.get("extra_options", [])
        def example_function(examples):
            tokenized_inputs = tokenizer(
                examples.get(info.text_column),
                text_pair=examples.get(info.text_pair_column),
                max_length=max_source_length,
                truncation=truncation,
                padding=padding,
                is_split_into_words=info.is_split_into_words,
            )
            labels = []
            for i, label in enumerate(examples["ner_tags"]):
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
    else:
        def example_function(examples):
            tokenized_inputs = tokenizer(
                examples.get(info.text_column),
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
    else:
        return TrainingArguments, Trainer


def get_data_collator(task_type: str):
    return DATA_COLLATOR.get(task_type, DataCollatorWithPadding)


def trim_task_name(name:str):
    name = name.replace(" ", "_").replace(".", "_")
    name = re.sub("[^a-zA-Z가-힣0-9\-_]+", "", name)
    return name


if __name__ == "__main__":
    print(PROCESS_FUNCTIONS_MAP.get("klue-sts", default_preprocess_function))