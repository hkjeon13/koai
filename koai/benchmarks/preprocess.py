from collections import OrderedDict


def default_preprocess_function(examples):
    return examples


def klue_sts_preprocess_function(examples):
    examples['labels'] = [label["binary-label"] for label in examples['labels']]
    return examples
