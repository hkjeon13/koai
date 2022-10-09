import evaluate
import numpy as np
from typing import Optional, Callable
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast
import nltk
nltk.download("punkt")
from nltk import sent_tokenize


def postprocess_text(preds, labels, metric='rouge'):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    if metric == 'rouge':
        preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(sent_tokenize(label)) for label in labels]
    elif metric == 'bleu':
        labels = [[label] for label in labels]
    return preds, labels

def get_metrics(task_type: str, metric_name: str,
                tokenizer: Optional[PreTrainedTokenizerBase or PreTrainedTokenizerFast]) -> Callable:
    if task_type == "token_classification":
        _metric = evaluate.load(metric_name)

        def compute_metrics(p):
            preds, labels = p
            preds = np.argmax(preds, axis=-1)
            true_predictions = [
                [p for (p, l) in zip(pred, label) if l != -100]
                for pred, label in zip(preds, labels)
            ]

            true_labels = [
                [l for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(preds, labels)
            ]

            results = _metric.compute(predictions=true_predictions, references=true_labels)
            if metric_name == "seqeval":
                results = {
                    "precision": results['overall_precision'],
                    "recall": results['overall_recall'],
                    "f1": results['overall_f1'],
                    "accuracy": results['overall_accuracy'],
                }
            return results

    elif task_type == "sequence-classification":
        _metric = evaluate.load(metric_name)

        def compute_metrics(p):
            preds, labels = p
            preds = np.argmax(preds, axis=-1)
            if metric_name == "f1":
                results = _metric.compute(predictions=preds, references=labels, average='macro')
            else:
                results = _metric.compute(predictions=preds, references=labels)
            return results
    elif task_type in ("conditional-generation", "sequence-to-sequence"):
        def compute_metrics(p):
            preds, labels = p
            preds = preds[0] if isinstance(preds, tuple) else preds
            decoded_preds = tokenizer.batch_decode(np.argmax(preds, axis=-1), skip_special_tokens=True)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels, metric_name)
            result = _metric.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                use_stemmer=True if metric_name == 'rouge' else False
            )
            result = {key: value.mid.fmeasure * 100
                      for key, value in result.items()} \
                if metric_name == 'rouge' else result

            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
            return result

    return compute_metrics
