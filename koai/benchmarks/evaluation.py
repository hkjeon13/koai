from typing import Optional, Callable, Dict
import evaluate
import nltk
import numpy as np
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

def postprocess_text(preds, labels, metric='rouge'):
    nltk.download("punkt")
    from nltk import sent_tokenize
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    if metric == 'rouge':
        preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(sent_tokenize(label)) for label in labels]
    elif metric == 'bleu':
        labels = [[label] for label in labels]
    return preds, labels


def get_metrics(
        task_type: str,
        metric_name: str,
        tokenizer: Optional[PreTrainedTokenizerBase or PreTrainedTokenizerFast],
        id2label: Optional[Dict[int, str]] = None) -> Callable:
    _metric = evaluate.load(*metric_name.split("-"))
    if task_type == "token-classification":
        def compute_metrics(p):
            preds, labels = p
            preds = np.argmax(preds, axis=-1)
            true_predictions = [
                [id2label[p] for (p, l) in zip(pred, label) if l != -100]
                for pred, label in zip(preds, labels)
            ]

            true_labels = [
                [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
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

    elif task_type in ("sequence-classification",):
        def compute_metrics(p):
            preds, labels = p
            _, dim = preds.shape
            if dim == 1:
                preds = np.squeeze(preds)
            else:
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

    elif task_type == "dependency-parsing":
        def compute_metrics(p):
            (preds_head, preds_dp), (labels_head, labels_dp) = p
            preds_head = np.argmax(preds_head, axis=-1)
            preds_dp = np.argmax(preds_dp, axis=-1)

            score_head = [
                [p == l for (p, l) in zip(pred, label) if l != -100]
                for pred, label in zip(preds_head, labels_head)
            ]

            score_dp = [
                [p == l for (p, l) in zip(pred, label) if l != -100]
                for pred, label in zip(preds_dp, labels_dp)
            ]

            uas = [
                sum(seq) / len(seq) for seq in score_head
            ]

            las = [
                sum([h and d for h, d in zip(head, dp)]) / len(head)
                for head, dp in zip(score_head, score_dp)
            ]

            return {
                "UAS": sum(uas) / len(uas),
                "LAS": sum(las) / len(las),
            }
    elif task_type == "question-answering":
        def compute_metrics(p):
            return _metric.compute(predictions=p.predictions, references=p.label_ids)

    return compute_metrics
