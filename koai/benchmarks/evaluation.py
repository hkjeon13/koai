import evaluate
import numpy as np


def get_metrics(task_type: str, metric_name: str):
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

    return compute_metrics
