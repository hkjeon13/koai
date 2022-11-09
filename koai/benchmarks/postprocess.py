from finetune_utils import TaskInfo
from transformers import EvalPrediction
from utils_qa import postprocess_qa_predictions


def mrc_post_processing_function(examples, features, predictions, info: TaskInfo,
                             log_level: str = "passive",
                             output_dir: str = "runs/", stage: str = "eval/"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    options = info.extra_options
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=options.get("version_2_with_negative"),
        n_best_size=options.get("n_best_size"),
        max_answer_length=options.get("max_answer_length"),
        null_score_diff_threshold=options.get("null_score_diff_threshold"),
        output_dir=output_dir,
        log_level=log_level,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    if options.version_2_with_negative:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    references = [{"id": ex["id"], "answers": ex[info.label_column]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)
