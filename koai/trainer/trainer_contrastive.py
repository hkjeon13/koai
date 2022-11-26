from transformers import Trainer
from transformers.trainer import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, unwrap_model
import torch

class ContrastiveBatchTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        print(torch.where(inputs['labels'], ))
        raise ValueError
        outputs = model(**inputs)
        print(torch.argsort(inputs['labels']))
        print(outputs)
        raise ValueError
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
    dataset = load_dataset('klue', 'ynat')
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    model = AutoModelForSequenceClassification.from_pretrained("klue/bert-base", num_labels=7)

    def example_function(examples):
        tokenized_examples = tokenizer(
            examples['title'],
            truncation=True,
            padding="max_length",
            max_length=256
        )
        if "label" in examples:
            tokenized_examples['labels'] = examples['label']
        return tokenized_examples

    dataset = dataset.map(example_function, batched=True)
    args = TrainingArguments("runs/")

    trainer = ContrastiveBatchTrainer(
        model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation']
    )

    trainer.train()
