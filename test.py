from koai import finetune
from transformers import PreTrainedModel

def main():

    model = finetune(
        "glue-cola",
        "klue/bert-base",
        do_train=False,
        do_eval=False,
        num_train_epochs=1,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        max_source_length=512,
        return_models=True
    )
    assert isinstance(model[0], PreTrainedModel)


if __name__ == "__main__":
    main()