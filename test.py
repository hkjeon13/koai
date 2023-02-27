from transformers import PreTrainedModel

from koai import finetune
from koai.models import AutoModelForBiEncoder


def test_finetune():
    return finetune(
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
    )[0]


def test_biencoder():
    model = AutoModelForBiEncoder.from_pretrained("klue/bert-base")

def main():
    # assert isinstance(test_finetune(), PreTrainedModel)
    test_biencoder()


if __name__ == "__main__":
    main()
