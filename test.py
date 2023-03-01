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
    from transformers import AutoTokenizer
    _NAME = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(_NAME)
    model = AutoModelForBiEncoder.from_pretrained(_NAME)
    model.eval()
    #input_text =

def main():
    # assert isinstance(test_finetune(), PreTrainedModel)
    #test_biencoder()
    import torch
    a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(torch.diag(a))


if __name__ == "__main__":
    main()
