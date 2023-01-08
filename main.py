from koai import finetune
def test_fineune():
    finetune(
        "glue-cola",
        "klue/bert-base",
        do_train=True,
        do_eval=True,
        num_train_epochs=1,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        max_source_length=512,
    )

def main():
    assert test_fineune() == None

if __name__=="__main__":
    main()