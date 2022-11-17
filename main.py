from koai import finetune

finetune(
    "klue-dp",
    "ainize/klue-bert-base-mrc",
    do_train=False,
    do_eval=True,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_strategy="epoch",
    max_source_length=512,
)