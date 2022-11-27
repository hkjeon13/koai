from koai import finetune

finetune(
    "glue-stsb",
    "klue/bert-base",
    do_train=True,
    do_eval=True,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_strategy="epoch",
    max_source_length=512,
)