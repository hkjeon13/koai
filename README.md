## KoAI; Korean AI Project. 한국어를 위한 인공지능 프로젝트

---

```
$ pip install koai
```

(develop version)
```
$ pip install git+https://github.com/hkjeon13/koai.git@develop
```


## FineTuning
```
from koai import finetune

finetune(
    "klue-sts", 
    "klue/bert-base", 
    do_train=True, 
    do_eval=True, 
    num_train_epochs=1, 
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_strategy="epoch"
)
```