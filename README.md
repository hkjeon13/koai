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

허깅페이스 허브(huggingface-hub) 또는 허깅페이스 라이브러리를 통해 로드 가능한 로컬 파일을, klue 벤치 마크에 대하여 테스트하는 예시입니다.

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
