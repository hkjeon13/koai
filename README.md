## KoAI; Korean AI Project. 한국어를 위한 인공지능 프로젝트

```
$ pip install koai
```

## FineTuning

허깅페이스 허브(huggingface-hub) 또는 허깅페이스 라이브러리를 통해 로드 가능한 로컬 파일을, klue 벤치 마크에 대하여 테스트하는 예시입니다.

```
from koai import finetune

# finetuning and evaluating on klue-sts dataset
finetune(
    task_name="klue-sts", 
    model_name_or_path="klue/bert-base", 
    do_train=True, 
    do_eval=True, 
    num_train_epochs=5, 
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_strategy="epoch"
)

# finetuning and evaluating on all klue dataset (except 'wos')
# if "finetune_model_across_the_tasks" is True, the model train all the tasks in KLUE
# but it is false(default is false), finetuning the language model individually.  
finetune(
    "klue", 
    "klue/bert-base", 
    do_train=True, 
    do_eval=True, 
    num_train_epochs=5, 
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_strategy="epoch"
)
```

- task_name: str, 과제의 이름을 설정합니다(klue 전체를 테스트 하려면, "klue". 특정 테스크를 선택하려면 "klue-mrc"와 같이 입력해주세요. "mrc"와 같은 하위 테스크 이름은
  허깅페이스 허브를 따릅니다.)

- model_name_or_path: str, 모델의 허깅페이스 허브 이름 또는 로컬 경로를 입력해 주세요.

- remove_columns: bool = True, 데이터 로드 후 모델 입력을 위한 프로세스 완료 후 기존 컬럼 이름을 삭제할지 여부를 설정합니다.

- custom_task_infolist: Optional[List[TaskInfo]] = None, 직접 TaskInfo 클래스를 설정하고 이를 리스트 안에 넣어 벤치마크 테스트를 할 수 있습니다.

- max_source_length: int = 512, 입력 텍스트의 최대 길이를 설정합니다.

- max_target_length: Optional[int] = None, (만약 있다면) 출력 텍스트의 최대 길이를 설정합니다.

- padding: str = "longest", padding의 방법을 설정합니다(`transformers.PretrainedTokenizerBase.__call__`의 'padding'인자와 동일합니다).

- save_model: bool = False, 모델을 내부에 저장할 지 여부를 설정합니다.

- return_models: bool = False, 함수가 학습된 모델을 반환할지 여부를 설정합니다.

- output_dir: str = "runs/", (save_model=True일 때), 저장할 디렉토리를 설정합니다.

- finetune_model_across_the_tasks: bool = False, 모델을 입력 받은 여러 벤치마크에 대해서 조정 학습 시, 초기화 할지를 설정합니다(True면 하나의 모델이 여러 벤치마크에
  대하여 학습합니다).

- add_sp_tokens_to_unused:bool, 과제에서 special_token 을 unused 토큰과 대치할 지를 설정합니다.

(그 밖에 허깅페이스의 transformers.TrainingArguments 의 모든 인자를 입력할 수 있습니다.)

## Available Tasks

- GLUE(except "glue-mnli_matched","glue-mnli_mismatched", and "glue-ax")
- KLUE(except "klue-wos")
- koai.benchmarks.finetune_utils.TaskInfo를 이용하여 커스텀 테스크에도 적용 가능합니다.
## Issue

- 현재 개발 중에 있는 프로젝트입니다. 향후 벤치마크가 추가될 예정입니다.
- 소스의 많은 부분들이, https://github.com/huggingface/transformers/ 를 참고 및 인용하여 제작되었습니다.
