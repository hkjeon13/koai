from collections import OrderedDict


def default_preprocess_function(examples):
    return examples


def klue_sts_preprocess_function(examples):
    examples['labels'] = [label["binary-label"] for label in examples['labels']]
    return examples


def klue_re_preprocess_function(examples, apply_type_tag=False,
                                sub_token=" <sub> ", unsub_token=" </sub> ", obj_token=" <obj> ",
                                unobj_token=" </obj> "):
    sentences = []
    for sentence, sub, obj in zip(examples["sentence"], examples["subject_entity"], examples["object_entity"]):
        sub["cls"], sub["uncls"], obj["cls"], obj["uncls"] = sub_token, unsub_token, obj_token, unobj_token
        if apply_type_tag:
            sub["cls"], sub["uncls"] = f' <{sub["type"]}> ', f' </{sub["type"]}> '
            obj["cls"], obj["uncls"] = f' <{obj["type"]}> ', f' </{obj["type"]}> '
        former, later = sorted([sub, obj], key=lambda x: x['start_idx'])

        sentence = sentence[:former["start_idx"]] + former["cls"] + sentence[former["start_idx"]:former["end_idx"] + 1] \
                   + former["uncls"] + sentence[former["end_idx"] + 1:later["start_idx"]] + later['cls'] \
                   + sentence[later["start_idx"]:later["end_idx"] + 1] + later["uncls"] + sentence[
                                                                                          later["end_idx"] + 1:]

        sentences.append(sentence)

    examples['sentence'] = sentences
    return examples


if __name__ == "__main__":
    from datasets import load_dataset

    dataset = load_dataset("klue", 're', split="train")
    dataset = dataset.map(klue_re_preprocess_function, batched=True)
    print(next(iter(dataset)))
