"""SentEval probing tasks (SOMO, CoordInv): loading into datasets splits and a left-padding collator."""

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
from datasets import Dataset

# Files of scripts/probing/download_senteval.sh.
TASK_FILE_NAMES = {
    "SOMO": "odd_man_out.txt",
    "CoordInv": "coordination_inversion.txt",
}


def load_senteval_data(task_name: str, data_dir: str):
    """(train, test) Datasets with columns text and label (labels encoded as integers)."""
    with open(f"{data_dir}/{TASK_FILE_NAMES[task_name]}", "r") as f:
        lines = f.readlines()

    dataset = {split: {"text": [], "label": []} for split in ("tr", "va", "te")}
    for line in lines:
        split, label, text = line.strip().split("\t")
        dataset[split]["text"].append(text.strip())
        dataset[split]["label"].append(label)

    assert set(dataset["va"]["label"]) == set(dataset["te"]["label"])
    assert set(dataset["tr"]["label"]) == set(dataset["te"]["label"])
    # sorted: a set's order changes with the hash seed of each process
    label_to_int = {label: i for i, label in enumerate(sorted(set(dataset["tr"]["label"])))}
    for split in ("tr", "te"):
        dataset[split]["label"] = [label_to_int[label] for label in dataset[split]["label"]]

    return Dataset.from_dict(dataset["tr"]), Dataset.from_dict(dataset["te"])


@dataclass
class DataCollatorForCausalLM:
    """Left-pads input_ids (with pad_token_id) and attention_mask (with 0)."""

    pad_token_id: int

    def __call__(self, instances: Sequence[Dict]) -> Dict:
        input_ids = [torch.tensor(instance["input_ids"]) for instance in instances]
        attention_mask = [torch.tensor(instance["attention_mask"]) for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id, padding_side="left"
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0, padding_side="left"
        )
        return dict(input_ids=input_ids, attention_mask=attention_mask)
