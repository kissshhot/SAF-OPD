import json
import os

import datasets

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from verl.utils.reward_score.math_reward import remove_boxed
except ImportError:
    remove_boxed = None


def process_dataset(data_source, jsonl_path):
    rows = []
    with open(jsonl_path) as f:
        for idx, line in enumerate(f):
            ex = json.loads(line)
            question = ex["problem"]
            answer_raw = str(ex["answer"])

            if remove_boxed is not None:
                try:
                    ground_truth = remove_boxed(answer_raw)
                except Exception:
                    ground_truth = answer_raw
            else:
                ground_truth = answer_raw

            rows.append({
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question + " " + INSTRUCTION}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {"index": idx, "question": question, "answer": answer_raw},
            })

    ds = datasets.Dataset.from_list(rows)
    out_path = os.path.join(os.path.dirname(jsonl_path), "test.parquet")
    ds.to_parquet(out_path)
    print(f"[{data_source}] {len(ds)} samples -> {out_path}")


if __name__ == "__main__":
    for name in sorted(os.listdir(DATA_DIR)):
        jsonl_path = os.path.join(DATA_DIR, name, "test.jsonl")
        if os.path.isfile(jsonl_path):
            process_dataset(name, jsonl_path)
