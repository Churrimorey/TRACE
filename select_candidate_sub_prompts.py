import argparse
import json
import os

from config import ensure_parent_dir

def select_fewest_harmful_sub_prompts(items, output_file):
    # Select the sub-prompt with the fewest harmful content
    selected_item = min(items, key=lambda x: sum(1 for s in x["harmful_scores"] if s > 0.5))

    with open(output_file, 'a') as f:
        f.write(json.dumps(selected_item) + '\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the least harmful sub-prompt sequence per task id.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file with harmful_scores.")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file for selected items.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_file = args.input_file
    output_file = args.output_file
    ensure_parent_dir(output_file)

    existing_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                item = json.loads(line)
                existing_ids.add(item["id"])

    item_dict = {}

    with open(input_file, 'r') as f:
        for line in f:
            items = json.loads(line)
            if items["id"] in existing_ids:
                print(f"skipping {items['id']} because it already exists")
                continue
            id = items.get('id')
            # sub_prompts = item.get('sub_prompts', [])
            if id not in item_dict:
                item_dict[id] = []
            item_dict[id].append(items)

    for id, items in item_dict.items():
        select_fewest_harmful_sub_prompts(items, output_file)

