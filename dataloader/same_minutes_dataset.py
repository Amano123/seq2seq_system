# %%

from tqdm import tqdm
from transformers import AutoTokenizer

from dataloader.minutes_dataset import minutes_dataset
# %%


def same_minutes_dataset(
    root_path, file_names: list,
    tokenizer,
    max_length=256
):
    same_encodings = []
    for f_name in tqdm(file_names, total=len(same_encodings), leave=False):
        encodings = minutes_dataset(
            root_path, f_name,
            tokenizer, max_length
        )
        same_encodings.extend(encodings)
    return same_encodings


# %%
if "__main__" == __name__:
    model_name = "stockmark/bart-base-japanese-news"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    root_path = "/workspace/dataset/digest_and_assembry_pair_dataset"
    file_names = ["一般質問(要旨)2月13日.csv", "一般質問(要旨)2月14日.csv"]
    encodings = same_minutes_dataset(root_path, file_names, tokenizer)

# %%
