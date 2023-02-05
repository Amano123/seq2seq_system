# %%
import polars as pl
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
# %%


def minutes_dataset(
    root_path, file_name,
    tokenizer,
    max_length=256
):
    """digest_and_assembry_pair dataset"""
    encodings = []
    dataset_df: pl.DataFrame = pl.read_csv(
        f"{root_path}/{file_name}"
    ).drop_nulls()
    for row in tqdm(dataset_df.iter_rows(named=True), total=dataset_df.height, leave=False):
        inputs = tokenizer(row['assembry_utterance'], padding='max_length', truncation=True, max_length=max_length)
        outputs = tokenizer(row["digest_utterance"], padding='max_length', truncation=True, max_length=max_length)
        inputs['decoder_input_ids'] = outputs['input_ids']
        inputs['decoder_attention_mask'] = outputs['attention_mask']
        inputs['labels'] = outputs['input_ids'].copy()
        inputs['labels'] = [
            -100
            if token == tokenizer.pad_token_id
            else token
            for token in inputs["labels"]
        ]
        inputs = {
            k: torch.tensor(v)
            for k, v in inputs.items()
        }
        encodings.append(inputs)
    return encodings
# %%


# %%
if "__main__" == __name__:
    model_name = "stockmark/bart-base-japanese-news"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    root_path = "/workspace/dataset/digest_and_assembry_pair_dataset"
    file_name = "一般質問(要旨)2月13日.csv"
    encodings = minutes_dataset(root_path, file_name, tokenizer)

# %%
