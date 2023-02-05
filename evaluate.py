# %%
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
import datetime
import pytz
import datasets
import polars as pl
from tqdm import tqdm
from accelerate import init_empty_weights, infer_auto_device_map

from dataloader.same_minutes_dataset import same_minutes_dataset
from dataloader.samefiles_polars_read import samefiles_polars_read
# %%
pretrained_model_path = "/workspace/results/bart/2023-02-03/01-14-42/checkpoint-7500"
model_name = "stockmark/bart-base-japanese-news"
config = AutoConfig.from_pretrained(model_name)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

device_map = infer_auto_device_map(
    model,  # using the empty model reference here saves time and RAM
    # change max_memory to suit your setup
    max_memory={
        0: "40GB",
        1: "40GB",
        2: "40GB",
        3: "40GB",
    },
)

pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(
    pretrained_model_path,
    device_map=device_map
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


# %%
BODY_MAX_LENGTH = 256
root_path = "/workspace/dataset/digest_and_assembry_pair_dataset"
files = os.listdir(root_path)
file_names = [
    f for
    f in files
    if os.path.isfile(os.path.join(root_path, f))
]
split_num = int(len(file_names) * 0.9)
train_file_name = file_names[:split_num]
eval_file_name = file_names[split_num:]

eval_dataset_df = samefiles_polars_read(root_path, eval_file_name)
# %%
rouge = datasets.load_metric("rouge")
# %%
df_list = []
for row in tqdm(eval_dataset_df.iter_rows(named=True), total=eval_dataset_df.height, leave=False):
    assembry_utterance = row["assembry_utterance"]
    digest_utterance = row["digest_utterance"]
    inputs = tokenizer([assembry_utterance], max_length=256, return_tensors="pt", truncation=True)
    label_id = tokenizer(digest_utterance)
    text_ids = pretrained_model.generate(
        inputs["input_ids"].to(pretrained_model.device),
        num_beams=10, min_length=0, max_length=40
    )
    input_data = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"].tolist()[0], skip_special_tokens=True
    )

    output = tokenizer.convert_ids_to_tokens(
        text_ids[0], skip_special_tokens=True
    )
    output_ids = tokenizer.convert_tokens_to_ids(
        output
    )

    label = tokenizer.convert_ids_to_tokens(
        label_id["input_ids"], skip_special_tokens=True
    )
    label_ids = tokenizer.convert_tokens_to_ids(
        label
    )

    rouge_score = rouge.compute(predictions=[output_ids], references=[label_ids])
    (precision, recall, fmeasure) = list(rouge_score["rouge1"].mid)

    _df = pl.DataFrame(
        {
            "assembly": "|".join(input_data),
            "digest": "|".join(label),
            "output": "|".join(output),
            "type": row["label"],
            "rouge_precision": precision,
            "rouge_recall": recall,
            "rouge_fmeasure": fmeasure
        }
    )
    df_list.append(_df)
# %%
result_df = pl.concat(df_list)
result_df.write_csv("/workspace/results/generate_digest/sample.csv")
