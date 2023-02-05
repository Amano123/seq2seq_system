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
from accelerate import init_empty_weights, infer_auto_device_map

from dataloader.same_minutes_dataset import same_minutes_dataset

torch.cuda.empty_cache()
now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
output_path = now.strftime('%Y-%m-%d/%H-%M-%S')

# %%
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

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    device_map=device_map
)
# %%
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
# %%
train_data = same_minutes_dataset(root_path, train_file_name, tokenizer)
eval_data = same_minutes_dataset(root_path, eval_file_name, tokenizer)

# %%
print('train size', len(train_data))
print('eval size', len(eval_data))

# %%
rouge = datasets.load_metric("rouge")


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


# %%

training_args = Seq2SeqTrainingArguments(
    output_dir=f'results/bart/{output_path}',
    evaluation_strategy="steps",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    warmup_steps=500,
    overwrite_output_dir=True,
    save_total_limit=3,
    logging_dir="logs",
    do_train=True,
    do_eval=True,
    weight_decay=0.1,
    label_smoothing_factor=0.1,
    # fp16=True,
    num_train_epochs=10
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=eval_data,
)

# %%
trainer.train()
