# %%
# from transformers import GPT2Model
from transformers import GPT2Config
import torch
from transformers import T5Tokenizer, AutoModelForCausalLM

# トークナイザーとモデルのロード
model_name = "rinna/japanese-gpt2-medium"
# model_name = "rinna/japanese-gpt2-small"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
add_cross_Attention_config = GPT2Config(add_cross_attention=True)
# model.transformer = GPT2Model(add_cross_Attention_config)

# GPU使用（※GPUを使用しない場合、文章生成に時間がかかります）
# if torch.cuda.is_available():
#     model = model.to("cuda")

# %%
prompt = "むかしむかしあるところにおじいさんとおばあさんがいました。おじいさんは"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
# %%
output = model.forward(
    input_ids
)
# %%
next_token_logits = output.logits[:, -1, :]
next_tokens = torch.argmax(next_token_logits, dim=-1)
print(tokenizer.decode(next_tokens))
# %%
