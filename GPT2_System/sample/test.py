# %%
from GPT2_System.GPT2_system import GPT2_system
import torch

model_name = "rinna/japanese-gpt2-small"
# %%
gpt_system = GPT2_system(model_name, is_cross_attention=True)

# %%
custom_embeds = torch.randn(1, 1, 768)
prompt = ""
input_ids = gpt_system.encoder(prompt)
output = gpt_system.generate(
    input_ids,
    encoder_hidden_states=custom_embeds,
    max_length=100,
    min_length=100,
    do_sample=True,
    top_k=0,
    top_p=0.95,
    num_beams=5, no_repeat_ngram_size=10
)
# %%
for i, beam_output in enumerate(output):
    print("{}: {}".format(i, gpt_system.tokenizer.decode(
        beam_output, skip_special_tokens=True)))
# %%
