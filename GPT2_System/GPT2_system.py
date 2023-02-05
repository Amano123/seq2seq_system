# %%
from transformers import T5Tokenizer, AutoModelForCausalLM
from transformers import GPT2Config
from transformers import GPT2Model
import torch

# %%


class GPT2_system:
    """GPT2 system"""

    def __init__(
            self,
            model_name, is_cross_attention=False
    ) -> None:
        self.model_name = model_name
        self.is_cross_attention = is_cross_attention
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.model_name
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name
        )
        self.model.prepare_inputs_for_generation = monkeypatch_prepare_inputs_for_generation
        if self.is_cross_attention:
            add_cross_Attention_config = GPT2Config(
                add_cross_attention=[True]
            )
            # GPT2Config(add_cross_attention=True)をAutoModelForCausalLM.from_pretrainedで読み込めないため、以下の方法で回避している。
            self.model.transformer = GPT2Model(
                add_cross_Attention_config
            )
        # GPU use flag
        # if torch.cuda.is_available():
        #     model = model.to("cuda")

    def encoder(self, text: str):
        input_ids = self.tokenizer.encode(
            text, return_tensors="pt"
        )
        return input_ids

    def decoder(self, ids: torch.Tensor):
        tokens = self.tokenizer.decode(
            ids
        )
        return tokens

    def generate(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states=None,
        max_length=100,
        min_length=100,
        do_sample=True,
        top_k=0,
        top_p=0.95,
        num_beams=5,
        no_repeat_ngram_size=10
    ):
        output = self.model.generate(
            input_ids,
            encoder_hidden_states=encoder_hidden_states,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size
        )
        return output

    def generate_n_word(
        self,
        input_sentence: str,
        n_word: int = 10,
        encoder_hidden_states=None
    ):
        index = 0
        input_ids = self.encoder(
            input_sentence
        )
        while True:
            output = self.model.forward(
                input_ids,
                encoder_hidden_states=encoder_hidden_states
            )
            next_token_logits = output.logits[:, -1, :]
            next_tokens = torch.argmax(
                next_token_logits,
                dim=-1
            )
            input_ids = torch.cat(
                [input_ids, next_tokens[:, None]],
                dim=-1
            )
            if index == n_word:
                break
            index += 1
        return input_ids


def monkeypatch_prepare_inputs_for_generation(input_ids, past=None, **kwargs):
    token_type_ids = kwargs.get("token_type_ids", None)
    # only last token for inputs_ids if past is defined in kwargs
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past:
            position_ids = position_ids[:, -1].unsqueeze(-1)
    else:
        position_ids = None
    model_inputs = {}
    if "inputs_embeds" in kwargs and past is None:
        model_inputs.update({"inputs_embeds": inputs_embeds})
    else:
        model_inputs.update({"input_ids": input_ids})

    if "encoder_hidden_states" in kwargs and past is None:
        model_inputs.update(
            {"encoder_hidden_states": kwargs.get("encoder_hidden_states", None)})
    else:
        model_inputs.update({"encoder_hidden_states": None})

    model_inputs.update({
        "past_key_values": past,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    })
    # print(model_inputs)
    return model_inputs


# %%
if __name__ == "__main__":
    pass
# %%
