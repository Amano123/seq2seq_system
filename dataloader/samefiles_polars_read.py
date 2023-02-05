# %%
import polars as pl
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
# %%


def samefiles_polars_read(
    root_path, file_names: list,
):
    df_list = []
    for f_name in file_names:
        _df = pl.read_csv(f"{root_path}/{f_name}").drop_nulls()
        fillter_df = _df.filter(
            (pl.col("digest_utterance") != "None") &
            (pl.col("assembry_utterance") != "None")
        )
        df_list.append(fillter_df)
    return pl.concat(df_list)
# %%


# %%
if "__main__" == __name__:
    pass
# %%
