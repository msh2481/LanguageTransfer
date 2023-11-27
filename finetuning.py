# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn.functional as F
from beartype import beartype as typed
from torch import Tensor as TT
from jaxtyping import Float, Int
from typing import Mapping
from itertools import islice
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from dvclive.huggingface import DVCLiveCallback
from IPython.display import clear_output

from languages import dependencies_tokenizer

%load_ext autoreload
%autoreload 2


# %%
def fetch_or_ask(var: str) -> str:
    if var not in os.environ:
        val = input(f"{var}: ")
        clear_output()
        os.environ[var] = val
    return os.environ[var]


gdrive_token = fetch_or_ask("GDRIVE_CREDENTIALS_DATA")
os.environ[
    "DVC_STUDIO_TOKEN"
] = "isat_1mr9HNvqAB6xw8OJ3dXe5O9vMaKol59LCoA5gGP3eLY8NoSF8"

# %%
dataset = load_dataset("Mlxa/flat", streaming=True)
model_name = "Mlxa/brackets-nested"
tokenizer = dependencies_tokenizer(vocab_size=500)
model = AutoModelForCausalLM.from_pretrained(model_name)

# %%
model.resize_token_embeddings(len(tokenizer))
for name, param in model.named_parameters():
    param.requires_grad = "wte" in name or "wpe" in name

# %%
tokens_sample = tokenizer(next(iter(dataset["train"]))["text"])["input_ids"]
print(len(tokens_sample))
print(tokens_sample[:10])


# %%
@typed
def tokenize_function(example: Mapping[str, str | int]) -> Mapping[str, list[int]]:
    result = tokenizer(example["text"])
    result["labels"] = result["input_ids"]
    return result


train_size = 100000
test_size = 1000
tokenized_train = (
    dataset["train"]
    .map(tokenize_function, batched=True)
    .remove_columns(["text"])
    .take(train_size)
)
tokenized_test = (
    dataset["test"]
    .map(tokenize_function, batched=True)
    .remove_columns(["text"])
    .take(test_size)
)


# %%
@typed
def show_string_with_weights(s: list[str], w: list[float] | Float[TT, "seq"]) -> None:
    from IPython.display import HTML, display
    from matplotlib.colors import rgb2hex
    from matplotlib import colormaps

    cmap = colormaps["coolwarm"]

    def brighten(rgb):
        return tuple([(x + 1) / 2 for x in rgb])

    colors = [brighten(cmap(alpha)) for alpha in w]
    html_str_colormap = " ".join(
        [
            f'<span style="background-color: {rgb2hex(color)}; padding: 1px; margin: 0px; border-radius: 5px;">{word}</span>'
            for word, color in zip(s, colors)
        ]
    )
    display(HTML(html_str_colormap))


@typed
def sample_and_logprobs(sample: Mapping[str, list[int]]) -> float:
    model.cuda()
    gen_length = 10
    with t.no_grad():
        inputs = {k: t.tensor([v], device="cuda") for k, v in sample.items()}
        ids = inputs["input_ids"][0]
        pos = len(ids) - gen_length
        truncated = {k: v[:, :pos] for k, v in inputs.items()}
        sampled_tokens = (
            model.generate(
                **truncated,
                max_new_tokens=gen_length,
                pad_token_id=tokenizer.pad_token_id,
                bad_words_ids=[[tokenizer.pad_token_id]],
                do_sample=True,
            )[0]
            .detach()
            .cpu()
        )
        without_prompt = tokenizer.decode(sampled_tokens[pos:])

        output = model(**inputs)
        logprobs: Float[TT, "seq vocab"] = F.log_softmax(
            output.logits.cpu().detach(), dim=-1
        ).squeeze(0)

        labels: Int[TT, "seq 1"] = inputs["input_ids"][0, 1:].cpu().unsqueeze(-1)
        lp_per_token: Float[TT, "seq"] = (
            logprobs[:-1].gather(-1, labels).squeeze(-1)[pos - 1 :]
        )
        weights = F.tanh(-lp_per_token)  # 0 for perfect prediction, 1 for infinite loss
        tokens = [tokenizer.decode(i) for i in ids[pos:]]

        show_string_with_weights(tokens, weights)
        print(without_prompt)
        return output.loss.item()


@typed
def train(batch_size: int, lr: float) -> None:
    training_args = TrainingArguments(
        output_dir="trainer",
        fp16=False,
        per_device_train_batch_size=batch_size,
        torch_compile=False,
        learning_rate=lr,
        logging_steps=10,
        num_train_epochs=1,
        max_steps=train_size // batch_size,
        save_total_limit=1,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
    )
    trainer.add_callback(DVCLiveCallback())
    trainer.train()


@typed
def evaluate(n_samples: int) -> None:
    losses = []
    for sample in islice(tokenized_test, n_samples):
        losses.append(sample_and_logprobs(sample))
    mean_loss = sum(losses) / len(losses)
    print(f"Mean loss: {mean_loss:.3f}")

# %%
# Fine-tuning only embeddings:
train(batch_size=8, lr=1e-2)

# %%
evaluate(n_samples=32)

# %%
# Fine-tuning only embeddings and layernorms:
for name, param in model.named_parameters():
    if "ln" in name:
        print(f"{name} unfrozen")
        param.requires_grad = True
train(batch_size=8, lr=2e-3)

# %%
evaluate(n_samples=32)

# %%
# Fine-tuning only embeddings, layernorms and the last block:
for name, param in model.named_parameters():
    if "h.7" in name:
        param.requires_grad = True
train(batch_size=8, lr=1e-3)

# %%
evaluate(n_samples=32)

# %%
for sample in islice(tokenized_train, 32):
    sample_and_logprobs(sample)

# %%
from huggingface_hub import notebook_login

notebook_login()

name = input("Model name, e.g. brackets-flat_shuffle: ")
model.push_to_hub(name)
tokenizer.push_to_hub(name)
# %%
1 / 0

# %%
import gc

gc.collect()
t.cuda.empty_cache()
