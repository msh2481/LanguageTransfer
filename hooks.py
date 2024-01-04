import torch.nn as nn
from beartype import beartype as typed
from beartype.door import die_if_unbearable as assert_type
from beartype.typing import Callable
from jaxtyping import Bool, Float, Int
from torch import Tensor as TT
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from language_modeling import tokenize, cross_loss, cross_loss_last


@typed
def lincomb(
    alpha: float,
    x: TT | tuple,
    beta: float,
    y: TT | tuple,
) -> TT | tuple:
    if isinstance(x, tuple) and len(x) == 1:
        return lincomb(alpha, x[0], beta, y)
    if isinstance(y, tuple) and len(y) == 1:
        return lincomb(alpha, x, beta, y[0])
    if isinstance(x, TT) or isinstance(y, TT):
        assert x.shape == y.shape
        return alpha * x + beta * y
    else:
        assert len(x) == len(y)
        return tuple(lincomb(alpha, xi, beta, yi) for xi, yi in zip(x, y))


@typed
def activation_saver(
    inputs_dict: dict[str, TT | tuple],
    outputs_dict: dict[str, TT | tuple],
) -> Callable:
    @typed
    def hook(
        name: str, _module: nn.Module, input: TT | tuple, output: TT | tuple
    ) -> None:
        inputs_dict[name] = input
        outputs_dict[name] = output

    return hook


@typed
def activation_modifier(
    deltas: dict[str, Float[TT, "seq d"]],
) -> Callable:
    @typed
    def hook(
        name: str, _module: nn.Module, _input: TT | tuple, output: TT | tuple
    ) -> TT | tuple:
        if name not in deltas:
            return output
        if isinstance(output, TT):
            assert_type(output, Float[TT, "1 seq d"])
            return output + deltas[name]
        assert isinstance(output, tuple) and isinstance(output[0], TT)
        assert_type(output[0], Float[TT, "1 seq d"])
        return (output[0] + deltas[name], *output[1:])

    return hook


@typed
def chain_patcher(
    steps: list[str],
    real_inputs: dict[str, TT | tuple],
    inputs_dict: dict[str, TT | tuple],
    outputs_dict: dict[str, TT | tuple],
) -> Callable:
    @typed
    def hook(
        name: str, _module: nn.Module, input: TT | tuple, output: TT | tuple
    ) -> TT | tuple:
        inputs_dict[name] = input

        if (name in steps) and ("wte" not in name):
            name_for_delta = name
            if "attn" in name:
                name_for_delta = ".".join(name.split(".")[:-1] + ["ln_1"])
            if "mlp" in name:
                name_for_delta = ".".join(name.split(".")[:-1] + ["ln_2"])
            delta = lincomb(
                1.0, real_inputs[name_for_delta], -1.0, inputs_dict[name_for_delta]
            ).detach()
            if (
                len(output) > len(delta)
                and len(delta) == 1
                and isinstance(delta, tuple)
            ) or (isinstance(output, tuple) and isinstance(delta, TT)):
                a, b = output
                return (lincomb(1.0, a, 1.0, delta), b)
            output = lincomb(1.0, output, 1.0, delta)

        outputs_dict[name] = output
        return output

    return hook


class Hooks:
    @typed
    def __init__(
        self,
        module: nn.Module,
        hook: Callable[[str, nn.Module, TT, TT], TT],
    ) -> None:
        from functools import partial

        self.handles = []
        self.module = module
        for name, submodule in module.named_modules():
            if "." in name:
                self.handles.append(
                    submodule.register_forward_hook(partial(hook, name))
                )

    @typed
    def __enter__(self) -> None:
        pass

    @typed
    def __exit__(self, *_) -> None:
        for handle in self.handles:
            handle.remove()


@typed
def get_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
) -> tuple[dict[str, TT | tuple], dict[str, TT | tuple]]:
    input_dict: dict[str, TT | tuple] = {}
    output_dict: dict[str, TT | tuple] = {}
    with Hooks(model, activation_saver(input_dict, output_dict)):
        model(**tokenize(tokenizer, prompt))
    return input_dict, output_dict


@typed
def prediction_evolution(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
) -> tuple[list[str], Float[TT, "n_layers seq vocab"]]:
    _, output_dict = get_activations(model, tokenizer, prompt)
    layer_names: list[str] = []
    layer_predictions: list[Float[TT, "seq vocab"]] = []

    @typed
    def process(name: str) -> None:
        activation = output_dict[name]
        if isinstance(activation, tuple):
            activation = activation[0]
        assert_type(activation, Float[TT, "batch seq d"])
        activation = activation.squeeze(0)
        logits = model.lm_head(model.transformer.ln_f(activation)).detach()
        assert_type(logits, Float[TT, "seq vocab"])
        logprobs = F.log_softmax(logits, dim=-1)

        layer_names.append(name[len("transformer.") :])
        layer_predictions.append(logprobs)

    process("transformer.wte")
    for layer in range(model.config.num_layers):
        process(f"transformer.h.{layer}.attn")
        process(f"transformer.h.{layer}.mlp")

    return layer_names, t.stack(layer_predictions)


@typed
def track_predictions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    position: int,
) -> None:
    evolution = prediction_evolution(model, tokenizer, prompt)
    assert position != -1
    print("Target token:", tokenizer.tokenize(prompt)[position + 1])
    int_predictions = evolution[1][:, position, :].argmax(dim=-1).numpy()
    print("Predictions:", tokenizer.convert_ids_to_tokens(int_predictions))
    names = evolution[0]
    for token in range(len(tokenizer)):
        history = evolution[1][:, position, token].exp()
        if history[-1] < 0.01:
            continue
        str_token = tokenizer.convert_ids_to_tokens(token)
        plt.plot(names, history, label=f"{str_token} ({token})")
    history = evolution[1][:, position, ::2].exp().sum(dim=-1)
    plt.plot(names, history, label="open (::2)")
    plt.ylim(0, 1)
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()


@typed
def get_layer_residuals(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    layer_list: list[str],
) -> list[Float[TT, "seq d"]]:
    _, output_dict = get_activations(model, tokenizer, prompt)
    activations = []
    for i, layer_name in enumerate(layer_list):
        assert "lm_head" not in layer_name
        raw_output = output_dict[layer_name]
        is_tensor = isinstance(raw_output, TT)
        tensor = (raw_output if is_tensor else raw_output[0]).squeeze(0)
        if i > 1:
            activations.append(tensor - activations[-1])
        else:
            activations.append(tensor)
    return activations


@typed
def path_patching(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    real_prompt: str,
    corrupted_prompt: str,
    steps: list[str],
) -> float:
    real_inputs: dict[str, TT | tuple] = {}
    real_outputs: dict[str, TT | tuple] = {}
    with Hooks(model, activation_saver(real_inputs, real_outputs)):
        real_loss = cross_loss(model, tokenizer, prompt=real_prompt, label=real_prompt)

    corrupted_inputs: dict[str, TT | tuple] = {}
    corrupted_outputs: dict[str, TT | tuple] = {}
    with Hooks(
        model,
        chain_patcher(
            steps=steps,
            real_inputs=real_inputs,
            inputs_dict=corrupted_inputs,
            outputs_dict=corrupted_outputs,
        ),
    ):
        corrupted_loss = cross_loss(
            model, tokenizer, prompt=corrupted_prompt, label=real_prompt
        )
    return corrupted_loss - real_loss


@typed
def layer_importance_on_last_token(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    real_prompt: str,
    corrupted_prompt: str,
) -> dict[str, float]:
    assert len(tokenizer.tokenize(real_prompt)) == len(
        tokenizer.tokenize(corrupted_prompt)
    )

    real_inputs: dict[str, TT | tuple] = {}
    real_outputs: dict[str, TT | tuple] = {}
    with Hooks(model, activation_saver(real_inputs, real_outputs)):
        real_loss = cross_loss_last(
            model, tokenizer, prompt=real_prompt, label=real_prompt
        )

    results: dict[str, float] = {}

    @typed
    def process(name: str) -> None:
        corrupted_inputs: dict[str, TT | tuple] = {}
        corrupted_outputs: dict[str, TT | tuple] = {}
        with Hooks(
            model,
            chain_patcher(
                steps=[name],
                real_inputs=real_inputs,
                inputs_dict=corrupted_inputs,
                outputs_dict=corrupted_outputs,
            ),
        ):
            corrupted_loss = cross_loss_last(
                model, tokenizer, prompt=corrupted_prompt, label=real_prompt
            )
        results[name] = corrupted_loss - real_loss

    process("transformer.wte")
    for layer in range(model.config.num_layers):
        process(f"transformer.h.{layer}.attn")
        process(f"transformer.h.{layer}.mlp")

    return results


@typed
def patch_all_pairs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    real_prompt: str,
    corrupted_prompt: str,
    layer_list: list[str],
) -> Float[TT, "n_layers n_layers"]:
    n_layers = len(layer_list)
    losses: Float[TT, "n_layers n_layers"] = t.zeros((n_layers, n_layers))
    for i in range(n_layers):
        for j in range(n_layers):
            losses[i, j] = path_patching(
                model,
                tokenizer,
                real_prompt=real_prompt,
                corrupted_prompt=corrupted_prompt,
                steps=[layer_list[i], layer_list[j]],
            )
    return losses
