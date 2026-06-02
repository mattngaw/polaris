"""Generation / inference for `polaris`.

Load a checkpoint, build the model from the checkpointed config, sample
tokens. Used to eyeball whether a trained model produces coherent output
and to compare qualitative behavior across (dense, moe) pairs.

A checkpoint-format note: the standard `torch.save` of a state_dict is
fine for this project, but if you ever move models across PyTorch versions
or out of the gfx803 container, `safetensors` is the portable choice.
Worth keeping the save path pluggable.
"""

# TODO(human): argument parsing (checkpoint path, prompt, num_samples,
#              temperature, top_k).
# TODO(human): load checkpoint, rebuild model from saved config, .eval().
# TODO(human): tokenize the prompt with the same tokenizer the training
#              run used (look it up via the meta.pkl alongside the .bin
#              files).
# TODO(human): generation loop — forward, sample next token, append,
#              repeat. The nanoGPT generate() is borrowable verbatim
#              (note in CREDITS).
# TODO(human): decode and print.

if __name__ == "__main__":
    raise NotImplementedError("TODO(human): wire up the entry point.")
