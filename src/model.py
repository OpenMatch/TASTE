"""
Model design code refers to FID code:https://github.com/facebookresearch/FiD/blob/main/src/model.py.
########################
"""
import torch
from torch import nn
from transformers import T5Model


class TASTEModel(T5Model):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()


    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(
            self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint


    def encode(self,input_ids, attention_mask):

        if input_ids.dim() == 3:
            self.encoder.n_passages = input_ids.size(1)
        input_ids = input_ids.view(input_ids.size(0), -1)
        attention_mask = attention_mask.view(attention_mask.size(0), -1)
        decoder_input_ids = torch.zeros((input_ids.shape[0], 1), dtype=torch.long)
        decoder_input_ids = decoder_input_ids.to(input_ids.device)
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        )
        hidden = output.last_hidden_state
        reps = hidden[:, 0, :]

        return hidden,reps

    def forward(self, *input):
        return self.encode(*input)



class EncoderWrapper(nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """

    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz * self.n_passages, passage_length)  # bn,l
        attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)

        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        outputs.last_hidden_state = outputs[0].view(bsz, self.n_passages *
                                                    passage_length, -1)  # b,nl,768

        # outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
        #                        return_dict=True)  # bn,l,768
        # outputs.last_hidden_state = outputs[0].view(bsz, self.n_passages *
        #                                             passage_length, -1)  # b,nl,768

        return outputs


class CheckpointWrapper(nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """

    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(
                hidden_states, attention_mask, position_bias, **kwargs)
        return output


def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block




