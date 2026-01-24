import re

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaConfig,
    AutoConfig,
    AutoModelForCausalLM,
    BertConfig,
    BertModel,
    BertForSequenceClassification
)

from transformers.modeling_outputs import CausalLMOutputWithPast


class coff_OPTDecoderLayer(nn.Module):
    def __init__(self, original_decoder_layer, coff=1.):
        super().__init__()
        self.embed_dim = original_decoder_layer.embed_dim

        self.self_attn = original_decoder_layer.self_attn

        self.do_layer_norm_before = original_decoder_layer.do_layer_norm_before
        self.dropout = original_decoder_layer.dropout
        self.activation_fn = original_decoder_layer.activation_fn

        self.self_attn_layer_norm = original_decoder_layer.self_attn_layer_norm
        self.fc1 = original_decoder_layer.fc1
        self.fc2 = original_decoder_layer.fc2
        self.final_layer_norm = original_decoder_layer.final_layer_norm

        self.a = coff

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if residual.device != hidden_states.device:
            residual = residual.to(hidden_states.device)

        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states * self.a,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class coff_LlamaAttnLayer(nn.Module):
    def __init__(self, original_decoder_layer):
        super().__init__()
        self.hidden_size = original_decoder_layer.hidden_size

        self.self_attn = original_decoder_layer.self_attn
        self.input_layernorm = original_decoder_layer.input_layernorm

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        outputs = residual.to(hidden_states.device) + hidden_states

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class coff_LlamaMlpLayer(nn.Module):
    def __init__(self, original_decoder_layer, coff=1.):
        super().__init__()
        self.hidden_size = original_decoder_layer.hidden_size

        self.mlp = original_decoder_layer.mlp
        self.post_attention_layernorm = original_decoder_layer.post_attention_layernorm

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual.to(hidden_states.device) + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (None,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs


def coff_block_replace(model):
    num_layers = len(model.model.layers)
    new_layers = []
    for i in range(num_layers):
        new_layers.append(coff_LlamaAttnLayer(model.model.layers[i]))
        new_layers.append(coff_LlamaMlpLayer(model.model.layers[i]))
    model.model.layers = nn.ModuleList(new_layers)
    # model.config.num_hidden_layers = len(new_layers)
    print("Replacement complete.")


class coff_LlamaDecoderLayer(nn.Module):
    def __init__(self, original_decoder_layer, coff=1.):
        super().__init__()
        self.hidden_size = original_decoder_layer.hidden_size

        self.self_attn = original_decoder_layer.self_attn
        self.mlp = original_decoder_layer.mlp
        self.input_layernorm = original_decoder_layer.input_layernorm
        self.post_attention_layernorm = original_decoder_layer.post_attention_layernorm

        self.a = 1.
        self.b = 1.
        if isinstance(coff, list) or isinstance(coff, tuple):
            self.a = coff[0]
            self.b = coff[1]
        else:
            self.b = coff

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual.to(hidden_states.device) + hidden_states

        hidden_states *= self.a

        # mlp
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual.to(hidden_states.device) + hidden_states

        outputs = (hidden_states * self.b,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class coff_QWenBlock(nn.Module):
    def __init__(self, original_decoder_layer, coff=1.):
        super().__init__()

        self.ln_1 = original_decoder_layer.ln_1
        self.attn = original_decoder_layer.attn
        self.ln_2 = original_decoder_layer.ln_2

        self.mlp = original_decoder_layer.mlp

        self.a = 1.
        self.b = 1.
        if isinstance(coff, list) or isinstance(coff, tuple):
            self.a = coff[0]
            self.b = coff[1]
        else:
            self.b = coff

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            rotary_pos_emb_list: Optional[List[List[torch.Tensor]]] = None,
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ):
        layernorm_output = self.ln_1(hidden_states)

        attn_outputs = self.attn(
            layernorm_output,
            rotary_pos_emb_list,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        residual = hidden_states
        layernorm_input = attn_output + residual.to(attn_output.device)

        layernorm_output = self.ln_2(layernorm_input)

        residual = layernorm_input
        mlp_output = self.mlp(layernorm_output)
        hidden_states = residual + mlp_output

        if use_cache:
            outputs = (hidden_states * self.b,) + outputs
        else:
            outputs = (hidden_states * self.b,) + outputs[1:]

        return outputs


def coff_decoder_replace(model, coffs, model_type="llama"):
    type = 0
    if "Llama" in model_type or "llama" in model_type or "vicuna" in model_type or "Mistral" in model_type:
        layers = model.model.layers
    elif "opt" in model_type:
        layers = model.model.decoder.layers
        type = 1
    elif "Qwen" in model_type:
        layers = model.transformer.h
        type = 2
    else:
        raise NotImplementedError
    num_layers = len(coffs)
    for i in range(num_layers):
        if type == 0:
            layers[i] = coff_LlamaDecoderLayer(layers[i], coffs[i])
        elif type == 1:
            layers[i] = coff_OPTDecoderLayer(layers[i], coffs[i])
        else:
            layers[i] = coff_QWenBlock(layers[i], coffs[i])
    print("Replacement complete.")


class OnOff_LlamaDecoderLayer(nn.Module):
    def __init__(self, original_decoder_layer):
        super().__init__()
        self.hidden_size = original_decoder_layer.hidden_size

        self.self_attn = original_decoder_layer.self_attn
        self.mlp = original_decoder_layer.mlp
        self.input_layernorm = original_decoder_layer.input_layernorm
        self.post_attention_layernorm = original_decoder_layer.post_attention_layernorm

        self.pass_mha = False
        self.pass_mlp = False
        self.input = None
        self.output = None

        self.s = None
        self.bias = None

    def turn_off(self):
        self.pass_mha = True
        self.pass_mlp = True

    def turn_on(self):
        self.pass_mha = False
        self.pass_mlp = False

    def turn_off_mha(self):
        self.pass_mha = True

    def turn_on_mha(self):
        self.pass_mha = False

    def turn_off_mlp(self):
        self.pass_mlp = True

    def turn_on_mlp(self):
        self.pass_mlp = False

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if not self.pass_mha:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, self_attn_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            hidden_states = residual.to(hidden_states.device) + hidden_states
        else:
            self_attn_weights = None

        if not self.pass_mlp:
            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual.to(hidden_states.device) + hidden_states

        if self.bias is None:
            outputs = (hidden_states,)
        else:
            outputs = (hidden_states * self.s.to(device=hidden_states.device) + self.bias.to(device=hidden_states.device), )

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


def block_replace(model):
    num_layers = len(model.model.layers)
    for i in range(num_layers):
        model.model.layers[i] = OnOff_LlamaDecoderLayer(model.model.layers[i])
    print("Replacement complete.")

    return model


def turn_off_layer(model, layer_idx):
    model.model.layers[layer_idx].turn_off()


def turn_on_layer(model, layer_idx):
    model.model.layers[layer_idx].turn_on()


def turn_off_mha(model, layer_idx):
    model.model.layers[layer_idx].turn_off_mha()


def turn_on_mha(model, layer_idx):
    model.model.layers[layer_idx].turn_on_mha()


def turn_off_mlp(model, layer_idx):
    model.model.layers[layer_idx].turn_off_mlp()


def turn_on_mlp(model, layer_idx):
    model.model.layers[layer_idx].turn_on_mlp()


def scan(model, num_blocks):
    alive_list = []
    skip_list = []

    for i in range(num_blocks):
        if model.model.layers[i].pass_layer == True:
            skip_list.append(i)
        elif model.model.layers[i].pass_layer == False:
            alive_list.append(i)

    print(
        f"pass layer: {skip_list}\n"
        f"do layer: {alive_list}"
    )


class DynamicLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, pruning_num, base_model=None, tokenizer=None, router_model=None, router_tokenizer=None, params_dict=None):

        super().__init__(config)
        self.n_layers = config.num_hidden_layers

        self.tokenizer = tokenizer
        self.router_tokenizer = router_tokenizer

        if base_model is not None:
            self.model = base_model
        else:
            print('no llm model loaded...')
        if router_model is not None:
            self.router = router_model
        else:
            print('no trained router loaded...')
        self.router.eval()

        self.seqlen = 2048
        self.params_dict = params_dict
        self.pruning_num = pruning_num

        self.input_set = {}

    def get_skip_mask(self, router_logits):

        probabilities = router_logits
        top_indices = torch.argmin(probabilities, dim=-1)
        predicted_label = top_indices.item()  # (1,1) -> int

        skip_layer = [idx for idx in range(predicted_label, predicted_label + self.pruning_num)]
        self.base_model.model.layers[predicted_label].s = self.params_dict[predicted_label][0]
        self.base_model.model.layers[predicted_label].bias = self.params_dict[predicted_label][1]

        return skip_layer

    def forward(self, input_ids=None, attention_mask=None, skip_layer=None, **kwargs):
        seq_len = input_ids.size(1)
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones((1, seq_len), device=device)

        if skip_layer is None:
            with torch.no_grad():
                input_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]

                answer_index = input_text.find("Answer")

                if answer_index != -1:
                    question_input = input_text[:answer_index]
                else:
                    match = re.search(r":\s*([^\.]*\.)", input_text)
                    if match:
                        question_input = match.group(1).strip()
                    else:
                        words = input_text.split()
                        question_input = " ".join(words[:7])

                if question_input in self.input_set:
                    skip_layer = self.input_set[question_input]

                else:
                    bert_inputs = self.bert_tokenizer(
                        input_text,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(device)

                    router_outputs = self.router(
                        input_ids=bert_inputs['input_ids'],
                        attention_mask=bert_inputs['attention_mask']
                    )
                    router_logits = router_outputs.logits  # shape: (batch=1, num_labels=10)

                    skip_layer = self.get_skip_mask(router_logits)
                    self.input_set[question_input] = skip_layer

                # input_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
                # bert_inputs = self.router_tokenizer(
                #     input_text,
                #     return_tensors='pt',
                #     padding=True,
                #     truncation=True,
                #     max_length=512
                # ).to(device)
                #
                # router_outputs = self.router(
                #     input_ids=bert_inputs['input_ids'],
                #     attention_mask=bert_inputs['attention_mask']
                # )
                # router_logits = router_outputs.logits  # shape: (batch=1, num_labels=10)
                #
                # skip_layer = self.get_skip_mask(router_logits)

        for idx in skip_layer:
            turn_off_layer(self.model, idx)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        for idx in skip_layer:
            turn_on_layer(self.model, idx)
        self.base_model.model.layers[skip_layer[0]].s = None
        self.base_model.model.layers[skip_layer[0]].bias = None

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            hidden_states=None,
            attentions=None,
            # cross_attentions=None
        )

