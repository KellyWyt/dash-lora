import json
from typing import List, Optional, Tuple

import torch
from torch import nn
# from transformers import Qwen3ForCausalLM,Qwen3Model,AutoModelForCausalLM,Qwen3Config,AutoConfig
from transformers import AutoConfig, AutoModelForCausalLM

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model
from models.unified_arch import UnifiedMetaForCausalLM, UnifiedMetaModel


class UnifiedConfig(Qwen3Config):
    model_type = "unified_qwen3"


class UnifiedModel(UnifiedMetaModel,Qwen3Model):
    config_class = UnifiedConfig

    def __init__(self, config: Qwen3Config, **kwargs):
        super(UnifiedModel, self).__init__(config, **kwargs)
        self.config = config


class UnifiedForCausalLM(Qwen3ForCausalLM,UnifiedMetaForCausalLM):
    config_class = UnifiedConfig

    def __init__(self, config: Qwen3Config, **kwargs):
        super().__init__(config)
        self.config=config
        self.model = UnifiedModel(config,**kwargs)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def get_model(self) -> UnifiedModel:
        return self.model


    def forward(
        self,
        batch_input_ids = None,
        batch_labels = None,
        # batch_attention_mask = None,
        batch_X_modals = None,
        # batch_question = None,
        batch_task_names = None,
        # used for inference
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):

        if input_ids is not None and input_ids.shape[1]==1:
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            input_ids = None

        elif inputs_embeds is None and batch_input_ids is not None:
            inputs = self.prepare_multimodal_inputs(
                batch_input_ids=batch_input_ids,
                # batch_attenion_mask=batch_attention_mask,
                batch_labels=batch_labels,
                batch_X_modals=batch_X_modals,
                # batch_question=batch_question,
                # return_multi_scale_features=False,
                # return_gt_mask=False,
            )

            input_ids = inputs['input_ids']
            inputs_embeds = inputs['inputs_embeds']
            attention_mask = inputs['attention_mask']
            labels = inputs['labels']
            position_ids = inputs['position_ids']
            # mask_token_mask = inputs['mask_token_mask']
            # multi_scale_image_features = inputs.get('multi_scale_image_features',None)
            # gt_mask = inputs.get('gt_mask',None)

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        return output


    @torch.no_grad()
    def generate(
        self,
        batch_input_ids,
        # batch_attention_mask,
        batch_labels,
        batch_X_modals,
        batch_task_names,
        **kwargs
    ):
        # print('get multimodal input')
        inputs = self.prepare_multimodal_inputs(
            batch_input_ids = batch_input_ids,
            batch_labels = batch_labels,
            batch_X_modals = batch_X_modals,
        )
        inputs_embeds_all = inputs['inputs_embeds']

        return super().generate(
            inputs_embeds=inputs_embeds_all[0],
            modality_masks=inputs_embeds_all[1:],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=False,
            return_dict_in_generate=False,
            **kwargs
        )
    

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        modality_masks=None,
        **kwargs,
    ):
        # print('into prepare inputs...   input_ids:  ',input_ids,'  past key values:  ',past_key_values is None, '   inputs_emebds: ',inputs_embeds is None)
        # if inputs_embeds is not None:
        #     print(inputs_embeds.shape)
        # if past_key_values is not None:
        #     print(f'past key values:  {past_key_values[10][0].shape}')
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            modality_masks=modality_masks,
            **kwargs,
        )
        # print(f'_inputs>>>>>  {_inputs.keys()}')
        # if 'input_ids' in _inputs.keys():
        #     print(_inputs['input_ids'])
        # if 'inputs_embeds' in _inputs.keys():
        #     print(_inputs['inputs_embeds'].shape)
        if images is not None:
            _inputs['images'] = images
        return _inputs

    
    @property
    def device(self):
        return list(self.parameters())[0].device
    

AutoConfig.register("unified_qwen3", UnifiedConfig)
AutoModelForCausalLM.register(UnifiedConfig, UnifiedForCausalLM)

