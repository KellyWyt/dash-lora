
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import transformers

@dataclass
class ModelArguments:  
    sed: int = field(default=42)
    # llm
    model_name_or_path: Optional[str] = field(default="/nfs1/WYT/models/Llama-2-7b-chat-hf") #/data/users/henghui_du/pretrain/video-llama2/Mistral-7B-Instruct-v0.2
    freeze_backbone: bool = field(default=True, metadata={"help": "Whether to freeze the LLM backbone."})
    llm_name: str = field(default='llama') #qwen
    ## visual module
    vit_ckpt_path: str = field(default='/nfs1/outdated/WYT/models/clip-vit-large-patch14') #/group/40061/cserdu/pretrain/openai-clip-vit-large-patch14-224
    select_layer_list = [14,23]
    select_feature: str = field(default='patch')
    image_size: int = field(default=224)
    patch_size: int = field(default=14)
    visual_query_token_nums: int = field(default=32)
    ## audio module
    BEATs_ckpt_path: str = field(default='/nfs1/outdated/WYT/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt') #/group/40061/cserdu/pretrain/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
    audio_query_token_nums: int = field(default=32)
    ## seg module
    prompt_embed_dim: int = field(default=256)
    mask_decoder_transformer_depth: int = field(default=2)
    low_res_mask_size: int = field(default=112)
    image_scale_nums: int = field(default=2)
    token_nums_per_scale: int = field(default=3)
    avs_query_num: int = field(default=300)
    num_classes: int = field(default=1)
    query_generator_num_layers: int = field(default=2)


@dataclass
class InferenceArguments:
    # used for inference
    ckpt_dir: str = field(default='')
    
    # for infer avs
    adapter_ckpt_path: str = field(default=None)
    test_name: str = field(default='test') # for ref-avs: test_u,test_s,test_n
    cut_folds:  int = field(default=2)

    device: str = field(default='cuda:0')
    

@dataclass
class DataArguments:
    # pretrain
    video_frame_nums: int = field(default=8)
    image_size = ModelArguments.image_size
    image_caption_task: bool = field(default=False)
    video_caption_task: bool = field(default=False)
    audio_caption_task: bool = field(default=False)
    # fine-tune
    avqa_task: bool = field(default=True) #False
    ave_task: bool = field(default=False)
    multi_frames: bool = field(default=False) # avs task input single frame



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    mm_projector_lr: Optional[float] = None
    freeze_mm_mlp_adapter: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    # Training Data Arguments 
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # Lora or Quant Arguments
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=32,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = True #False
    lora_r: int = 444
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    ## my
    reserved_modality: str = field(default=None)
    loramethod: str = field(default='train') #None
    blc_alpha: float = field(default=1) #0.5
    blc_weight: float = field(default=1) #0.5

    loratype: str = field(default=None) #NOTE qwen3

    audio_branch: bool = field(default=True) #False
    visual_branch: bool = field(default=True)

    save_modules: str = field(default='vl_projector,al_projector,lora')

    exp_desc: str = field(default='baseline') #exp

    # NOTE 补充的debug所需要的参数
    deepspeed: str = field(default='deepspeed/stage2-offload.json')
    # freeze_backbone: bool = field(default=True)
    bf16: bool = field(default=False) 
    tf32: bool = field(default=True)
    fp16: bool = field(default=False)
    # video_frame_nums: int = field(default=10)
    output_dir: str = field(default='results/finetune_share_12_2/llama_music')
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=2)
    ddp_find_unused_parameters: bool = field(default=True)
    evaluation_strategy: str = field(default="no")
    save_strategy: str = field(default="steps")
    save_steps: float = field(default=0.1)
    save_total_limit: float = field(default=10)
    learning_rate: float = field(default=1e-4) 
    weight_decay: float = field(default=0.) 
    warmup_ratio: float = field(default=0.03) 
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=1)
    gradient_checkpointing: bool = field(default=True)
    half_precision_backend: str = field(default="auto")
    dataloader_num_workers: int = field(default=4)
    report_to: str = field(default="tensorboard")#  >> "${OUTPUT_LOG}" 2>&1

    #TODO
    hierarchical_lora: bool = field(default=True)
    top_k_layers: int = field(
        default=65, 
        metadata={"help": "the number of importance layer,but 49+7=56"}
    )
    ratio:int = field(
        default=3, 
        metadata={"help": "text:video:audio = ratio:1:1"}
    )

