
bitnet_base = {
    'name': 'bitnet-base',
    "architectures": ["BitnetForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 1536,
    "initializer_range": 0.02,
    "input_bits": 8,
    "intermediate_size": 4096,
    "max_position_embeddings": 2048,
    "model_type": "llama",
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "num_key_value_heads": 16,
    "pad_token_id": 32000,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
    "torch_dtype": "float32",
    "use_cache": True,
    "vocab_size": 32002,
    "weight_bits": 1
}

tiny_stories_base = {
  "_name_or_path": "EleutherAI/gpt-neo-125M",
  "activation_function": "gelu_new",
  "architectures": [
    "GPTNeoForCausalLM"
  ],
  "attention_dropout": 0,
  "attention_layers": [
    "global",
    "local",
    "global",
    "local",
    "global",
    "local",
    "global",
    "local"
  ],
  "attention_types": [
    [
      [
        "global",
        "local"
      ],
      4
    ]
  ],
  "bos_token_id": 50256,
  "embed_dropout": 0,
  "eos_token_id": 50256,
  "gradient_checkpointing":False,
  "hidden_size": 64,
  "initializer_range": 0.02,
  "intermediate_size": None,
  "layer_norm_epsilon": 1e-05,
  "max_position_embeddings": 2048,
  "model_type": "gpt_neo",
  "num_heads": 16,
  "num_layers": 8,
  "resid_dropout": 0,
  "summary_activation": None,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": True,
  "summary_type": "cls_index",
  "summary_use_proj": True,
  "torch_dtype": "float32",
  "use_cache": True,
  "vocab_size": 50257,
  "window_size": 256
}


def llm_shape(base, hidden_size, heads = 16, layers = 2):
    b = base.copy()
    if b["intermediate_size"] is not None:
        b["intermediate_size"] = hidden_size * 2
    return b.update({
        "hidden_size" : hidden_size,
        "num_heads" : heads,
        "num_attention_heads": heads,
        "num_hidden_layers": layers,
        "num_key_value_heads": heads,
    })

bitnet_64_2 = llm_shape(bitnet_base, hidden_size=64)

def tiny_stories_ref(hidden_size=64, heads=16, layers=2):
    from transformers import GPTNeoConfig, GPTNeoForCausalLM
    return GPTNeoForCausalLM(GPTNeoConfig(
        hidden_size=hidden_size,
        num_layers=layers,
        attention_types=[[["global", "local"], layers//2]],
        num_heads=heads,
    ))

