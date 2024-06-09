def tiny_stories_ref(hidden_size=64, heads=16, layers=2):
    from transformers import GPTNeoConfig, GPTNeoForCausalLM
    return GPTNeoForCausalLM(GPTNeoConfig(
        hidden_size=hidden_size,
        intermediate_size=hidden_size*4,
        num_layers=layers,
        attention_types=[[["global", "local"], layers//2]],
        num_heads=heads,
    ))
    
def bitnet_ref(hidden_size=64, heads=16, layers=2):
    from modeling_bitnet import BitnetForCausalLM , BitnetConfig
    return BitnetForCausalLM(BitnetConfig(
        hidden_size=hidden_size,
        intermediate_size=hidden_size*4,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        vocab_size=32001, # temporary workaround for tokenizer
    ))

