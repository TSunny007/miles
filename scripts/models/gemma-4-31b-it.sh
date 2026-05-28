# Google Gemma 4 31B-it (BF16, DENSE — enable_moe_block=False, no experts).
# HF config (text_config of google/gemma-4-31B-it):
#   architectures: Gemma4ForConditionalGeneration  (a VLM repo; text path is what RL uses)
#   num_hidden_layers=60  hidden_size=5376  intermediate_size=21504  (dense MLP)
#   attention: hybrid sliding+global; head_dim=256 (sliding) / global_head_dim=512
#   num_attention_heads=32  num_key_value_heads=16  num_global_key_value_heads=4
#   sliding_window=1024  max_position_embeddings=262144  vocab_size=262144
#   tie_word_embeddings=True  final_logit_softcapping=30.0  attention_k_eq_v=True
#   hidden_size_per_layer_input=0  (no per-layer embeddings → handled by Gemma4VLBridge)
# The AutoBridge path (--megatron-to-hf-mode bridge) routes this dense config
# through gemma4_vl_bridge.Gemma4VLBridge (Gemma4ForConditionalGeneration). Dense
# support requires the radixark/Megatron-Bridge zhichen/gemma4-dense branch
# (#3885-style dense unblock + attention_k_eq_v K=V tying + dense GatedMLP mappings).
# MODEL_ARGS below carry the attention-side knobs that miles' arg parser inspects
# independently of the bridge-built provider.

MODEL_ARGS=(
   --disable-bias-linear
   --group-query-attention
   --num-attention-heads 32
   --num-query-groups 16
   --kv-channels 256
   --num-layers 60
   --hidden-size 5376
   --ffn-hidden-size 21504
   --normalization RMSNorm
   --norm-epsilon 1e-06
   --position-embedding-type rope
   --rotary-base 1000000
   --vocab-size 262144
   --make-vocab-size-divisible-by 128
   --max-position-embeddings 262144
   # tie_word_embeddings=True per HF config; do not pass --untie-embeddings-and-output-weights
   # DENSE: no --num-experts / --moe-* args (Gemma4VLBridge builds a plain gated MLP).
)
