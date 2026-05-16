# Google Gemma 4 26B-A4B-it (BF16, MoE with 128 experts top-k=8).
# HF config (verified 2026-05-15, from text_config of google/gemma-4-26B-A4B-it):
#   architectures: Gemma4ForConditionalGeneration  (a VLM repo; text path is what RL uses)
#   num_hidden_layers=30  hidden_size=2816  ffn_intermediate_size=2112
#   attention: hybrid sliding+global; head_dim=256 (sliding) / global_head_dim=512
#   num_attention_heads=16  num_key_value_heads=8  num_global_key_value_heads=2
#   sliding_window=1024  max_position_embeddings=262144  vocab_size=262144
#   MoE: num_experts=128  top_k_experts=8  moe_intermediate_size=704
#   tie_word_embeddings=True  final_logit_softcapping=30.0  attention_k_eq_v=True
# The AutoBridge path (--megatron-to-hf-mode bridge) routes Gemma4 configs
# through megatron.bridge.models.gemma.gemma4_bridge.Gemma4Bridge (text-only) or
# gemma4_vl_bridge.Gemma4VLBridge (multimodal). MODEL_ARGS below carry the
# attention-side knobs that miles' arg parser inspects independently of the
# bridge-built provider.

MODEL_ARGS=(
   --disable-bias-linear
   --group-query-attention
   --num-attention-heads 16
   --num-query-groups 8
   --kv-channels 256
   --num-layers 30
   --hidden-size 2816
   --ffn-hidden-size 2112
   --normalization RMSNorm
   --position-embedding-type rope
   --rotary-base 1000000
   --vocab-size 262144
   --make-vocab-size-divisible-by 128
   --max-position-embeddings 262144

   # MoE specifics (matches nemotron-3-nano-30b-a3b style; routing replay
   # requires the radixark/Megatron-Bridge bridge port from
   # zhichen/gemma4-on-bridge for the Gemma4Bridge mappings + ABSENT_PROJECTION
   # sentinel + FusedExpertMapping).
   --num-experts 128
   --moe-router-topk 8
   --moe-ffn-hidden-size 704
   --moe-router-score-function softmax
   --moe-grouped-gemm
   --moe-router-dtype fp32
   --moe-router-num-groups 1
   --moe-router-group-topk 1
   --moe-router-load-balancing-type seq_aux_loss
   --moe-router-bias-update-rate 0
   --moe-aux-loss-coeff 0
)
