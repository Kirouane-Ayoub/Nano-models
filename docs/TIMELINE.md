# Timeline

Every component and architecture in this repo, in the order the ideas were
published. Dates are the month of the originating paper (the arXiv identifier
encodes it: `2405.21060` is May 2024) or of the model release when there is no
paper. Later adoption milestones are labelled separately from the originating
idea. The last column identifies the implementation or notes related work
that is not implemented. `ARCHITECTURES.md` explains each component; this file
only orders them.

Two things the order shows. First, almost nothing in a 2026 model is from 2026:
the components are 2023–2025 ideas that a lab finally shipped together. Second,
the ideas that survived are the ones that shrink the KV cache or the per-token
compute — GQA, MLA, sliding windows, linear mixers, sparse selection — and the
2026 releases are mostly new combinations of them.

## Foundations, 2016–2022

| Date | Idea | Paper / origin | Here |
|---|---|---|---|
| 2016-08 | Tied input/output embeddings | Press & Wolf, arXiv 1608.05859 | every model ties `head` to `tok_emb` |
| 2017-01 | Sparse MoE, top-k routing | Shazeer et al., [arXiv 1701.06538](https://arxiv.org/abs/1701.06538); later top-1 simplification: Switch Transformer, arXiv 2101.03961 (2021-01) | `deepseek_nano.py` |
| 2017-06 | Multi-head attention, the transformer | *Attention Is All You Need*, arXiv 1706.03762 | `--attention mha` |
| 2017-06 | Embedding scaled by √d | *Attention Is All You Need*, [arXiv 1706.03762](https://arxiv.org/abs/1706.03762), §3.4 | `gemma_nano.py` |
| 2018-07 | Depth recurrence (looped layers) | *Universal Transformers*, arXiv 1807.03819 | origin of `looped_nano.py` |
| 2019-10 | RMSNorm | arXiv 1910.07467 | all model families except `gpt_nano`, which uses LayerNorm |
| 2020-02 | Gated FFNs: SwiGLU, GeGLU | *GLU Variants Improve Transformer*, arXiv 2002.05202 | SwiGLU in `qwen_nano`, GeGLU in `gemma_nano` |
| 2020-04 | Sliding-window attention | Longformer, arXiv 2004.05150; Mistral 7B, arXiv 2310.06825 (2023-10) | `--attention swa`, `--linear swa` |
| 2020-10 | Query-key normalization | Henry et al., [arXiv 2010.04245](https://arxiv.org/abs/2010.04245) (L2 normalization with learned scale) | precursor of the RMSNorm-based `qk_norm` in `qwen_nano`, `gemma_nano` |
| 2021-04 | RoPE | RoFormer, arXiv 2104.09864 | `qwen_nano` and every model built on it |
| 2021-05 | Sandwich normalization | CogView, [arXiv 2105.13290](https://arxiv.org/abs/2105.13290) (LayerNorm); later RMSNorm variant: Gemma 2 (2024-08) | `--norm sandwich`, `gemma_nano` use RMSNorm before and after sublayers |
| 2021-06 | Parallel attention + FFN block | GPT-J (EleutherAI); PaLM, arXiv 2204.02311 (2022-04) | `gpt_nano --block parallel` |
| 2022-02 | Expert-choice routing | arXiv 2202.09368 | `deepseek_nano --routing expert` |
| 2022-03 | NoPE | Haviv et al., arXiv 2203.16634; Kazemnejad et al., arXiv 2305.19466 (2023-05) | `qwen_next_nano --posenc nope` |

## Efficiency, 2023–2024

| Date | Idea | Paper / origin | Here |
|---|---|---|---|
| 2023-02 | QK-norm at 22B scale (later adoption) | *Scaling ViT to 22B*, arXiv 2302.05442 (LayerNorm on Q/K) | related variant; this repo uses RMSNorm on Q/K |
| 2023-05 | Grouped-query attention | arXiv 2305.13245 | `--attention gqa` |
| 2023-07 | Lightning attention (linear, fixed decay) | TransNormerLLM, arXiv 2307.14995; Lightning Attention-2, arXiv 2401.04658 (2024-01) | `--attention lightning` |
| 2023-09 | Attention sinks (the diagnosis) | StreamingLLM, arXiv 2309.17453 | `--attention sink`, `swa` + `attn_sink` |
| 2024-01 | Shared expert | DeepSeekMoE, arXiv 2401.06066 | `shared_expert` in `deepseek_nano` |
| 2024-03 | `(1 + w)` RMSNorm | Gemma 1, arXiv 2403.08295 | `GemmaRMSNorm` in `gemma_nano` |
| 2024-04 | Mixture-of-Depths | arXiv 2404.02258 | `qwen_next_nano --mod-capacity` |
| 2024-04 | Multi-token prediction | Meta, arXiv 2404.19737; DeepSeek-V3 (2024-12) | `qwen_next_nano --mtp-weight` |
| 2024-05 | Multi-head latent attention | DeepSeek-V2, arXiv 2405.04434 | `--attention mla`, `deepseek_nano`, `qwen_next_nano --attn mla` |
| 2024-05 | Cross-layer KV sharing | CLA, arXiv 2405.12981; Gemma 4 (2026-07) | `qwen_next_nano --kv-share` |
| 2024-05 | Mamba-2 / state-space duality | arXiv 2405.21060 | `--attention mamba2`, `mamba_nano.py` |
| 2024-08 | Logit softcapping; RMSNorm sandwich adoption | Gemma 2, arXiv 2408.00118 | `--attention softcap`, `--logit-softcap`, `--norm sandwich`, `gemma_nano` |
| 2024-08 | Aux-loss-free load balancing | arXiv 2408.15664; DeepSeek-V3 | `deepseek_nano --balance-speed` |
| 2024-09 | Hyper-connections | ByteDance, arXiv 2409.19606 | precursor of mHC below |
| 2024-10 | Differential attention | arXiv 2410.05258 | `--attention diff` |
| 2024-10 | p-RoPE (partial RoPE) | Barbero et al., arXiv 2410.06205; Gemma 4 | `--posenc prope`, `gemma_nano` |
| 2024-12 | Gated DeltaNet | arXiv 2412.06464 | `--attention deltanet`, default `--linear` |
| 2024-12 | Sigmoid routing, MTP and aux-free balancing at scale | DeepSeek-V3, arXiv 2412.19437 | `deepseek_nano --router sigmoid` |
| 2024-12 | Muon optimizer | Keller Jordan, blog post (2024-12); *Muon is Scalable*, arXiv 2502.16982 (2025-02) | `nano/optim.py`, `--optim muon` |

## Convergence, 2025

| Date | Idea | Paper / origin | Here |
|---|---|---|---|
| 2025-01 | Lightning attention at 456B | MiniMax-01, arXiv 2501.08313 | `--attention lightning` |
| 2025-01 | Scalable softmax | Nakanishi, arXiv 2501.19399 | `--attention ssmax` |
| 2025-02 | Depth recurrence revived | Huginn, arXiv 2502.05171; Ouro, arXiv 2510.25741 (2025-10) | `looped_nano.py` |
| 2025-02 | Mixture of block attention | MoBA, arXiv 2502.13189 | `--attention moba` |
| 2025-03 | 5:1 local:global, dual RoPE base, QK-norm over softcap | Gemma 3, arXiv 2503.19786 | `gemma_nano.py` |
| 2025-05 | Per-layer embeddings | Gemma 3n, announced Google I/O 2025-05-20; Gemma 4 report | `qwen_next_nano --ple-dim`, `components.py` |
| 2025-07 | Short convolutions as a mixer | LFM2, released 2025-07-10, report arXiv 2511.23404 | `--short-conv` (`ShortConv` in the zoo) |
| 2025-07 | MuonClip in production (related adoption) | Kimi K2, released 2025-07-11, [arXiv 2507.20534](https://arxiv.org/abs/2507.20534) | `--optim muon` implements Muon + auxiliary AdamW; MuonClip's QK clipping is not implemented |
| 2025-08 | Learned attention sinks, 128-token windows alternating with full | gpt-oss, released 2025-08-05 | `--attention sink`, `swa` + `attn_sink` |
| 2025-09 | Gated attention, 3:1 Gated DeltaNet hybrid | Qwen3-Next, released 2025-09-12 | `--attention gated`, the hybrid's default layout |
| 2025-09 | DeepSeek sparse attention, lightning indexer | DeepSeek-V3.2-Exp, [release and technical report, 2025-09-29](https://api-docs.deepseek.com/news/news250929/); later DeepSeek-V3.2 report, arXiv 2512.02556 (2025-12) | `--attention dsa`, `qwen_next_nano --dsa-top-k` |
| 2025-10 | Kimi Delta Attention, KDA + gated MLA | Kimi Linear, arXiv 2510.26692 | `--attention kda`, `configs/kimi_like.json` |
| 2025-12 | Mamba-2 + attention + MoE hybrid | Nemotron 3, arXiv 2512.20856 | `qwen_next_nano --linear mamba2` |
| 2025-12 | Manifold-constrained hyper-connections | mHC, DeepSeek, arXiv 2512.24880 | `--residual mhc`, `components.py` |

## 2026

| Date | Idea | Paper / origin | Here |
|---|---|---|---|
| 2026-01 | Engram conditional memory | DeepSeek, arXiv 2601.07372 | `qwen_next_nano --engram-dim`, `components.py` |
| 2026-02 | Gated attention + Gated DeltaNet at 397B | Qwen3.5, released 2026-02-16 | the hybrid's default layout |
| 2026-02 | Linear attention + sparse MoE at "Flash" size | Qwen3.5-Flash, released 2026-02-25 | the hybrid's default layout, `deepseek_nano` MoE |
| 2026-03 | Mamba-3: trapezoidal step, complex state | arXiv 2603.15569 | `mamba_trapezoidal`, `mamba_complex`; `mamba_nano --discretization trapezoidal --complex` |
| 2026-04 | LatentMoE | Nemotron 3 Super, arXiv 2604.12374 | `deepseek_nano --moe-latent-dim` |
| 2026-06 | Short conv vs. short window, the open argument | *Dynamic Short Convolutions*, arXiv 2606.03825 | note in the ShortConv entry |
| 2026-06 | IndexShare: one indexer per layer group | MiniMax Sparse Attention, arXiv 2606.13392 | `qwen_next_nano --index-share` |
| 2026-06 | Lightning attention + MLA hybrid | Ling 2.6, arXiv 2606.15079 | `--attention lightning`, `--attn mla` |
| 2026-06 | Compressed sparse / heavily compressed attention; mHC shipped | DeepSeek-V4, arXiv 2606.19348 | `--attention csa`, `--attention hca`, `--residual mhc` |
| 2026-07 | K reused as V, p-RoPE and PLE shipped, KV sharing, last layer always global | Gemma 4, arXiv 2607.02770 | `--attention kv1`, `gemma_nano.py`, `--kv-share`, `--ple-dim` |
| 2026-07 | V4 at 284B/13B: CSA+HCA, mHC ×4, hash-routed bootstrap MoE, Muon, K=V MQA with per-head sinks | DeepSeek-V4-Flash, released 2026-07-31 | `csa`, `hca`, `--residual mhc`, `--optim muon`, `kv1`, `sink`; routing trio via `--hash-layers --router sqrtsoftplus --swiglu-limit`, preset `configs/deepseek_v4_flash.json` |
| 2026-08 | 3:1 GDN + Qwen Sparse Attention (4-token micro-block indexer, top-512 blocks), 51B n-gram embedding at layer 2, 512-expert MoE top-10 | Qwen3.8-Flash-Next, released 2026-08; arXiv 2608.30320 | `--attention qsa`, `--attn qsa` in the hybrid, `--engram-dim`; preset `configs/qwen_flash_next.json` |

## Reading the columns

- **Zoo entries** (`--attention <name>`) are in `nano/attention_zoo.py` and run in
  `gpt_nano`; the linear ones also fill the hybrid's cheap slot via `--linear`.
- **Hybrid flags** are on `nano/models/qwen_next_nano.py`, with the mechanisms
  in `nano/components.py`.
- **Whole-model files** — `gemma_nano.py`, `mamba_nano.py`, `looped_nano.py`,
  `deepseek_nano.py` — bundle the components a family actually shipped together.

Where a paper and a shipping model differ by a year or more (MTP, CLA, p-RoPE,
PLE, Muon), both dates are given: the idea's, and the release that made it
standard.
