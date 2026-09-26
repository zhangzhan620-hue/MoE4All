# Yandex AliceAI-Foundation (`aliceai`) support

`general.architecture = "aliceai"` covers Yandex's AliceAI-Foundation family, e.g.
`AliceAI-Foundation-80B-A3B-Base`. It is a hybrid recurrent/attention MoE whose per-layer graph is
**not** the shared transformer skeleton: three of every four layers run a **KDA** (Kimi delta-rule)
recurrent mixer, the fourth is GQA full attention, and there is no plain residual stream — the trunk
is a **depth-residual softmax mixing** over completed blocks plus a running partial. Every layer
carries a 512-expert sigmoid MoE with a gated shared expert.

The authoritative summary of the arch lives in [`crates/infr-llama/src/arch.rs`'s `ALICEAI`
doc](../crates/infr-llama/src/arch.rs); the config fields are on `Config` in
[`crates/infr-llama/src/config.rs`](../crates/infr-llama/src/config.rs); the graph wiring is in
[`crates/infr-llama/src/seam/runner.rs`](../crates/infr-llama/src/seam/runner.rs).

## Model shape (`AliceAI-Foundation-80B-A3B-Base`)

| metadata key (`aliceai.` prefixed) | value |
| --- | --- |
| `block_count` | 48 |
| `embedding_length` | 2048 |
| `attention.head_count` | 16 |
| `attention.head_count_kv` | per-layer array, `0` on KDA layers / `2` on full-attention layers |
| `attention.key_length` / `attention.value_length` | 256 |
| `rope.dimension_count` / `rope.freq_base` | 64 / `1e6` |
| `kda.head_dim` | 128 |
| `ssm.time_step_rank` | 32 (the KDA key/value head count) |
| `ssm.conv_kernel` | 4 |
| `expert_count` / `expert_used_count` | 512 / 10 |
| `expert_feed_forward_length` / `expert_shared_feed_forward_length` | routed / shared width |
| `expert_gating_func` | 2 (sigmoid) |
| `expert_weights_norm` / `expert_weights_scale` | `true` / `1.0` |
| `attn_res.block_size` | 4 |

## Period-4 KDA / full-attention split

Layer `il` is **full attention** iff `(il + 1) % 4 == 0`; the other three are **KDA**. The
authoritative per-layer marker is **not** a modulo in the loader — it is the GGUF's
`attention.head_count_kv` **array**: `0` marks a recurrent/KDA layer, `2` marks a full-attention
layer. `Config::from_gguf` reads the array into `alice_full_attn`, takes its max (2) as the
full-attention `n_kv`, and `Config::is_alice_attn_layer` / `Config::is_recurrent_layer` expose the
per-layer choice. `kda_n_head` (`ssm.time_step_rank`, 32) is the KDA head count and is distinct from
the attention head count (`n_head`, 16); the KDA conv/state widths derive from it
(`Config::recurrent_conv_channels` = `3 * kda_n_head * kda_head_dim`,
`Config::recurrent_state_elems` = `kda_n_head * kda_head_dim * kda_head_dim`).

## Depth-residual softmax mixing

There is no `hidden += sub` residual stream. Instead the trunk keeps:

- `alice_srcs` — a contiguous `[n_src, batch, ne]` f32 stack of "completed block" residuals, slot 0
  being the raw embeddings;
- `alice_partial` — the running accumulator for the block currently being built;
- `alice_mix` — one mix's output, feeding either `attn_norm` or `ffn_norm`.

The rhythm per layer:

1. **Block boundary** (`l > 0 && l % alice_block == 0`): commit the just-finished `partial` into
   `alice_srcs`, then advance the block count.
2. **Attention input**: layer 0 reads the raw embeddings; every later layer
   `Op::AttnResMix`es the completed sources plus the live `partial` through the layer's
   `attn_res_score`, then `RmsNorm(attn_norm)`.
3. **After the mixer**: `partial += sub` (or seed `partial = sub` on the block's first layer), copy
   `partial` into the next `alice_srcs` slot, `Op::AttnResMix` through `ffn_res_score`, then
   `RmsNorm(ffn_norm)` and the FFN.
4. **Model head**: commit the final `partial` and one last `Op::AttnResMix` through
   `output_res_score`, before `output_norm`.

`Op::AttnResMix` scores each source **weightlessly**: for source `s`,
`sc[s] = Σ_c score[c]·src[c] / sqrt(mean_c(src[c]²) + eps)`, then
`dst[r,c] = Σ_s softmax_s(sc)·src_s[c]`. The converter folds the per-source RMSNorm weight INTO the
`*_res_score` vector (`res_score = res_proj ⊙ res_norm_weight`, verified numerically), which is why
the op's normalization has no weight operand. `n_src == 1` is the identity mix.

The layer-0 `attn_res_score` is absent (it would mix a single source); `ffn_res_score` is present on
every layer, and `output_res_score` once at the head. `LayerW::alice_attn_res` is therefore `None`
on layer 0 and `Some` elsewhere; `alice_ffn_res` is always `Some`.

## KDA softplus decay

The recurrent layers are Kimi-delta-rule attention with a depthwise q/k/v causal conv
(`ssm_conv1d`, kernel 4, SiLU). Each state row gets its own log-decay:

```text
decay[h,k] = exp(ssm_a[h] * softplus(alpha[h,k] + dt_bias[h,k]))
```

The GGUF's `ssm_a` already holds the **converter-prepared** coefficient `-exp(A_log)` (the
source converter's `conversion.json` records "transform KDA log-decay parameter"), so the runtime
multiplies it in directly and must **not** re-exponentiate. This is the `softplus = true` mode of
[`crates/infr-core/src/graph.rs`](../crates/infr-core/src/graph.rs): `a` is used as-is,
`lower_bound` is ignored, and `beta = sigmoid(b_proj)` with no lower bound and no `*2`. It is
distinct from Ling's form (`bailingmoe3`), which has no softplus:
`decay[k] = exp(lower_bound * sigmoid(a[h] * (forget[k] + dt_bias[k])))`. The `softplus` selector is
what keeps the two archs on one op instead of silently sharing the wrong decay.

The KDA alpha (`ssm_f_a` → `ssm_f_b`) and output gate (`ssm_g_a` → `ssm_g_b`) are **low-rank**
projections on aliceai, so the loader keeps them as four separate handles and the runner emits the
two-stage projection; the KDA output norm (`ssm_norm`) is a weighted RMSNorm and is the one norm
that stays **raw** (not zero-centered). The Q/K/V projections ship as separate tensors (often
different quants), packed into `Op::Kda`'s `qkv` via `CopyStrided`.

## Full attention

The every-4th layer is GQA (16 q heads / 2 kv heads, `head_dim` 256) with:

- **Interleaved q + output gate** in `attn_q` (`[h0 q(hd) | h0 gate(hd) | h1 q | h1 gate | …]`, not
  two contiguous blocks) — the same trap as Qwen3.5 (`Config::attn_out_gate`). The query is read
  directly out of the interleaved buffer with a stride by `Op::QkNormRope`, and the gate re-enters
  through `Op::GatedAct` (sigmoid) applied to the attention output **before** the o-projection.
- **Partial NEOX rope** — `rope.dimension_count` is 64 (a quarter of `head_dim` 256) at theta `1e6`.
  `Op::QkNormRope` rotates the first `rope_dim` dims of each head with **NEOX** (split-half) pairing
  and passes the tail through **normed**, so the partial rope needs no extra split.
- **Zero-centered q/k RMSNorm** — `attn_q_norm` / `attn_k_norm` are baked by the converter as
  `1 + weight` and consumed as ordinary weighted RMSNorm; the same is true of
  `attn_norm`/`ffn_norm`/`output_norm`. `ssm_norm` is the sole exception.

## MoE

Every layer is a routed MoE, not a dense-lead schedule:

- `Op::MoeFfn` with `MoeGating::Sigmoid` (`expert_gating_func = 2`);
- **router bias** `exp_probs_b` — the selection score is the biased score, the returned weight comes
  from the unbiased probability;
- **top-k renormalization** (`expert_weights_norm = true`), routed scale `1.0`;
- a **gated shared expert** (`expert_shared_feed_forward_length`, `Config::shexp_gated`): a dense
  SwiGLU branch summed with the routed output through `Op::MoeSharedExpertAdd`, gated by a per-token
  sigmoid from `ffn_gate_inp_shexp`.

## GGUF tensor-name mapping

| role | GGUF name | notes |
| --- | --- | --- |
| input norm | `blk.N.attn_norm.weight` | zero-centered, baked to `1 + weight` |
| depth-residual (attn) | `blk.N.attn_res_score.weight` | absent on layer 0 |
| depth-residual (ffn) | `blk.N.ffn_res_score.weight` | every layer |
| depth-residual (head) | `output_res_score.weight` | before `output_norm` |
| full-attn Q (q + gate) | `blk.N.attn_q.weight` | `[n_embd, 2*n_head*head_dim]`, interleaved |
| full-attn K / V | `blk.N.attn_k.weight` / `blk.N.attn_v.weight` | GQA, `n_kv = 2` |
| full-attn q/k norm | `blk.N.attn_q_norm.weight` / `blk.N.attn_k_norm.weight` | zero-centered, baked |
| full-attn o-proj | `blk.N.attn_output.weight` | applied after the sigmoid gate |
| KDA Q / K / V | `blk.N.attn_q.weight` / `attn_k.weight` / `attn_v.weight` | separate tensors |
| KDA conv | `blk.N.ssm_conv1d.weight` | depthwise, kernel 4, SiLU |
| KDA low-rank alpha | `blk.N.ssm_f_a.weight` / `blk.N.ssm_f_b.weight` | `f_b(f_a(x))` → `forget` |
| KDA output gate | `blk.N.ssm_g_a.weight` / `blk.N.ssm_g_b.weight` | `g_b(g_a(x))` |
| KDA beta | `blk.N.ssm_beta.weight` | per-head |
| KDA decay | `blk.N.ssm_a` / `blk.N.ssm_dt.bias` | pre-converted `-exp(A_log)`; used as-is |
| KDA output norm | `blk.N.ssm_norm.weight` | weighted (not zero-centered) |
| KDA o-proj | `blk.N.attn_output.weight` | shared name with the full-attn o-proj |
| FFN norm | `blk.N.ffn_norm.weight` | zero-centered, baked |
| router | `blk.N.ffn_gate_inp.weight` | sigmoid gate |
| router bias | `blk.N.exp_probs_b.bias` | selection-only bias |
| routed experts | `blk.N.ffn_gate_exps.weight` / `ffn_up_exps.weight` / `ffn_down_exps.weight` | 512 experts |
| shared expert | `blk.N.ffn_gate_shexp.weight` / `ffn_up_shexp.weight` / `ffn_down_shexp.weight` | dense SwiGLU |
| shared-expert gate | `blk.N.ffn_gate_inp_shexp.weight` | per-token sigmoid logit |

## Backend status

- **CPU and Vulkan are implemented.** `Op::AttnResMix` has a CPU interpreter arm and a Vulkan
  `attn_res_mix.comp` kernel, and the `softplus` mode of `Op::Kda` is handled on both (the Vulkan
  KDA shader takes the `softplus` flag in its push constants).
- **Metal is not implemented.** There is no `AttnResMix` kernel and no KDA-softplus handling on
  Metal; a Metal run of an aliceai model does not build. The gated shared expert is also absent
  there (`Op::MoeSharedExpertAdd` is CPU + Vulkan only).

## Verification

Everything is synthetic — the real 80B-A3B checkpoint is not exercised in-tree.

- `synthetic_aliceai_generates` in
  [`crates/infr-llama/tests/synthetic_deepseek2.rs`](../crates/infr-llama/tests/synthetic_deepseek2.rs)
  builds a tiny structurally complete aliceai GGUF (period-4 KDA/full-attention layers, block-size-4
  depth-residual mixing, zero-centered norms baked to `1 + weight`, a sigmoid MoE with router bias +
  gated shared expert), drives the **real** `wload`/`wpush` lockstep and the **real** emit, and
  asserts the CPU logits are finite and non-degenerate.
- `synthetic_aliceai_vulkan_generates` is its Vulkan twin (`#[ignore]`d, needs a device): it
  exercises the new `attn_res_mix` and KDA-softplus shader paths, asserts finite/non-degenerate
  logits, requires CPU and Vulkan to agree on the top token, and bounds CPU-vs-Vulkan cosine at
  `> 0.99` — looser than the deepseek2 twin's `0.9999` because the synthetic model runs 8 layers of
  KDA recurrence through f16 KV staging.
- Op-level CPU tests in [`crates/infr-cpu/src/lib.rs`](../crates/infr-cpu/src/lib.rs):
  `kda_softplus_decay_matches_reference` pins
  `decay = exp(a·softplus(forget + dt_bias))` against a hand computation, and
  `attn_res_mix_softmax_weights_sources` pins the weightless-RMSNorm/softmax mix.
