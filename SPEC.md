# CircuitAtlas: a mechanism dictionary learned from per-token attribution-graph edge flows

**Repo:** `github.com/asystemoffields/interp-big` (currently empty) · **Package:** `circuitatlas` · **CLI:** `catlas` · **License:** MIT
**Spec date:** 2026-10-15 · **Audience:** an autonomous coding agent building the repo from scratch in a cloud sandbox (CPU, optionally one 24–48 GB GPU)

---

## 0. Operating rules for the build agent (read first)

1. **Pre-register, then run.** Before any Tier-2 (real-LM) experiment, freeze `docs/PREREGISTRATION.md` (Section 7 of this spec, with your concrete thresholds), commit it, and record its commit hash in the README. After that, never change a threshold. Log every deviation in `docs/DEVIATIONS.md` with a reason.
2. **Report honestly either way.** Every primary endpoint gets PASS, FAIL or NOT-RUN in `results/summary.json`. The README results table is generated from that file and never hand-edited. Negative results are first-class deliverables (Section 7.4).
3. **Never fabricate numbers, citations or model/dataset IDs.** If a resource such as a transcoder repo, a dataset or a code release cannot be loaded, record that and use the documented fallback.
4. **Locked test split.** Split by document into train/val/test. K, λ and k are chosen on val only. The test split is used once per configuration, for the final numbers.
5. **Compute gates.** Before each tier, run `catlas cost-probe` and follow the reduction order in Section 10 if the projection exceeds budget. Log the decision in `docs/COMPUTE_LOG.md`.
6. **Tests gate milestones.** A milestone is done only when its validation check (Section 9) passes in CI or in a logged run.
7. **Ship early.** Tag `v0.1` as soon as Tier 1 (ground truth) is finished, whatever the outcome. Several groups are converging on adjacent ideas.

---

## 1. Pitch

Attribution graphs show how a model computes one token on one prompt, and nothing more global than that. "Can we merge attribution graphs, or otherwise obtain a more global circuit picture?" is an open problem listed in the multi-lab Circuits Research Landscape (https://www.neuronpedia.org/graph/info).

CircuitAtlas answers it by moving polygenic mechanism decomposition into attribution-graph edge space. The decomposition lineage is NPEFF and L3D: non-negative, additive, sparse decompositions of per-example attributions. CircuitAtlas applies it to edge space as follows:

- **Edge flows per token.** For every token in a corpus, it computes how much of the correct-token logit margin flows through each transcoder-feature → feature (or → output) edge, with positions collapsed.
- **Shared coordinates.** Transcoder features give every token the same coordinate system, so these sparse edge-flow vectors from different inputs can be compared directly.
- **Dictionary.** It learns a non-negative sparse dictionary over the vectors. Each dictionary atom is a readable, feature-level subgraph that many tokens reuse, and a token's computation is a sparse sum of atoms, so mechanisms can co-occur.
- **Validation.** Atoms are checked in the real model with live attention and live nonlinearities. They are compared against ground truth in planted-mechanism transformers and against strong nulls, chiefly a node-attribution dictionary and NPEFF.

The output is a census instrument: coverage curves of how much of a model's next-token computation K causally checked mechanisms account for, with an explicit dark-matter ledger. It also enables mechanism-level tests of the quantization model of scaling, of universality across model sizes, and of what instruction tuning changes.

**Scaling path.** Extraction is amortized per sequence and the dictionary is learned by streaming, as with SAEs. A stretch "mechanism lens" (Section 13) predicts atom activations directly from activations, so mechanisms can be monitored at forward-pass cost.

---

## 2. Positioning: what is new and what is not

### 2.1 Lineage (cite all of these in README and paper)

| Work | Unit discovered | Space | Polygenic (co-active)? | Atoms readable as feature circuits? | Relation to CircuitAtlas |
|---|---|---|---|---|---|
| QDG, Michaud et al. 2023 https://arxiv.org/abs/2303.13506 | token clusters ("quanta") | per-token parameter gradients | No: one cluster per token | No | Secondary baseline; quanta theory we test |
| Sparse Feature Circuits, Marks et al. https://arxiv.org/abs/2403.19647 | behaviors, then circuits | gradient clusters → SAE circuits | No | Per behavior, hand-picked | Secondary baseline |
| NPEFF, Matena & Raffel (v3 June 2026) https://arxiv.org/abs/2310.04649, code https://github.com/mmatena/npeff_torch | non-negative "processing strategies" | per-example Fisher | Yes, and argues clustering "essentially guarantees the recovery of only monogenic factors" | No: building circuits is explicitly left to future work | **Primary baseline.** We take the step NPEFF leaves open |
| L3D, Chrisman, Bushnaq & Sharkey 2025 https://arxiv.org/abs/2504.00194 | low-rank parameter subnetworks | per-sample loss-gradient reconstruction | Yes | Partially (parameter directions) | Baseline (stretch) |
| Susceptibilities, Baker et al. 2025 https://arxiv.org/abs/2504.18274; Embryology https://arxiv.org/abs/2508.00331 | token × component response modules | coarse components (heads/MLPs) | Low-rank | No | Susceptibility-style proxy baseline |
| APD https://arxiv.org/abs/2501.14926 · SPD https://arxiv.org/abs/2506.20790 · SPD on small transformers https://arxiv.org/abs/2511.08854 · tPD https://arxiv.org/abs/2607.13047 · ASPD https://arxiv.org/abs/2609.37731 · VPD https://www.goodfire.com/research/interpreting-lm-parameters | parameter components | weights | Yes | In weight space | Orthogonal route (weights vs. per-token graphs) |
| MOLT https://transformer-circuits.pub/2025/bulk-update/index.html; JSAE https://arxiv.org/abs/2502.18147 | per-layer computation dictionaries | one layer's transform | Yes | Within one layer | Disproves any "first SAE for computation" claim |
| CircuitLens, Golimblevskaia et al. https://arxiv.org/abs/2510.14936 | input clusters per feature | transcoder + head contributions | No (clustering) | Per feature | Already uses transcoders as shared coordinates; we go model-wide |
| ADAG https://arxiv.org/abs/2604.07615; probe prompting https://arxiv.org/abs/2511.07002; Circuit-Diff https://arxiv.org/abs/2609.23892; LLM-annotated graphs https://arxiv.org/abs/2608.02632; HAGD https://arxiv.org/abs/2601.12879; ILP signatures https://arxiv.org/abs/2605.21303; concept-targeted attribution https://arxiv.org/abs/2608.27510; circuit-motifs https://github.com/mkenney2/circuit-motifs | supernodes, motifs, per-graph annotations | within one graph or across a few | n/a | Yes, but no corpus dictionary | ADAG-style profiles inform our role vocabulary |
| Weights-only global maps: TWERA/global weights (Ameisen et al.) https://transformer-circuits.pub/2025/attribution-graphs/methods.html, Circuits Updates May 2026 https://transformer-circuits.pub/2026/may-update/index.html, Communication Map https://arxiv.org/abs/2608.22007 | averaged virtual weights | weights | n/a | Yes, but averaged, not decomposed | We use expected virtual weights for the node→edge null |
| Vision precedents: SpRAy https://arxiv.org/abs/1902.10178, PCX https://arxiv.org/abs/2311.16681 | clusters of per-sample attributions | input attributions | No | No | Precedent for "global strategies from local attributions" |
| Many Circuits, One Mechanism https://arxiv.org/abs/2606.06267 | equivalence classes of circuits | — | — | — | Motivates functional-equivalence merging (Section 4.12) |

### 2.2 Novelty statement (use this wording, adjusted only for results)

> To our knowledge (literature checked through 2026-10-15; LessWrong/Alignment Forum coverage incomplete), no published method learns a sparse dictionary over per-token attribution-graph edge-flow vectors to produce a corpus-level, loss-accounted catalog of mechanisms. CircuitAtlas moves NPEFF/L3D-style polygenic decomposition into transcoder edge space, so each component becomes a readable feature-level subgraph that can be intervened on in the real model. This is the step NPEFF leaves to future work.

**Claims the agent must not make:**
- "first SAE for computation" (MOLT and JSAE exist);
- "transcoders as shared coordinates is our key insight" (CircuitLens);
- "first test of the quantization model", unless the Pythia acquisition test (Section 6.5) is actually run, and even then, "first test using causally checked circuit-level units";
- "a heavy-tailed frequency distribution validates the quantization model".

---

## 3. Research questions and hypotheses

**RQ.** Over a model's natural input distribution, does next-token computation decompose into a sparse dictionary of reusable, feature-level subgraphs ("mechanisms")? Specifically:
- (a) Do these recover planted mechanisms, including co-active ones?
- (b) Do they carry information beyond node-level attributions plus static wiring?
- (c) Do they predict where real-model interventions do damage, better than strong nulls?
- (d) How much of the computation do they cover, and what is left over?

**Hypotheses.** All are falsifiable; thresholds are in Section 7.

- **H1, ground-truth recovery.** In transformers trained on a language with planted, Zipf-weighted, co-occurring mechanisms, edge atoms recover the learned planted mechanisms, including tokens where two mechanisms are co-active, and do so better than QDG, NPEFF and node-attribution dictionaries.
- **H2, edges beat nodes.** On real LMs, edge-flow codes explain held-out edge flows better than node-attribution codes given an optimized edge decoder, and edge-atom knockouts concentrate damage better than node-atom knockouts at matched mass.
  - *Null:* edge flows are fully determined by node attributions, activations and static (expected) virtual weights, so an edge dictionary reduces to a node dictionary composed with fixed wiring.
- **H3, causal carving.** Atom knockouts in the real model are selective. Top-coefficient tokens are damaged far more than random tokens, at matched effect size, compared with NPEFF and equal-mass random edge sets. Atom coefficients also predict damage beyond raw edge presence (shared-edge specificity).
- **H4, sufficiency and coverage.** At most about 10 active atoms per token retain most of the model's improvement over a bigram baseline. A dictionary with K ≤ 2048 atoms covers most of the in-vocabulary edge-flow mass on held-out tokens. The dark matter (error nodes, pruning, truncation, out-of-vocabulary edges, attention-pattern effects) is reported explicitly.
- **H5, content vs. algorithm.** Keying edges by feature families and relational attention attributes (the "role vocabulary") makes known content-general mechanisms (induction, copying, successor, name-moving) collapse into a few atoms with low mutual information with token identity. The feature-keyed vocabulary splits them by content.
- **H6, quanta (exploratory, calibrated).** Across Pythia sizes and training checkpoints, mechanism-defined token groups are learned abruptly and in order of atom frequency, more sharply than frequency-matched random token groups. Gains between model sizes concentrate on tokens whose atoms are new at the larger size. The alternative, smooth co-improvement as in the superposition account of Liu, Liu & Gore https://arxiv.org/abs/2505.10465, predicts no excess sharpness over matched nulls.
  - This test is valid only if the same analysis code classifies the toy positive control (independent planted lookups) as quantized and the toy negative control (a shared-mechanism world) as not.

---

## 4. Method

### 4.1 Replacement model and linearization

- **Model and transcoders.** Model M has L layers and residual stream x^ℓ_p ∈ ℝ^d. Each MLP is paired with a per-layer transcoder (PLT):

  T^ℓ(u) = Σ_i a^ℓ_i(u) d^ℓ_i + W^ℓ_skip u + b^ℓ, with a^ℓ_i(u) = σ(w^{ℓ,enc}_i · u + b^{ℓ,enc}_i).

  σ is JumpReLU, TopK or ReLU, and W_skip is optional. Cross-layer transcoders (CLTs) are a stretch goal, SG4.
- **Spliced model.** MLP^ℓ(u) = T^ℓ(u) + ε^ℓ_p, where ε is the error node. This reproduces M exactly.
- **Linearization at the clean input**, following Ameisen et al. https://transformer-circuits.pub/2025/attribution-graphs/methods.html and https://github.com/safety-research/circuit-tracer:
  - freeze attention patterns;
  - freeze normalization denominators (LayerNorm/RMSNorm scale, including Gemma's pre/post norms and QK-norm);
  - freeze the transcoder active set: an active feature's activation equals its pre-activation (exact at the clean point for JumpReLU, ReLU and TopK);
  - freeze every other elementwise nonlinearity (e.g., logit soft-caps) as a value-preserving multiplicative gate;
  - treat error terms and token embeddings as constant sources.
  
  The system is then exactly linear and reproduces clean values at the clean point.
- **Notation:**
  - F = active feature nodes (ℓ, p, i).
  - Sources Σ = {EMB_p} ∪ {ERR_{ℓ,p}} ∪ {BIAS}. Source "value" ≡ 1, and the column holds the source's contribution.
  - P_{rs} = the direct effect of node s on the pre-activation of node r. "Direct" means paths through the residual stream, frozen attention OV, skip connections and frozen norms only, never through another feature.
  - For r ∈ F: v_r = Σ_{s ∈ F ∪ Σ} P_{rs} v_s. P is strictly lower-triangular in layer order.

### 4.2 Target functional

For target position t with next token y = x_{t+1}, the **logit margin** is

m_t = z_{t,y} − (1/|V|) Σ_v z_{t,v}.

The margin does not vanish on confidently predicted tokens, unlike the loss gradient. m_t = Σ_s P_{ts} v_s, where the target row uses the frozen final norm and the unembedding direction u_y − ū.

- **Ablation target:** log-prob of y, with QDG-style per-token normalization.
- **Loss accounting** is always relative to a count-based **bigram baseline** fitted on the train split, with add-k smoothing: the "bigram→model CE gap" is CE_bigram − CE_model.

### 4.3 Influence, edge flows and conservation

- **Total influence:** τ^{(t)}_r = ∂m_t/∂v_r, propagated through the linear system. It is computed for all nodes with one vector-Jacobian product (VJP) of m_t through the linearized replacement. In row form, τ^{(t)} = P_{t,:}(I − P)^{-1} on features. For sources, τ_s = Σ_r P_{rs} τ_r + P_{ts}.
- **Node attribution:** n^{(t)}_r = v_r τ^{(t)}_r.
- **Edge flow:**
  - φ^{(t)}_{s→r} = v_s P_{rs} τ^{(t)}_r for r ∈ F;
  - φ^{(t)}_{s→OUT} = v_s P_{ts}.
  
  The edge flow is the total of all source→target path products that pass through that edge.
- **Exact identities.** These are unit-tested to 1e-5 relative error:
  - (I1) inflow: Σ_s φ_{s→r} = n_r for r ∈ F (BIAS included as a source);
  - (I2) outflow: Σ_{r∈F} φ_{s→r} + φ_{s→OUT} = n_s for s ∈ F ∪ Σ;
  - (I3) completeness: Σ_{s∈Σ} n_s = m_t.

So edge flows are a flow decomposition of node attributions. φ_{sr} = v_s P_{rs} τ_r makes plain how much of the edge vector node quantities already determine. Section 4.9 measures this directly rather than assuming it away.

### 4.4 Two-stage amortized extraction (per sequence)

**Input:** a sequence x_{1:T} (default T = 64) and target positions S (default |S| = 16, sampled uniformly from t ≥ 2; BOS is never a target).

1. **Forward pass with cache:** attention patterns, norm scales, transcoder pre-activations and active sets, and error vectors.
2. **Stage A, target influences.** Run one batched VJP with |S| seed rows (one per target margin) through the linearized replacement. Gradient passes through active-feature gates, frozen attention values and frozen norms. Errors and embeddings are leaves. Read τ^{(t)} for every active node and source:
   - τ_s = d_s · g(ℓ_s, p_s) for features;
   - e_p · g(0, p) for EMB;
   - ε · g for ERR.
   
   Compute n^{(t)}.
3. **Node selection:** U_t = top-N_node active features by |n^{(t)}_r|. All sources are always kept.
4. **Stage B, direct-effect rows.** For r ∈ U = ∪_t U_t, run batched VJPs from r's encoder direction (through its frozen pre-MLP norm) back through the residual-only linear system, cutting at MLP/transcoder boundaries. This gives the full row P_{r,:} over all active nodes and sources, plus the target rows P_{t,:}. P does not depend on the target, so it is shared by all targets in the sequence. This is the key amortization.
5. **Flows:** φ^{(t)}_{s→r} for r ∈ U_t ∪ {OUT} and s ∈ U_t ∪ Σ.
   - Inflow to r from non-selected active sources is aggregated into a `PRUNED_ℓ` source key.
   - Outflow from s to non-selected targets is computed by subtraction (n_s − retained outflows) into a `PRUNED` target key.
   - Conservation (I1–I2) therefore stays exact after node pruning.
6. **Collapse positions to keys** (Section 4.5), then keep the top-E keys by |flow|. Record the dropped mass.
7. **Accounting scalars per target** (Section 4.6).

**Cost.** About (|S| + |U|) batched VJP rows per sequence. Stage-B rows only traverse attention, skip and norm paths, at roughly O(L(T²d + Td²)) per row. |U| ≤ |S|·N_node, and in practice it is much smaller because selections overlap. The real cost is measured in M5 and M11, not assumed.

**Implementation decision rule.** Prefer reusing circuit-tracer's `ReplacementModel` (MIT; TransformerLens backend, or nnsight for Gemma 3 with the `mwhanna/gemma-scope-2-transcoders-circuit-tracer` collection) for the frozen linearization wherever it supports the model, and implement Stages A/B on top of it. Otherwise implement `circuitatlas.linearize.LinearizedReplacement` on TransformerLens hooks (https://github.com/TransformerLensOrg/TransformerLens) for toy, TinyStories, GPT-2 and Pythia models, and on HF/nnsight (https://github.com/ndif-team/nnsight) for Gemma 3. Either way, M4 requires a numerical cross-check against circuit-tracer adjacency on a model it supports.

### 4.5 Edge vocabularies

**Edge key:** κ(s, r) = (ι(s), ι(r), ρ(s, r)), where ι is node identity without position:
- features: (layer, idx);
- `EMB:<token or class>`;
- `ERR:<layer>`;
- `BIAS`;
- `PRUNED_ℓ`;
- `OUT`.

**Relation ρ:**

| Vocabulary | ρ for same-position edges | ρ for cross-position (attention-mediated) edges | EMB keyed by |
|---|---|---|---|
| **V_feat (primary, pre-registered)** | SAME | ATTN | token id |
| **V_role** (H5) | SAME | ATTN(offset bucket ∈ {1, 2, 3–4, 5–8, 9–16, 17+}, `tok_match` = [x_p == x_q], `prev_match` = [x_{p−1} == x_q]) | token class |

- **V_role feature families:** per-layer spherical k-means on the concatenated normalized [decoder; encoder] vectors at granularity F_ℓ ∈ {16, 64, 256, all}. This forms a hierarchy; Matryoshka SAEs are a stretch alternative, https://arxiv.org/abs/2503.17547.
- **V_role token classes:** 64 k-means clusters of embedding vectors crossed with 4 frequency bands.
- **Relational predicates** use context tokens only, never y, to avoid leaking the label. `prev_match` is the induction signature.
- **Optional `head` attribute:** the head carrying the largest share of the edge's final-hop flow (stretch).
- The sweep over F_ℓ is the H5 content-vs-algorithm curve.
- **Vocabulary building:** keep keys with train-split count ≥ c_min, capped at V_max per sign by count. Out-of-vocabulary (OOV) mass is reported, never silently dropped.
- **Storage:** an `EdgeKey` codec packs keys into two int64 words (src kind/layer/idx, tgt kind/layer/idx, rel bits). It must be unit-tested for round-trips.

### 4.6 Accounting: the census's dark matter

Each target records:
- m_t and its reconstruction from (I3);
- **error share** Σ_{ERR}|n_s| / Σ_Σ|n_s|, signed and absolute;
- embedding share;
- bias share;
- **node-pruned share** (mass on `PRUNED` keys);
- **truncated share** (mass outside the top-E keys);
- **OOV share** (after vocabulary building);
- **QK share:** first-order attribution of m_t to attention-score changes in the full (unfrozen) model, Σ_{ℓ,h,q≤t,k≤q} |∂m_t/∂S^{ℓh}_{qk} · (S^{ℓh}_{qk} − mean_k S^{ℓh}_{qk})|, reported as a ratio to Σ|n|. Attribution graphs freeze attention, so QK-implemented computation such as induction matching is otherwise invisible. Kamath et al. https://transformer-circuits.pub/2025/attention-qk/index.html give the full QK-edge version; it is stretch goal SG3.

The **waterfall figure** has one bar per model: of the total attribution mass, the share in error nodes, bias, pruned, truncated and OOV, then what K atoms cover at k active. The QK ratio is shown alongside. This waterfall is the honest headline, ahead of any atom count.

### 4.7 Token representation for dictionary learning

- **Signed split:** x_t = [max(e_t, 0); max(−e_t, 0)] ∈ ℝ^{2V}_{≥0}. Inhibitory flows (for example negative name movers) are kept as their own coordinates.
- **Normalization:**
  - shape mode (primary): x̃_t = x_t / ‖x_t‖₁;
  - scale s_t = ‖x_t‖₁ is stored and used for mass-weighted coverage.
- **Atom frequency** = the fraction of tokens with c_tk > 0.1 · max_j c_tj. This threshold is pre-registered. A raw TopK count is never used, because it would make "polygenic" counts an artifact of k.

### 4.8 Dictionary learning

**Graph SAE (primary).**

- Encoder: c_t = BatchTopK(ReLU(W_enc x̃_t + b_enc)) with average k active (BatchTopK, https://arxiv.org/abs/2412.06410). Per-token sparsity therefore varies. W_enc x̃ is a sparse-dense matmul.
- Decoder: x̂_t = D c_t, with D ∈ ℝ^{2V×K}_{≥0}, columns unit-L2, enforced by projection after each step. There is no decoder bias.
- Loss:

  L = Σ_t ‖Ω^{1/2}(x̃_t − D c_t)‖² + λ_flow Σ_k ‖B_int(D⁺_k − D⁻_k)‖₁ + λ_aux L_AuxK

  - **Exact Gram trick:** ‖Ω^{1/2}(x − Dc)‖² = xᵀΩx − 2xᵀΩDc + cᵀ(DᵀΩD)c. Only rows of D on x's support enter the cross term, and DᵀΩD is K×K per step.
  - **Ω:** weight λ_out (default 4) on OUT-incident keys. This is the linearized causal term: it makes Σ_k c_tk o_k track the token's margin flow, where o_k is atom k's net OUT mass.
  - **Flow-conservation prior:** B_int is the node×key incidence matrix over internal feature identities (+1 for inflow, −1 for outflow). True per-token flows satisfy B_int e_t = 0 exactly after pruning (Section 4.4 step 5), so coherent mechanisms should approximately conserve flow. This is the path-coherence prior. λ_flow ∈ {0, 1e-3, 1e-2, 1e-1}, chosen on toy-tier val and then frozen.
  - **AuxK** dead-atom revival follows Gao et al. https://arxiv.org/abs/2406.04093.
- Initialization: k-means++ on token vectors.
- Optimizer: Adam (lr 1e-3), batch 4096 tokens, early stopping on val reconstruction.
- **Amortization gap:** also report non-negative least squares (NNLS)-refined codes on the encoder's support.

**NMF (same-family baseline).** sklearn `MiniBatchNMF` (Frobenius) on sparse X, with codes top-k-sparsified at inference to match L0. Online LDA (sklearn) on quantized mass is optional.

**Choosing K.** Report full curves over K ∈ {16 … 4096} (powers of two, per tier). K* is the knee of the held-out coverage curve, with a two-part MDL proxy reported alongside:

bits = Σ_k nnz(D_k)(log₂ 2V + 8) + Σ_t nnz(c_t)(log₂ K + 8) + Gaussian residual code.

No claim may depend on K* alone. Three seeds per main configuration.

**Atom diagnostics:**
- **path coherence:** the fraction of atom mass on edges lying on a source→OUT path within the atom's thresholded support graph;
- **conservation residual:** ‖B_int D_k‖₁ / ‖D_k‖₁;
- OUT mass o_k;
- layer span.

### 4.9 Edge-vs-node decomposition (the decisive null, H2)

1. **Node vectors.** y_t = position-collapsed, signed-split, normalized node attributions over feature identities and sources.
2. **Node-predictability of edges (R²_{node→edge}).** Predict each position-level flow from node quantities plus static wiring:
   - φ̂_{s→r} = n_r · (v_s P̄^ρ_{rs}) / Σ_{s'}(v_{s'} P̄^ρ_{rs'});
   - P̄^ρ is the corpus-expected direct effect per (feature pair, relation), computed TWERA-style on train;
   - collapse φ̂ to keys and report held-out explained mass.
   
   If this is ≥ 0.9, edges add little at this granularity. That is a clean finding in either direction.
3. **Equal-budget code comparison.** Train a node dictionary (same architecture, K, k) on y_t. Freeze its codes c^node_t and fit a non-negative edge decoder D′ (projected gradient on train) so that x̃_t ≈ D′ c^node_t. Compare held-out unexplained edge mass for edge codes vs. node codes. This asks whether node-defined token groupings explain edge flows as well as edge-defined ones.
4. **Causal comparison.** Node atoms are knocked out two ways:
   - (α) node ablation, removing all outgoing contributions of the atom's top features on active tokens;
   - (β) induced-edge ablation, removing only edges between the atom's top features.
   
   Compare with edge-atom knockouts at matched corpus mass (Section 4.11).
5. **Weight-shuffled null.** Build flows with P_{r,π_ℓ(s)}, where π_ℓ is a fixed random permutation of source-feature ids within each source layer. Dictionaries on shuffled flows should lose path coherence and conservation. If they don't, the method is not using wiring.

### 4.10 Interventions in the real model

**Responsibility.** Atom k's share of key κ on token t (computed separately on the positive and negative halves):

ρ_tk(κ) = c_tk D_k[κ] / (Σ_j c_tj D_j[κ] + 1e-8), clipped to [0, 1].

**Edge knockout primitive.** Removing a fraction f of position-level edge s→r means subtracting f · A_{rs} from r's pre-activation, where A_{rs} = v_s P_{rs} is the frozen-estimate direct contribution. For OUT edges, subtract the propagated direct-path residual vector at position t before the final norm. This yields full-vocabulary logit changes, so cross-entropy (CE) and KL can be computed.

**Execution modes.** All are implemented and all are reported.
- **REAL-DELTA (primary):** run the real model (real MLPs, live attention, live norms). At each layer, add Σ_i [σ(pre′_i − δ_i) − σ(pre′_i)] d_i to the MLP output, where pre′ is the live pre-activation and δ is the subtracted contribution. With no intervention the output equals M exactly. Live JumpReLU thresholds can switch features on or off.
- **SPLICED-FROZEN-ERR:** the spliced transcoder model with errors frozen at clean values.
- **FROZEN-LIN (secondary only):** exact solution of the frozen linear system with the edited edge weights, (I − P′)⁻¹. This is the "linearized prediction". It is never the primary evidence, because using it as such would be circular.

**Intervention families:**
- **Token-specific atom knockout KO_t(k):** f = ρ_tk(κ) on every tracked edge with key κ.
- **Global atom knockout G(k):** a fixed, token-independent model edit. On every input, remove fully every position-level edge whose key lies in S_k (the minimal key set covering 80% of D_k's mass, with matching sign). It runs on the fly: for each forward pass, compute A_{rs} only for target features r ∈ tgt(S_k) at the positions where they are active (a batched VJP over those rows). This edit is comparable to NPEFF's and L3D's parameter perturbations.
- **Sufficiency KEEP_t:** f = 1 − Σ_{k active} ρ_tk(κ) on every tracked edge. Only active atoms' claimed flows survive; untracked mass is left untouched and reported.
- **Matched nulls:**
  - equal-corpus-mass random key sets (sampled to match S_k's train-split mass histogram over layers);
  - oracle per-token top-M edge pruning (M = edges claimed by active atoms);
  - node-atom knockouts (α) and (β);
  - NPEFF component perturbations from their code.

### 4.11 Causal metrics

Given a knockout, ΔKL_t = KL(p_clean ‖ p_edit) at t, and Δm_t is the change in margin.

- **Necessity faithfulness (S1, secondary):** Spearman(FROZEN-LIN predicted Δm, REAL-DELTA actual Δm) over (atom, token) pairs. This measures linearization quality, not atom validity. Also report the frozen-vs-live gap.
- **Matched-effect selectivity (P3).** Scale each global intervention's strength (edge fraction f, or NPEFF perturbation norm) so that the mean ΔKL on the atom's top-200 tokens by coefficient equals 0.1 nats. Then Sel_k = mean ΔKL(top) / mean ΔKL(random 2,000 held-out tokens). This is NPEFF's top-vs-random KL ratio, made comparable across interventions by matching the effect.
- **Damage concentration (P2c):** DC_k = the fraction of total held-out corpus ΔKL under G(k) that falls in the top 1% of tokens by c_tk. For random key sets and node atoms, rank tokens by their own presence or coefficient. Compare at matched corpus mass.
- **Shared-edge specificity, CPS (P3).** Bin held-out tokens by presence mass π_tk = Σ_{κ∈S_k}|e_t[κ]|. Within bins, compare the mean ΔKL under G(k) for tokens with c_tk in the top decile against tokens with c_tk = 0 (atom k's edges present, but claimed by other atoms). CPS_k = the bin-weighted ratio.
  - Interpretation, pre-registered: under a perfectly additive linear model, CPS ≈ 1. CPS > 1 means the atom captures non-additive structure, edges that matter jointly when the whole mechanism is engaged, which is evidence of carving. CPS ≈ 1 means atoms are a valid additive accounting but do not carve mechanisms.
- **Boundary test (secondary).** For atom pairs (k, j) sharing ≥ 20% of key mass:
  - G applied to k's exclusive keys should damage T_k (k active, j inactive) and spare T_j;
  - G applied to the shared keys should damage both;
  - report a 2×2 effect matrix per pair, against equal-mass random splits.
- **Sufficiency (P4):**
  - CE-gap retention = (CE_bigram − CE_KEEP) / (CE_bigram − CE_clean) on held-out tokens (each token gets its own intervention; batched forward passes);
  - margin recovery = (m^KEEP − m^KO-all-tracked) / (m^clean − m^KO-all-tracked);
  - both are compared with oracle top-M pruning, random equal-mass keep-sets and node-atom keep-sets.

### 4.12 Census well-definedness

- **Counts are curves, not numbers.** Report atom counts at fixed coverage levels (50/70/90%) as functions of K, k, transcoder width, transcoder L0, vocabulary (V_feat vs. V_role granularity) and merge threshold.
- **Functional-equivalence merging,** motivated by phantom specialization (https://arxiv.org/abs/2606.06267). Two atoms are merged into one mechanism if both of the following hold:
  - (a) their V_role projections have cosine ≥ τ_r, meaning they share structure up to feature families and relations;
  - (b) their **task signatures** match: the vector of task-metric drops under G(·) over a fixed probe battery (Section 6.3 tasks plus PlantedLM probes where applicable) has cosine ≥ τ_f.
  
  Merge by agglomerative clustering, and report counts over a grid of (τ_r, τ_f). This is explicitly heuristic.
- **Monogenic vs. polygenic tokens:** the effective number of atoms exp(H(c_t/‖c_t‖₁)), and its relation to token loss, token frequency and position.

### 4.13 Cross-dictionary and cross-model matching with noise floors

- **Atom token profile:** c_k ∈ ℝ^{N_eval} over a shared evaluation corpus. Models with the same tokenizer have the same token positions.
- **Matching:** atom k in dictionary A matches B if max_j corr(c^A_k, c^B_j) ≥ ρ*; also report Hungarian one-to-one matching.
- **Noise floors (required before any cross-model claim):**
  - (i) same model, same transcoder, different dictionary seeds (seed floor);
  - (ii) same model, different transcoder width or L0 (transcoder floor).
  
  ρ* is set at the 5th percentile of best-match correlations under floor (i). "Fraction matched across models" is always shown next to floor (ii) as its ceiling.
- **PT vs. IT:** run (A) IT transcoders on the IT model and (B) PT transcoders on the IT model, which keeps coordinates fixed (report PT-transcoder fidelity on IT first). Atoms "gained" or "lost" must be unmatched beyond floor (ii) in both A and B.
- **Optional interoperation:** export atoms to interp-lab's `compare`/`validate-matches` as a cross-check (Section 8.3).

### 4.14 Labels and interpretability evaluation

- **Atom card** (static HTML):
  - top-coefficient contexts with the target highlighted;
  - the atom subgraph (top keys as a layered DAG, with feature labels from max-activating examples, plus Neuronpedia or auto-interp labels where available);
  - OUT effect (which output tokens the atom promotes or suppresses);
  - relational attributes;
  - knockout summary;
  - path coherence;
  - frequency.
- **Blinded LLM judge (optional; skipped and logged if no API key).** It sees pairs of cards (edge atom vs. QDG cluster vs. NPEFF component vs. node atom, with matched budget and identities hidden) and rates whether the card describes one coherent mechanism. Simulation test: given only a method's description, predict held-out token membership on a balanced set; score = balanced accuracy.
- **Human ratings:** ≥ 50 pairs, done by the user, with a template provided.

---

## 5. Baselines (all at matched K and average L0 where meaningful)

| ID | Baseline | Role | Notes |
|---|---|---|---|
| B1 | **Node-attribution dictionary**, Graph SAE and NMF on y_t (Section 4.9) | **Primary null (H2)** | Plus induced-edge variant and edge decoder D′ |
| B2 | **NPEFF**, via https://github.com/mmatena/npeff_torch (G-NPEFF from per-token gradients if LRM-PEF for causal-LM tokens is impractical; follow the paper's definitions) | **Primary competitor (H1, H3)** | Use its perturbations for selectivity; record the exact variant |
| B3 | QDG: per-token loss gradients w.r.t. parameters, random-projected to 8,192 dimensions, cosine affinity, spectral clustering | Secondary | Subsample ≤ 20k tokens |
| B4 | k-means (spherical) on x̃_t | Secondary | Clustering, so monogenic by construction |
| B5 | Co-occurrence NMF on binary feature co-activation per token | Secondary | Tests "atoms = co-active features" |
| B6 | Susceptibility-style proxy: per-token attribution-patching effects of each head and MLP (component × token matrix), then NMF | Secondary | Coarse-component control; it is not Baker et al.'s SGLD method, so do not call it that |
| B7 | Component-edge dictionary: per-token edge attribution patching (EAP, https://arxiv.org/abs/2310.10348) over head/MLP→head/MLP edges, then Graph SAE | Secondary | Tests whether transcoder-feature resolution is needed |
| B8 | Equal-mass random key sets; weight-shuffled-wiring flows | Nulls | Section 4.9 step 5 and Section 4.10 |
| B9 | Oracle per-token top-M edge pruning | Upper reference for sufficiency | |
| B10 | L3D (https://arxiv.org/abs/2504.00194) | Stretch | Reimplement on toy and TinyStories only if time allows |

---

## 6. Experiments

### 6.1 Tier 0: correctness on synthetic linear DAGs (CPU, minutes)

Build random sparse layered linear systems with known node values. Check that:
- flows equal brute-force path enumeration;
- identities I1–I3 hold;
- pruning and aggregation preserve conservation;
- FROZEN-LIN knockouts match an explicit re-solve.

Also check dictionary recovery on synthetic x_t = sparse non-negative combinations of known non-negative atoms plus noise.

### 6.2 Tier 1: ground truth with PlantedLM (CPU-feasible; decisive)

**Generator (`circuitatlas.synth.planted`).** Vocabulary of about 600 tokens. Each sequence has length 64 and is a random interleaving of segments, so mechanisms co-occur within sequences. Every target carries a generator label G_t ⊆ {mechanisms}.

| Mechanism | Segment | Predictable target |
|---|---|---|
| M1 induction | span of 4–8 random word tokens, repeated later | tokens 2..n of the repeat |
| M2 bigram memory | chain x → f_b(x) over a 60-token domain | each next chain token |
| M3 topic / skip-trigram | topic marker T_i; later content drawn from topic subset | content tokens (set-valued; scored by in-topic probability mass) |
| M4 bracket matching | nested ( [ { with fillers | closing bracket type |
| M5 successor | N_a N_{a+1} N_{a+2} … | next number |
| M6 modular addition | N_a + N_b = | N_{(a+b) mod 23} |
| M7 variable binding | V_x := A_i … V_x ? | A_i |
| LOOKUP_j, j = 1..48 | L_j A_i | g_j(A_i), an independent random table per j; task frequency ∝ j^{−1.5} (planted Zipf, in the style of Michaud's quanta) |
| C1 polygenic (composite) | V_x := N_b … N_a + V_x = | needs M7 and M6 |
| C2 polygenic (redundant) | induction span drawn from a bigram chain | both M1 and M2 predict it |
| FILLER | random word tokens | none |

**Negative-control world (PlantedLM-Shared).** Identical, except g_j(A_i) = A_{(i + s_j) mod 16}: one shared shift mechanism parameterized by L_j, at the same Zipf rates. Prediction: few atoms, and per-task acquisition not ordered by frequency.

**Models and dictionaries:**
- **Toy model:** GPT-style, 4 layers, d = 256, 8 heads, d_mlp = 1024, pre-LN, GELU, learned positions, context 64. AdamW at 1e-3 with cosine schedule, 60M tokens (CPU default; 150M on GPU). Save 20 log-spaced checkpoints.
- **Width sweep** for the quanta calibration: d ∈ {64, 128, 256, 512}.
- **Behavioral probes:** per mechanism, 500 fresh instances. A mechanism counts as "learned" if top-1 accuracy is ≥ 0.9 (M3: in-topic mass ≥ 0.9). Recovery is scored only on learned mechanisms, which removes circularity about what the model actually learned.
- **Transcoders:** per-layer BatchTopK (k = 16), width 4096, 30M tokens. Gate: FVU ≤ 0.1 per layer, and splicing with errors zeroed costs ≤ 10% of the unigram→model CE gap.
- **Extraction:** 24k targets (20k train / 2k val / 2k test, split by sequence).
- **Dictionaries:** Graph SAE and NMF, K ∈ {16 … 512}, k ∈ {4, 8}.

**Metrics:**
- **Best-atom AUROC:** for each learned mechanism m, the best single atom (and best OR of ≤ 3 atoms) by AUROC of c_tk for m ∈ G_t.
- **Polygenic recall:** on C1 and C2 targets, the fraction where all constituent mechanisms' matched atoms are active. Clustering baselines are capped at 1/2 by construction.
- **Purity:** the coefficient mass of the matched atom that falls on mechanism m.
- **Knockout selectivity matrix:** G(k) for the matched atom vs. probe accuracy drop on every task. Diagonal dominance means the drop on its own task is ≥ 5× the mean drop on other tasks.
- **Planted-exponent check** (calibration, necessary but not sufficient): the exponent of the frequency of LOOKUP-matched atoms vs. the planted 1.5. Token frequency trivially carries the same exponent here, so this only checks the pipeline.
- **Quanta calibration:** the acquisition analysis of Section 6.5, run on PlantedLM (positive control) and PlantedLM-Shared (negative control) across the width sweep and checkpoints. It must classify them correctly.
- **Ablations:**
  - λ_flow ∈ {0, 1e-3, 1e-2, 1e-1};
  - λ_out ∈ {1, 4};
  - N_node ∈ {64, 128, 256};
  - E ∈ {150, 300, 600};
  - V_feat vs. V_role;
  - Graph SAE vs. NMF;
  - shape vs. mass weighting;
  - shuffled wiring.
- **Falsification:** if edge atoms do not beat QDG, NPEFF and the node dictionary on polygenic recall or AUROC (P1), the method's central claim fails in the setting most favourable to it. Stop making real-LM claims and publish the negative result.

### 6.3 Tier 2: real LM on CPU (TinyStories-33M; GPT-2 small as stretch)

**Primary model.** `roneneldan/TinyStories-33M` (GPT-Neo, 4 layers, d = 768, 16 heads; https://huggingface.co/roneneldan/TinyStories-33M; paper https://arxiv.org/abs/2305.07759).
- **Transcoders:** self-trained per-layer PLTs using `sparsify --transcode` (https://github.com/EleutherAI/sparsify) or the in-repo trainer. Width 12,288, k = 32, 50M tokens of TinyStories train, all 4 layers in one pass. Report FVU, error share and spliced CE.
- **Data:** TinyStories validation and train; 20k targets (16k / 2k / 2k by story).

**Stretch model.** GPT-2 small with public transcoders: `pchlenski/gpt2-transcoders` (Dunefsky et al. https://arxiv.org/abs/2406.11944, code https://github.com/jacobdunefsky/transcoder_circuits), or EleutherAI's sparsify-format GPT-2 transcoders (e.g., https://huggingface.co/EleutherAI/gpt2-mntss-transcoder-relu-sp6-skip, which has no model card; inspect its files before use). Corpus: `NeelNanda/pile-10k` or OpenWebText; verify availability and log the revision.

**Experiments:**
1. Cost probe (10 sequences), then full extraction.
2. Accounting waterfall (Section 4.6).
3. Coverage curves over K × k × {V_feat, V_role at 4 granularities}.
4. Edge-vs-node (Section 4.9): R²_{node→edge}, the equal-budget code comparison, causal comparisons and shuffled wiring.
5. Causal panel (Sections 4.10–4.11) on the top-200 atoms by mass plus 100 random atoms, with all nulls.
6. Sufficiency (P4).
7. **Known mechanisms (P6).** Pre-registered prompt batteries (≥ 200 prompts each), with matched control positions from natural text:
   - induction on repeated random-token sequences (https://arxiv.org/abs/2209.11895), plus natural repeated names;
   - name copying / IOI-style in TinyStories ("Lily and Tom went… Lily gave the ball to" → Tom), and full IOI on GPT-2 (https://arxiv.org/abs/2211.00593);
   - closing-quote and bracket matching;
   - successor (days, numbers, letters);
   - greater-than (GPT-2 only, https://arxiv.org/abs/2305.00586).
   
   The task atom set is the ≤ 3 atoms with the highest mean coefficient on task targets minus controls. Batteries are synthetic, so they are out of distribution for the atom encoder. That is part of the test.
8. Interpretability evaluation (S3) and NPEFF agreement (S5).

### 6.4 Tier 3: GPU (Gemma 3 270M, then 1B; PT and IT)

**Model and transcoders.** Gemma 3 270M PT with Gemma Scope 2 `transcoder_all` (every layer; https://huggingface.co/google/gemma-scope-2-270m-pt; cc-by-4.0). Use the small-L0 variant first. Load either via circuit-tracer's nnsight backend and the `mwhanna/gemma-scope-2-transcoders-circuit-tracer` collection, or via the in-repo adapter.

**Data.** 100k targets (80/10/10) from mixed web, code and math. Examples (verify and log revisions): a FineWeb-Edu sample, a permissively licensed code corpus and a math corpus.

**Experiments:**
1. Cost probe and go/no-go.
2. Full Tier-2 panel.
3. **Stability:** repeat extraction and census with ≥ 1 other L0 (medium) and, if `transcoder_all` offers it, ≥ 1 other width. Report coverage-curve and matched-atom stability against the seed floor (S4).
4. Gemma 3 1B PT, then 270M→1B matching with floors (X3).
5. PT vs. IT (X4), using the fixed-coordinate control of Section 4.13.
6. Stretch: QK edges (SG3) and CLT coordinates (SG4).

### 6.5 Tier 4: quanta acquisition test (GPU; exploratory, calibrated by Tier 1)

**Models.** Pythia-70m-deduped, 160m-deduped and, as a stretch, 410m-deduped (https://arxiv.org/abs/2304.01373). They share a tokenizer and data order, and each has 154 checkpoints.

**Transcoders.** No public all-layer Pythia transcoders were found (checked 2026-10-15). Train per-layer PLTs at the final checkpoint with sparsify. Eval corpus: `NeelNanda/pile-10k` (verify), with 50k targets per size.

**Tests:**
- **Q1, frequency-ordered acquisition across scale.** Atoms of the larger model that have no match in the smaller model (beyond floor (ii)) should have lower frequency than matched atoms (Mann–Whitney). The null comes from frequency-matched random splits.
- **Q2, concentration of improvement.** The per-token CE improvement from size n to size n+1 should concentrate on tokens whose active atoms are newly acquired. Measure the share of improvement vs. the share of tokens, against frequency-matched tokens.
- **Q3, abrupt and ordered over training.** For each atom group of the largest model (top-coefficient tokens, at least 200 each), compute CE on those tokens over about 30 checkpoints (log-spaced to step 512, then linear). No transcoders are needed at checkpoints.
  - Fit (a) a sigmoid in log-step and (b) a smooth power law / exponential; compare by cross-validated likelihood over tokens.
  - Sharpness = maximum slope / total drop.
  - Acquisition time = sigmoid midpoint.
  - **Nulls:** frequency-matched random token groups of equal size; token-identity groups; QDG cluster groups; node-dictionary groups.
  - **Quanta prediction:** atom groups are sharper than matched random groups (paired test), and acquisition time correlates with atom frequency (Spearman ≤ −0.3) after partialling out token unigram frequency.
  - **Smooth/superposition prediction:** no excess sharpness, and acquisition explained by token frequency alone.
  - Chang et al. (https://arxiv.org/abs/2308.15419) already report sudden per-token changes and frequency ordering at the token level. Our contribution is mechanism-defined grouping against matched nulls. Cite this as context.
- **Frequency-law readout (X2, not used as falsification).** Fit atom frequency distributions with Clauset–Shalizi–Newman (https://arxiv.org/abs/0706.1062), comparing power law, lognormal and truncated power law by likelihood ratio, across K, k, L0 and width. Compare against nulls: token unigram, transcoder-feature frequency, node-atom frequency and shuffled-wiring atoms. Never claim that heavy tails confirm or refute the quantization model.
- **Validity condition.** The same code must report "quantized" on PlantedLM and "not quantized" on PlantedLM-Shared (Tier 1). Otherwise Q1–Q3 are reported as uninterpretable.

---

## 7. Pre-registration (freeze before Tier 2)

### 7.1 Primary endpoints

Holm-corrected within tier. 95% CIs by bootstrap over documents (1,000 resamples). Three dictionary seeds.

| ID | Endpoint | PASS criterion |
|---|---|---|
| **P1** | Ground-truth recovery (Tier 1) | ≥ 80% of learned mechanisms have best-atom (or ≤ 3-atom OR) AUROC ≥ 0.95; polygenic recall ≥ 0.7 on C1/C2; beats each of QDG, NPEFF and node dictionary at equal K by ≥ 0.05 mean AUROC **or** ≥ 0.2 polygenic recall; ≥ 70% of matched atoms diagonally dominant in the knockout matrix |
| **P2** | Edges add information beyond nodes (Tier 2, then 3) | Report R²_{node→edge}. Edge codes reduce held-out unexplained edge mass by ≥ 10% relative to node codes with an optimized edge decoder (equal K, k); **and** edge-atom damage concentration exceeds node atoms, both (α) and (β), at matched mass (paired Wilcoxon p < 0.01, median ratio ≥ 1.25) |
| **P3** | Causal selectivity vs. strong nulls | Median matched-effect selectivity of edge atoms ≥ NPEFF's and ≥ 2× that of equal-mass random key sets; CPS median ≥ 1.5 (otherwise report "additive accounting, no carving") |
| **P4** | Sufficiency | KEEP with ≤ 10 effective active atoms retains ≥ 60% of the bigram→model CE gap (REAL-DELTA), and ≥ 70% of what oracle top-M pruning retains at matched edge count |
| **P5** | Coverage | Some K ≤ 2048 with average ≤ 10 active atoms reconstructs ≥ 70% of held-out in-vocabulary edge-flow mass (mass overlap Σ min(x, x̂)/Σ x), with the full dark-matter waterfall reported |
| **P6** | Known mechanisms (Tier 2/3) | ≥ 3 of 5 pre-registered mechanisms are each captured by ≤ 3 atoms with AUROC ≥ 0.9, NMI(atom activation, target-token identity) ≤ 0.2, and G-knockout dropping the task metric by ≥ 50% of the gap to mean ablation with ≤ 10% of that degradation on control text; also beats equal-mass random key sets |

### 7.2 Secondary endpoints

| ID | Endpoint | Criterion |
|---|---|---|
| S1 | Linearization faithfulness | Spearman(FROZEN-LIN, REAL-DELTA) ≥ 0.6; frozen-vs-live gap reported |
| S2 | Content vs. algorithm | V_role reaches P5 coverage with fewer atoms and lower token-identity NMI than V_feat; curve over family granularity |
| S3 | Interpretability | LLM judge prefers edge-atom cards over QDG cards ≥ 60% (binomial CI excludes 50%); simulation balanced accuracy is higher |
| S4 | Stability | Across transcoder L0/width, matched-atom fraction ≥ 50% of the seed-floor match rate; coverage curves within CI |
| S5 | NPEFF agreement | Distribution of best token-profile correlation between edge atoms and NPEFF components, against shuffled-profile null |
| S6 | Wiring matters | Real-wiring atoms have higher path coherence and lower conservation residual than shuffled-wiring atoms (p < 0.01) |

### 7.3 Exploratory readouts

X1 quanta tests Q1–Q3 · X2 frequency-law fits with nulls · X3 270M↔1B matching vs. floors · X4 PT→IT gained/lost atoms (fixed-coordinate control) · X5 monogenic/polygenic statistics · X6 merged-mechanism counts over the (τ_r, τ_f) grid.

### 7.4 What each outcome means (write these up whatever the result)

| Outcome | Interpretation to report |
|---|---|
| P1 fails | Edge-flow dictionaries do not recover planted mechanisms even with ground truth. Publish as a negative result with diagnostics; do not make Tier 2/3 mechanism claims. The instrument (extraction plus accounting) is still released |
| P1 passes, P2 fails with R² ≥ 0.9 | "At transcoder-feature granularity, per-token computation is node co-activation composed with static wiring." Edge atoms reduce to node atoms; recommend node-level census plus weights. Clean, publishable finding |
| P3: CPS ≈ 1 but selectivity passes | Atoms are a valid additive accounting of computation that predicts where damage lands, but do not carve non-additive mechanism boundaries |
| P4/P5 fail | Computation is not low-dimensional in this basis, or dark matter dominates. Lead with the waterfall: which share is error nodes vs. QK vs. the long tail |
| P6 fails, P1 passes | Natural-text mechanisms are content-fragmented (check V_role) or live in QK (check the QK share) |
| X1 positive control fails | The quanta test is uninterpretable at this scale; report so |
| X1: atoms not sharper than matched nulls (controls pass) | Evidence against mechanism-level quantization in Pythia, consistent with the smooth/superposition account |

---

## 8. Repository layout and interfaces

### 8.1 Layout

```
interp-big/
  pyproject.toml            # name=circuitatlas; extras: [dev], [gpu], [tl], [nnsight], [npeff], [judge], [interplab]
  README.md                 # results table generated from results/summary.json
  LICENSE  CITATION.cff
  configs/                  # toy.yaml, toy_shared.yaml, tinystories33m.yaml, gpt2.yaml,
                            # gemma3_270m_pt.yaml, gemma3_1b_pt.yaml, gemma3_270m_it.yaml, pythia_{70m,160m,410m}.yaml
  docs/  METHOD.md  PREREGISTRATION.md  DEVIATIONS.md  COMPUTE_LOG.md  DATA_FORMATS.md  RESULTS.md
  src/circuitatlas/
    config.py               # pydantic configs, config hashing
    models/                 # base.py (ModelAdapter protocol), tl_adapter.py, hf_gemma3.py, toy.py
    transcoders/            # base.py (Transcoder protocol: encode/pre_acts/decode/skip),
                            # load_gemmascope2.py, load_sparsify.py, load_dunefsky.py, load_local.py, train.py
    linearize/              # replacement.py (frozen linear system), vjp.py (batched VJP utilities),
                            # circuit_tracer_bridge.py
    graphs/                 # extract.py (two-stage), keys.py (EdgeKey codec), accounting.py, qk.py, store.py
    vocab/                  # build.py (min-count, OOV), roles.py (families, token classes, relational attrs)
    dictionary/             # graph_sae.py, nmf.py, lda.py, train.py, select_k.py, flow_prior.py, refine.py
    baselines/              # node_dict.py, npeff.py, qdg.py, kmeans.py, cooc_nmf.py, susceptibility.py,
                            # eap_components.py, random_keys.py, shuffled_wiring.py, l3d.py (stretch)
    interventions/          # real_delta.py, spliced.py, frozen_lin.py, knockout.py, sufficiency.py
    metrics/                # coverage.py, node_edge.py, causal.py (selectivity, DC, CPS, boundary),
                            # recovery.py, known_mech.py, matching.py, powerlaw.py, acquisition.py, stats.py
    synth/                  # planted.py (PlantedLM + Shared), probes.py, train_toy.py
    tasks/                  # induction.py, ioi.py, name_copy.py, brackets_quotes.py, successor.py, greater_than.py
    census/                 # merge.py (functional equivalence), waterfall.py, report.py
    label/                  # cards.py (HTML atom cards), judge.py (optional LLM judge), human_template.py
    export/                 # interplab.py (activation records / feature dumps), atlas_site.py
    cli.py                  # typer app: `catlas`
  tests/                    # unit + integration (CPU, < 10 min total)
  notebooks/                # 01_planted_ground_truth, 02_tinystories_census, 03_known_mechanisms,
                            # 04_gemma3_census_stability, 05_quanta_pythia (executed, outputs saved)
  scripts/                  # run_tier{0..4}.sh, cost_probe.py, make_figures.py
  results/                  # summary.json + small per-experiment JSON/CSV (committed); large artifacts gitignored
```

### 8.2 CLI (illustrative)

```
catlas synth train --config configs/toy.yaml
catlas transcoders train --config configs/toy.yaml
catlas cost-probe --config configs/gemma3_270m_pt.yaml --n-seq 10
catlas extract --config configs/tinystories33m.yaml --n-targets 20000 --out runs/ts33m/graphs
catlas vocab --run runs/ts33m --vocab feat --min-count 5
catlas fit --run runs/ts33m --method graph_sae --K 512 --k 8 --seeds 0,1,2
catlas baselines --run runs/ts33m --all
catlas validate --run runs/ts33m --dict graph_sae_K512_k8 --panel causal,sufficiency,node_edge
catlas recover --run runs/toy                 # ground-truth metrics
catlas known-mech --run runs/ts33m --tasks induction,name_copy,quotes,successor
catlas census --run runs/ts33m                # waterfall, coverage curves, merge grid
catlas match --a runs/g270m --b runs/g1b --floors runs/g270m_seed1,runs/g270m_medL0
catlas quanta --config configs/pythia_70m.yaml --checkpoints log30
catlas report --run runs/ts33m --html --update-summary
```

### 8.3 Data formats (`docs/DATA_FORMATS.md`)

- **Graph shards** (`.npz` + parquet index), one row per target:
  - doc_id, seq_id, pos, token, next_token, margin;
  - accounting scalars;
  - edge CSR (indptr, key_hi, key_lo int64, val float32);
  - node CSR (keys, n values).
  
  Shards of 2,048 targets each, plus a manifest with config hash, git SHA, model/transcoder revisions and seed.
- **Dictionaries:** safetensors (W_enc, b_enc, D) plus vocabulary JSON plus training log.
- **Interp-lab interop (optional extra):** export atoms as interp-lab activation records / feature dumps (JSONL; per-token c_tk with contexts). Read interp-lab's `docs/COMMANDS.md` at build time for the exact schema (https://github.com/asystemoffields/interp-lab). This lets interp-lab's `compare`, `validate-matches` and `intervene` (with its `random_feature` / `matched_frequency` / `placebo` controls) grade atoms as an independent check. CircuitAtlas must never depend on interp-lab at runtime.

---

## 9. Milestones with validation checks

| # | Milestone | Validation check (must pass) |
|---|---|---|
| M0 | Scaffold: pyproject, CI (CPU), configs, CLI skeleton, docs stubs | `pip install -e .[dev]`; `pytest -q` green; `catlas --help` lists all commands |
| M1 | Linear-flow core on synthetic DAGs (Tier 0) | Flows equal brute-force path sums (rel err < 1e-6); I1–I3 hold; pruning/aggregation keeps I1–I2 exact; FROZEN-LIN knockout equals explicit re-solve |
| M2 | PlantedLM + Shared generators, probes, toy training, width sweep, checkpoints | Generator labels unit-tested; toy d=256 learns M1–M7, C1 and C2 (probe acc ≥ 0.9) and the top ≥ 16 LOOKUP tasks; determinism: same seed gives identical batches |
| M3 | Transcoder protocol, loaders, toy transcoder training, spliced model | Spliced-with-errors logits equal original (max abs diff < 1e-4 fp32); toy FVU ≤ 0.1; zero-error CE cost ≤ 10% of gap; Gemma Scope 2 and sparsify loaders load one layer each (smoke test, skippable offline) |
| M4 | Linearized replacement + two-stage extraction | Linearized model reproduces clean m_t exactly at clean point; Stage-A τ equals finite differences on the linear system (rel err < 1e-3); Stage-B P rows equal JVP from source decoders; I1–I3 on real toy graphs (rel < 1e-4); **circuit-tracer cross-check** on one prompt for Gemma-3-270M (nnsight) or Qwen3-0.6B: direct-effect weights on the shared node set within 1e-2 relative (document any convention differences) |
| M5 | Corpus extraction, storage, accounting, vocabularies (feat, role), cost probe | Waterfall shares sum to 1 ± 1e-6; EdgeKey round-trip property tests; toy throughput logged; OOV share on val < 10% at chosen c_min |
| M6 | Graph SAE, NMF, K-sweep, refinement, MDL proxy | Gram-trick loss equals dense loss (rel < 1e-5); synthetic planted dictionary: ≥ 90% atoms recovered at cosine ≥ 0.95; dead atoms < 5% |
| M7 | Interventions: REAL-DELTA, SPLICED-FROZEN-ERR, FROZEN-LIN, KO, G, KEEP | Null intervention gives bit-identical outputs; tiny single-edge knockouts: Spearman(FROZEN-LIN, REAL-DELTA) ≥ 0.9 on toy; G(k) with an empty key set is a no-op; batched per-token interventions equal unbatched |
| M8 | All baselines B1–B9 | Each runs end-to-end on toy; QDG clusters recover ≥ 50% of the high-frequency LOOKUP tasks (sanity, Michaud-style); NPEFF variant documented |
| M9 | **Tier 1 complete** + quanta calibration on PlantedLM/Shared → **tag v0.1** | P1 PASS/FAIL recorded with CIs; λ_flow frozen; positive/negative control classification recorded; atom cards for all matched atoms |
| M10 | Freeze PREREGISTRATION.md; Tier 2 TinyStories-33M | Transcoder quality report; P2–P6 and S1–S3, S5, S6 recorded; notebooks 02 and 03 executed |
| M11 | GPU cost probe → Gemma 3 270M census, stability, 1B, PT/IT | Projection logged; S4 recorded; X3/X4 with floors |
| M12 | Pythia quanta test | X1/X2 recorded, with the control-validity flag |
| M13 | Writeup, static atlas site, release v1.0 | README table auto-generated; every figure reproducible via `scripts/make_figures.py`; RESULTS.md lists every PASS/FAIL/NOT-RUN |

---

## 10. Compute plan

**Default hyperparameters:**

| Param | PlantedLM | TinyStories-33M (CPU) | GPT-2 small (stretch) | Gemma 3 270M (GPU) |
|---|---|---|---|---|
| T | 64 | 64 | 64 | 64 |
| targets/seq |S| | 16 | 16 | 8 | 16 |
| N_node | 128 | 256 | 256 | 256 |
| top-E keys | 300 | 500 | 500 | 500 |
| targets (train/val/test) | 20k/2k/2k | 16k/2k/2k | 16k/2k/2k | 80k/10k/10k |
| c_min / V_max per sign | 3 / 2¹⁵ | 5 / 2¹⁶ | 5 / 2¹⁶ | 20 / 2¹⁷ |
| K sweep | 16–512 | 64–2048 | 64–2048 | 128–4096 |
| avg k | 4, 8 | 8 (4, 16) | 8 | 8 (4, 16) |

**Budgets.** These are estimates; the cost probe overrides them.
- **CPU tier** (16–32 cores, 64 GB RAM):
  - PlantedLM model and width sweep: 4–8 h;
  - toy transcoders: about 1 h;
  - toy extraction: < 30 min;
  - toy dictionaries and baselines: < 3 h;
  - TinyStories transcoders: 4–8 h;
  - TinyStories extraction (20k targets): projected 2–8 h;
  - interventions panel (about 10⁵ batched forwards): 1–3 h;
  - QDG and NPEFF: 2–5 h;
  - **total ≈ 2–3 days wall-clock.**
  
  GPT-2 small roughly triples extraction and intervention cost; run it only if the Tier-2 TinyStories run finishes with time to spare.
- **GPU tier** (1× 24–48 GB):
  - Gemma 3 270M extraction: projected 0.1–1 GPU-day per transcoder variant, because two-stage pruning and target subsampling avoid full sequence graphs (the red team's 10× warning applies to full graphs);
  - ≥ 2 variants for stability;
  - 1B at about 3× the cost;
  - IT runs: 2 more;
  - Pythia transcoders: 70m about 3 GPU-h, 160m about 8 GPU-h, 410m about 1–2 GPU-days (stretch);
  - checkpoint loss curves: < 2 GPU-h;
  - **total ≈ 4–8 GPU-days for the full Tier 3+4; minimum GPU deliverable (270M census plus one stability variant) ≈ 1–2 GPU-days.**
- **Storage:** about 10 KB per target for edges plus nodes, so about 1 GB per 100k-target run; < 50 GB overall.
- **Cost-gate reduction order** (apply until the projection fits the budget, and log it):
  1. |S| (targets per sequence)
  2. N_node
  3. T (to 32)
  4. number of transcoder variants
  5. drop the 1B model
  6. drop 410m

  Never cut the toy controls, the baselines B1/B2, or the nulls.

---

## 11. Writeup outline

**README.md**
1. One-paragraph pitch and novelty statement (Section 2.2).
2. Install; 15-minute CPU quickstart (`scripts/run_tier1.sh --tiny`).
3. Concepts: edge flows, conservation, atoms, dark matter (one diagram).
4. Results table, auto-generated, PASS/FAIL/NOT-RUN with CIs.
5. Key figures:
   - (1) PlantedLM recovery and polygenic recall vs. baselines;
   - (2) waterfall plus coverage curves;
   - (3) edge-vs-node gap;
   - (4) a known-mechanism atom (for example an induction atom subgraph);
   - (5) quanta test with controls.
6. Limitations and "what this is not".
7. Interop with interp-lab and circuit-tracer.
8. Citation and references.

**Paper (`docs/RESULTS.md` → arXiv draft)**
1. Introduction: from per-prompt graphs to a census; the open problem; contributions.
2. Related work: the Section 2.1 table.
3. Method: linearization; flows and identities; two-stage extraction; vocabularies; dictionary with flow prior; interventions; accounting.
4. Ground truth: PlantedLM and Shared.
5. Real LMs: dark-matter waterfall and coverage; edges vs. nodes; causal panel; sufficiency; known mechanisms; content vs. algorithm.
6. Census structure: frequency laws with nulls; monogenic vs. polygenic; stability; cross-scale; PT vs. IT.
7. Quanta acquisition test.
8. Limitations: frozen QK; dependence on the transcoder; phantom splitting; linearization gap; truncation.
9. Conclusion.

Appendices: the pre-registration verbatim; deviations; compute log; hyperparameters; all negative results.

---

## 12. Risks and mitigations

| Risk | Mitigation built into the spec |
|---|---|
| Atoms reduce to node co-activation plus wiring | R²_{node→edge}, the equal-budget code comparison, node-atom knockouts, shuffled wiring (P2, S6); reported as a finding if true |
| Trivially passing causal tests | Matched-effect selectivity, CPS with presence matching, boundary test, equal-mass random keys, oracle pruning; linearized agreement demoted to S1 |
| Error nodes dominate (especially with weaker transcoders) | Error share in the waterfall; transcoder quality gates; stability across L0 and width |
| QK-implemented mechanisms invisible | QK share reported as dark matter; `prev_match`/`tok_match` relational keys in V_role; QK edges (SG3) |
| Content fragmentation (one atom per name) | V_role and family-granularity sweep (S2); token-identity NMI in P6 |
| Phantom splitting makes counts ill-defined | Counts only as curves; functional-equivalence merge grid; coverage leads |
| Quanta test confounded by Zipf statistics | Acquisition/sharpness tests against frequency-matched nulls; positive and negative toy controls; exponent fits never used as falsification |
| Cross-model diffs measure transcoder disagreement | Seed and transcoder floors; fixed-coordinate PT→IT control |
| Extraction cost | Two-stage pruning, shared P rows, target subsampling, cost probe and reduction order |
| Transcoders for Pythia or TinyStories unavailable | Self-train with sparsify; quality gates |
| Scoop risk | Tag v0.1 after Tier 1; README states the novelty with its date |

---

## 13. Stretch goals (ideas grafted from runner-up proposals)

- **SG1, mechanism lens (scaling path).** Train a light encoder from transcoder activations (all layers, positions ≤ t) to atom coefficients c_t. If held-out R² ≥ 0.8, mechanism activity can be read at forward-pass cost with no graph extraction, which makes monitoring across billions of tokens feasible. Validate by re-running P3 selectivity using lens-predicted coefficients.
- **SG2, bits-priced explanations of atoms** (from the DECOMPILE / Nats-of-Understanding runners-up).
  - Price each atom's subgraph in bits (nnz × (log₂ 2V + 8)) and plot a coverage-vs-bits frontier for edge atoms, node atoms and per-token graphs.
  - Express each atom's *trigger* as a short executable program over a closed, priced DSL (token sets, regexes, previous-k predicates, bracket depth, the `prev_match` relation). No LLM calls may run inside the program; LLMs may only propose programs offline.
  - Swap programs for encoder coefficients in KEEP and measure the fraction of sufficiency retained per bit.
  - Mandatory baseline: an LLM-free MDL search (greedy decision lists) over the same DSL.
  - Context: Gao et al. §4.4 https://arxiv.org/abs/2406.04093, Paulo & Belrose https://arxiv.org/abs/2501.18838, Hayes et al. https://arxiv.org/abs/2606.19317, Gauderis et al. https://arxiv.org/abs/2605.08934.
- **SG3, QK edges.** Kamath-style bilinear QK attribution (https://transformer-circuits.pub/2025/attention-qk/index.html) as relation type `QK(h)` between query-side and key-side nodes in V_feat. Remeasure the QK share as covered vs. uncovered.
- **SG4, CLT coordinates.** Repeat the Gemma 3 270M census with Gemma Scope 2 CLTs, using CLT-Forge (https://arxiv.org/abs/2603.21014) or circuit-tracer CLT loaders, and compare coverage and atom counts against PLTs.
- **SG5, developmental atoms** (from the Chronoscope runner-up). On toy checkpoints, where transcoders are cheap to train per checkpoint, run censuses and match atoms over time by token profile. Find births, deaths and "fossil" atoms (present mid-training, gone at the end). Apply a successor-matching null (a death counts only if no newborn atom has a matching token profile and task signature) and validate fossils by knockout at the peak checkpoint. Cite Pellegrino et al. https://arxiv.org/abs/2308.11567 for the tensor-over-training lineage.
- **SG6, L3D head-to-head** on toy and TinyStories.
- **SG7, public atlas site.** A static, Neuronpedia-style browser of atom cards, with export to interp-lab feature records.

---

## 14. Definition of done

- `pip install circuitatlas` (or `pip install -e .`) works on CPU; `scripts/run_tier1.sh --tiny` runs end-to-end in ≤ 15 minutes on a laptop CPU.
- Tier 1 complete, with P1 reported and v0.1 tagged. Tier 2 complete, with P2–P6 reported. Each GPU tier is either complete or explicitly NOT-RUN, with the reason in `results/summary.json`.
- Every number in the README comes from `results/summary.json`, and every figure regenerates from committed small artifacts plus documented large-artifact paths.
- All references in README and paper are from Section 15, or have been added after verification with a working URL.

---

## 15. References (verified URLs only)

**Attribution graphs and the open problem:**
- Ameisen et al. 2025, Circuit Tracing (methods, global weights, TWERA): https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- Lindsey et al. 2025, On the Biology of a Large Language Model: https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- Kamath et al. 2025, Tracing Attention Computation Through Feature Interactions (QK attribution): https://transformer-circuits.pub/2025/attention-qk/index.html
- Circuits Updates, May 2026: https://transformer-circuits.pub/2026/may-update/index.html
- Circuits Research Landscape (open problems): https://www.neuronpedia.org/graph/info
- circuit-tracer: https://github.com/safety-research/circuit-tracer
- CLT-Forge: https://arxiv.org/abs/2603.21014
- Elhage et al. 2021, A Mathematical Framework for Transformer Circuits: https://transformer-circuits.pub/2021/framework/index.html

**Transcoders, dictionaries and tooling:**
- Gemma Scope 2 (270M PT): https://huggingface.co/google/gemma-scope-2-270m-pt
- Dunefsky et al. 2024, Transcoders: https://arxiv.org/abs/2406.11944; weights: https://huggingface.co/pchlenski/gpt2-transcoders; code: https://github.com/jacobdunefsky/transcoder_circuits
- EleutherAI GPT-2 transcoders (no model card): https://huggingface.co/EleutherAI/gpt2-mntss-transcoder-relu-sp6-skip
- sparsify: https://github.com/EleutherAI/sparsify
- Paulo, Shabalin & Belrose 2025, Transcoders Beat Sparse Autoencoders: https://arxiv.org/abs/2501.18823
- Gao et al. 2024, Scaling and Evaluating Sparse Autoencoders: https://arxiv.org/abs/2406.04093
- Bussmann, Leask & Nanda, BatchTopK SAEs: https://arxiv.org/abs/2412.06410
- Bussmann et al., Matryoshka SAEs: https://arxiv.org/abs/2503.17547
- TransformerLens: https://github.com/TransformerLensOrg/TransformerLens
- nnsight: https://github.com/ndif-team/nnsight
- interp-lab: https://github.com/asystemoffields/interp-lab

**Global and polygenic decompositions (direct competitors):**
- Michaud et al. 2023, Quantization Model: https://arxiv.org/abs/2303.13506
- Marks et al., Sparse Feature Circuits: https://arxiv.org/abs/2403.19647
- Matena & Raffel, NPEFF: https://arxiv.org/abs/2310.04649; code: https://github.com/mmatena/npeff_torch
- Chrisman, Bushnaq & Sharkey, L3D: https://arxiv.org/abs/2504.00194
- Baker et al., Structural Inference with Susceptibilities: https://arxiv.org/abs/2504.18274
- Wang et al., Embryology of a Language Model: https://arxiv.org/abs/2508.00331
- Golimblevskaia et al., Circuit Insights (CircuitLens): https://arxiv.org/abs/2510.14936
- Arora et al., ADAG: https://arxiv.org/abs/2604.07615
- Probe prompting: https://arxiv.org/abs/2511.07002
- Circuit-Diff: https://arxiv.org/abs/2609.23892
- Patel, Zhang & Hu, LLMs Can Annotate Attribution Graphs: https://arxiv.org/abs/2608.02632
- HAGD: https://arxiv.org/abs/2601.12879
- Aljaafari et al., ILP circuit signatures: https://arxiv.org/abs/2605.21303
- Palit et al., concept-targeted attribution: https://arxiv.org/abs/2608.27510
- Wang 2026, Communication Map of a Transformer: https://arxiv.org/abs/2608.22007
- circuit-motifs: https://github.com/mkenney2/circuit-motifs
- Bayat Makou et al., Many Circuits, One Mechanism: https://arxiv.org/abs/2606.06267
- SpRAy: https://arxiv.org/abs/1902.10178
- PCX: https://arxiv.org/abs/2311.16681

**Parameter-space and computation dictionaries:**
- APD: https://arxiv.org/abs/2501.14926
- SPD: https://arxiv.org/abs/2506.20790
- SPD on small transformers: https://arxiv.org/abs/2511.08854
- Targeted PD: https://arxiv.org/abs/2607.13047
- ASPD: https://arxiv.org/abs/2609.37731
- VPD (Goodfire): https://www.goodfire.com/research/interpreting-lm-parameters
- MOLT (Circuits bulk update, July 2025): https://transformer-circuits.pub/2025/bulk-update/index.html
- Jacobian SAEs: https://arxiv.org/abs/2502.18147

**Scaling theory and learning dynamics:**
- Liu, Liu & Gore, Superposition Yields Robust Neural Scaling: https://arxiv.org/abs/2505.10465
- Chang, Tu & Bergen, Characterizing Learning Curves During LM Pre-Training: https://arxiv.org/abs/2308.15419
- Clauset, Shalizi & Newman, Power-law distributions in empirical data: https://arxiv.org/abs/0706.1062
- Pythia: https://arxiv.org/abs/2304.01373

**Known mechanisms, causal methods and models:**
- Olsson et al., Induction Heads: https://arxiv.org/abs/2209.11895
- Wang et al., IOI: https://arxiv.org/abs/2211.00593
- Hanna et al., Greater-than: https://arxiv.org/abs/2305.00586
- Geiger et al., Causal Abstractions: https://arxiv.org/abs/2106.02997
- Syed, Rager & Conmy, Attribution Patching (EAP): https://arxiv.org/abs/2310.10348
- Eldan & Li, TinyStories: https://arxiv.org/abs/2305.07759; model: https://huggingface.co/roneneldan/TinyStories-33M

**Stretch-goal context:**
- Paulo & Belrose, Partially Rewriting a Transformer in Natural Language: https://arxiv.org/abs/2501.18838
- Hayes, Li & Andreas, Explaining Attention with Program Synthesis: https://arxiv.org/abs/2606.19317
- Gauderis et al., From Mechanistic to Compositional Interpretability: https://arxiv.org/abs/2605.08934
- Pellegrino et al., Low Tensor Rank Learning of Neural Dynamics: https://arxiv.org/abs/2308.11567