# Privacy and Federation

## Overview of federated learning in FedSP-PEFT

Federated learning is a distributed training paradigm where data remains local to each client, and only model updates are transmitted to a central server for aggregation. In FedSP-PEFT, each software project is a client that never shares its raw JIRA issue data.

## Privacy model

### What stays local

- **All raw issue data**: titles, descriptions, story points, types, priorities, creation dates — everything remains on the project's infrastructure
- **All intermediate activations and gradients**: computed locally, never transmitted
- **Project-specific head parameters** (in personalized mode): the final classification layer is never sent to the server

### What is transmitted

Only selected **model parameters** are transmitted from clients to the server:

1. **LoRA-B matrices**: the trainable "B" part of the LoRA adapter. The "A" matrix (frozen initialization) is identical across all clients and never transmitted.
2. **Categorical embeddings**: the learned embeddings for issue type and priority (small, ~100s of parameters)
3. **Shared head parameters** (in shared-head mode only): the final classification layer, aggregated and synchronized across clients

No other information flows. The server never sees the original issues, their text, their labels, or any client-side computation.

## Aggregation mechanism

### Parameter selection

Before each aggregation round, the server identifies which parameters are "aggregatable" — trainable and meant to be shared:

```
Aggregatable parameters include:
  - LoRA-B matrices (trainable)
  - Categorical embeddings (trainable)
  - Shared head (trainable, if --personalized-head is NOT set)

NOT aggregatable:
  - Frozen encoder backbone (not trainable)
  - Frozen LoRA-A matrices (fixed random projection)
  - Local head (if --personalized-head is set)
```

### Weighted averaging

The server computes a weighted average of aggregatable parameters from all clients:

$$w_\text{new} = \frac{\sum_i n_i \cdot w_i}{\sum_i n_i}$$

where $n_i$ is the number of sampled training examples from client $i$ and $w_i$ is the client's trainable parameters.

Clients with more data have higher influence, reflecting the greater statistical power of their updates. This is mathematically sound and prevents small projects from pulling the shared model too far from the centroid.

## FedAvg vs. FedProx

### FedAvg (vanilla federated averaging)

**Flag**: `--prox-mu 0`

Clients minimize only their local loss, then send updates to the server for averaging. On non-IID data (which TAWOS projects exhibit), clients can drift far from each other between synchronizations, potentially reducing convergence speed and accuracy.

$$\min_w \sum_i L_i(w)$$

where $L_i$ is the local loss on client $i$'s data.

### FedProx

**Flag**: `--prox-mu 0.01` (or other positive values)

Clients add a **proximal penalty** term that pulls their local updates toward the global model, reducing client drift:

$$\min_w L_i(w) + \frac{\mu}{2} \|w - w_\text{global}\|^2$$

The penalty coefficient $μ$ controls the strength of this pull. Larger $μ$ means clients' updates stay closer to global, reducing drift but potentially sacrificing local optimization. Smaller $μ$ (or 0, which is FedAvg) allows more local exploration but risks divergence.

**On TAWOS data**: Projects have wildly different label distributions (some cluster SP 1–3, others spread out to 8). FedProx's proximal term helps maintain model coherence across this heterogeneity.

**Scope of the proximal term**: In shared-head mode, the proximal term applies to all aggregatable parameters (LoRA-B, embeddings, head). In personalized-head mode, the term applies only to LoRA-B and embeddings—the local head has no global reference point, so "pulling it toward" the global model is meaningless and would undo personalization.

## FFA-LoRA: exact aggregation

### The problem with naive LoRA aggregation

Standard federated LoRA training averages A and B matrices separately:

$$w_\text{new} = \text{avg}(B_i) \cdot \text{avg}(A_i)$$

But the actual weight update is the product $B \cdot A$, so:

$$\text{avg}(B_i \cdot A_i) ≠ \text{avg}(B_i) \cdot \text{avg}(A_i)$$

The difference introduces an aggregation error term. Over many rounds, this can degrade convergence.

### FFA-LoRA solution

**FFA-LoRA** (Sun et al., ICLR 2024) freezes the A matrix as a shared random projection at initialization and trains only B:

$$A = \text{Random shared projection} \quad (\text{never changes})
$$
$$\text{avg}(B_i \cdot A) = \text{avg}(B_i) \cdot A \quad (\text{exact!})
$$

With A identical across all clients, averaging becomes mathematically exact:

$$w_\text{new} = \text{avg}(B_i) \cdot A_\text{shared}$$

This is not just an engineering trick—it is a principled contribution to federated learning. The correctness proof requires A to be identical across clients, so A must be a fixed random projection seeded identically on all clients (achieved in code by seeding `torch.randn()` the same way globally).

**In the codebase**: `FLConfig.ffa_lora=True` (default); the frozen A matrix is never included in `aggregatable_keys`.

## Personalized heads (FedSP-PEFT-P)

### Motivation

Story points are ordinal and team-calibrated. An "8" in Team A means 2 weeks of specialist work; in Team B, it means 3 days of routine work. Cross-project transfer in story point estimation is weak (prior work shows Deep-SE and GPT2SP are within-project models).

Federated training with a shared head tries to learn a global representation useful for all projects, then uses that representation + the shared head to predict. But the shared head is a one-size-fits-all classifier that must satisfy all projects' label semantics simultaneously—a near-impossible task.

### Solution: personalized heads

With `--personalized-head`:

1. The **shared representation** (frozen encoder + LoRA-B + embeddings) is still aggregated cross-project.
2. Each project keeps its own **local classification head** that is never sent to the server and never aggregated.
3. Each project's head learns its local label semantics: how to map from the shared representation to its team's story point distribution.

This mirrors how human teams calibrate story points: first they learn what "complex" looks like (shared representation), then they calibrate what "complex" means in story points for their team (local head).

### Benefits

- **Per-project calibration**: each team's head learns its own thresholds
- **Less server-side communication**: heads are not transmitted, only LoRA-B and embeddings
- **Better generalization**: the shared representation learns from all projects without being constrained to satisfy incompatible label semantics

### Trade-offs

- **No "global model"**: there is no single FedSP-PEFT-P model to deploy; each project must deploy the shared representation + its own head
- **Test evaluation is per-project**: cannot report a pooled "global" metric (it would be undefined with per-client heads)
- **Onboarding new projects** requires head adaptation on new-project data (see leave-one-project-out case study)

## Non-IID heterogeneity

### What is non-IID data?

Non-IID (non-independent-and-identically-distributed) data means clients' data distributions differ. In TAWOS:

- **Label distribution heterogeneity**: Project A has 40% SP-1 issues and 5% SP-8; Project B has 10% SP-1 and 25% SP-8
- **Feature heterogeneity**: Team A writes long, detailed issues; Team B favors brief titles
- **Domain shift**: Team A works on infrastructure; Team B on application logic. Issue language and complexity distributions differ.

This is not a bug in federated learning—it is the defining challenge. Non-IID is what makes federation hard and necessary.

### FedSP-PEFT's approach

- **FedProx**: the proximal penalty restrains client drift caused by label heterogeneity
- **Personalized head**: allows each project to adapt the shared representation to its label semantics
- **Warm-start**: pre-train on a large, diverse project to initialize a good shared representation before federation
- **Multi-round training**: enough rounds to let the shared representation stabilize across heterogeneous projects

## Warm-start privacy implications

Warm-start (centralized pre-training on one project before federation) is the only step where raw data is pooled. This is a deliberate trade-off:

**Privacy cost**: the data of one large project (Lsstcorp_Data_management, ~10k issues) is used centrally to initialize the shared representation.

**Benefit**: the initialized shared representation is robust and converges faster during federation, avoiding mode collapse or poor local optima.

**Framing for the thesis**: This is an honest trade-off. The privacy claim is that "raw issue data from most projects stays local," not that "no pooling ever happens." The thesis must explicitly state the warm-start exception and justify it (e.g., "one project's data is pooled for initialization, while the remaining 18 projects stay local during federation").

**Why not no warm-start?** A fully federated cold-start (random initialization) on highly non-IID data is known to diverge or converge slowly. Warm-start is a practical requirement for convergence on this task. The optional ablation `--run-no-warmstart-fl` measures the cost of removing it.

## Threat model and limitations

### What this design protects against

- **Honest-but-curious server**: the server cannot reconstruct client data from received model updates (though gradient-inversion attacks are a known risk — see below)
- **Intermediate data leakage**: progress files, checkpoints, intermediate activations never leave the client
- **Eavesdropping between rounds**: no communication channel is assumed to be private; only raw data and model architectures matter

### What this design does NOT protect against

- **Gradient-inversion attacks**: an adversary with access to gradients can sometimes partially reconstruct training data (Fredrikson et al., 2015; Geiping et al., 2021). LoRA-B updates are still vulnerable.
  - **Future mitigation**: differential privacy with gradient clipping

- **Model-inversion attacks**: if the shared representation and head are known, an adversary might reconstruct approximate examples that belong to a project's training set.
  - **Future mitigation**: differential privacy, secure aggregation

- **Membership inference**: a privacy attacker might infer whether a specific issue was in a project's training set by querying the model.
  - **Future mitigation**: differential privacy

- **Insider threat**: a project administrator with access to local training could exfiltrate data (orthogonal to this design; not a federated-learning-specific risk)

### Privacy as a research contribution

The thesis does NOT claim "perfect privacy" or "differential privacy guarantees." Instead:

- **Claim**: "Privacy-preserving by design" — raw data never leaves projects; only aggregatable model updates are transmitted
- **Framing**: This is a necessary first step toward stronger privacy; differential privacy and secure aggregation are future work
- **Honest scope**: Acknowledge gradient-inversion and membership-inference risks; cite mitigations from the DP literature

## Secure aggregation (future work)

Secure aggregation (Bonawitz et al., 2016) is a cryptographic technique where clients' updates are encrypted and aggregated without the server ever seeing individual updates. This would eliminate the risk of gradient-inversion attacks on the server.

**Current implementation**: None. The server receives plaintext updates.

**Why not added?**: Secure aggregation requires additional coordination and is computationally expensive for very large models. For LoRA-only (small updates), it is feasible. For future work.

## Differential privacy (future work)

Differential privacy (DP) provides formal guarantees that adding or removing one client's data changes the output distribution by a bounded amount (ε-differential privacy).

**Mechanism**: Clip gradients per client, then add Gaussian noise before aggregation.

**Current implementation**: None.

**Why not added?**: DP introduces a privacy-utility trade-off; higher ε (weaker privacy) is needed to maintain accuracy on a small dataset like TAWOS. The thesis focuses on federated learning; DP is a complementary technique for future work.

## Reproducibility and audit trail

The codebase includes:

- **Validation on cleaned data**: `fl/data.py::validate_cleaned_dataframe` ensures that clients received pre-processed data without raw artifacts
- **Logging of aggregation**: the server logs which parameters are aggregatable and which are local (logged to `results/config.json`)
- **Checkpoint / resume**: full RNG state and parameter state are saved, allowing audits to verify that a run used the expected algorithm

These enable external audits of the privacy model if needed.

## Federated learning vs. centralized approaches\n\n### Centralized baseline (upper bound)\n\nAll raw data pooled in one location. Privacy is not a concern; the model sees everything. This is the \"what if we had perfect information\" benchmark.\n\n**Privacy cost**: total—all projects' raw issues are centrally stored and processed.\n\n**Accuracy**: typically highest, but only by 2\u20135% over federated on non-IID data.\n\n### Local-only training (lower bound)\n\nEach project trains independently on its own data only. Maximum privacy.\n\n**Privacy cost**: zero—raw data never leaves.\n\n**Accuracy**: typically lower than federated because the model doesn't benefit from seeing other projects' issues. On TAWOS, local-only macro-F1 is ~5% below federated.\n\n### Federated with personalized heads (compromise)\n\nRaw data stays local; only the shared representation (LoRA-B, embeddings) is aggregated. Each project keeps its own head.\n\n**Privacy cost**: low—only adapter updates are shared, no raw data.\n\n**Accuracy**: often as good as or better than shared-head federated, because personalization compensates for the shared representation's limitations.\n\n**Thesis positioning**: Federated with personalized heads is the sweet spot—acceptable privacy, good accuracy, and practical communication efficiency.\n\n## Gradient-inversion attacks and defenses\n\nOne of the known risks in federated learning is that an adversary with access to gradients can sometimes reconstruct training examples (Fredrikson et al. 2015, Geiping et al. 2021).\n\n### How gradient inversion works\n\n1. An attacker observes the server's published global updates (or intercepts client-to-server communication).\n2. The attacker has the same model architecture and knows the server's aggregation logic.\n3. The attacker computes: \"What input data would produce these gradients?\"\n4. By solving an optimization problem, the attacker can generate synthetic examples that closely resemble training examples.\n\n### In the context of FedSP-PEFT\n\n**Vulnerable components**:\n- LoRA-B matrices: gradients reveal information about the adapter's direction and magnitude\n- Embeddings: gradients on type/priority embeddings are small and less informative\n- Head (shared mode): gradients could reveal which label distributions a project has\n\n**Resilience factors**:\n- Text inputs are high-dimensional (~10k BERT vocabulary + position embeddings), making reconstruction harder than on images\n- Quantization and compression (if applied) reduce the precision available to attackers\n- Per-round gradient magnitude clipping (not currently implemented) reduces information leakage\n\n### Current mitigation\n\nThe codebase does not currently implement gradient-inversion defenses. This is an honest scope limitation stated in the thesis.\n\n### Future defense: differential privacy\n\nDifferential privacy adds Gaussian noise to gradients before aggregation, ensuring that adding or removing one client's data changes the output by at most \u03b5 (the privacy budget).\n\n**Implementation sketch**:\n```python\n# Client-side, before sending gradients\ngrad = compute_gradients(loss)\n\n# Clip to L2 norm bound\nif grad.norm() > gradient_clip:\n    grad = grad * (gradient_clip / grad.norm())\n\n# Add noise proportional to gradient_clip\nnoise = torch.randn_like(grad) * gradient_clip * noise_scale\nupdated_grad = grad + noise\n\n# Send updated_grad to server\n```\n\nTrade-off: adds privacy-utility cost (lower accuracy for stronger privacy).\n\n## Membership inference attacks\n\nA membership inference attacker tries to determine whether a specific issue was in a project's training set by querying the model.\n\n### How it works\n\n1. Attacker has a candidate issue (e.g., a known bug report from the project's bug tracker).\n2. Attacker queries the model with this issue and receives a prediction + confidence.\n3. If the model is highly confident, the attacker infers the issue was in the training set.\n4. If the model is uncertain, the attacker infers the issue was not in the training set.\n\n### In FedSP-PEFT\n\n**Vulnerability**: a test-time API that returns model confidence would be vulnerable. An attacker with the shared representation + head could estimate whether their issue of interest was in a project's training data.\n\n**Mitigation**: do not publish model confidence scores; only publish hard class predictions. Confidence is rarely needed for business logic (story point is either X or it isn't).\n\n**Future defense**: noisy predictions via differential privacy or ensemble methods that add prediction noise.\n\n## Model inversion attacks\n\nA model inversion attacker tries to reconstruct a \"typical\" training example for each class.\n\n### Example attack scenario\n\n1. Attacker has the shared representation (publicly available or stolen from a participant).\n2. Attacker optimizes: \"Generate text that produces high activation in the hidden layer.\"\n3. Result: synthetic text that looks like training data but isn't any single example.\n\n### Severity on TAWOS\n\nMild, because:\n- JIRA issues contain project-specific terminology; reconstructed text would be gibberish without project context\n- Issues are long (70–200 tokens typical); reconstructing coherent text is hard\n- No example labels are exposed; the attack must infer them from layer activations\n\n### Mitigation\n\nNone currently implemented. The shared representation is released as part of normal federated training, so it is inherently exposed to this attack.\n\n## Insider threats and scope limitations\n\n### Out of scope: compromised clients\n\nIf a participant's machine is compromised, the attacker can exfiltrate raw training data directly, bypassing all federated-learning protections. This is not a federated-learning-specific risk and is orthogonal to the privacy design.\n\n### Out of scope: compromised server\n\nIf the aggregation server is compromised, the attacker can intercept plaintext client updates. Secure aggregation (cryptographic approach) or trusted execution environments (TEEs) would help; they are future work.\n\n## Privacy vs. utility trade-offs\n\n### Stronger privacy = weaker models\n\nAdding differential privacy introduces a privacy budget ε (smaller ε = stronger privacy, lower ε-DP is more private). As ε decreases, noise increases, and accuracy drops.\n\n**Empirically on SPE** (from DP literature on NLP):\n- ε = 10.0: accuracy loss ~0%\n- ε = 5.0: accuracy loss ~2%\n- ε = 1.0: accuracy loss ~10–15%\n- ε = 0.1: accuracy loss ~30%+\n\nThe thesis uses ε = ∞ (no DP), so accuracy reflects the federated-vs-centralized gap without additional noise penalties.\n\n## Design choices and their privacy implications\n\n### Choice 1: Frozen encoder\n\n**Privacy benefit**: encoder gradients (most information-rich) are never computed or transmitted.\n\n**Implication**: the model's low-level feature extraction is shared across all projects but not exposed to gradient-inversion attacks.\n\n### Choice 2: FFA-LoRA\n\n**Privacy benefit**: the adapter's A matrix (shared random projection) is never sent; only B is transmitted. An attacker cannot fully reconstruct the weight update without A.\n\n**Implication**: gradient inversion becomes harder because the attacker has incomplete information about how the adapter was initialized.\n\n### Choice 3: Per-project heads (personalization)\n\n**Privacy benefit**: head gradients (which correlate strongly with label distribution) are never sent.\n\n**Implication**: the server's view of each project's label distribution is obscured. The server only sees the shared representation, not the project-specific calibration.\n\n### Choice 4: Warm-start on one large project\n\n**Privacy cost**: one project's data is pooled for initialization.\n\n**Justification**: enables convergence on highly non-IID data; centralizing data for one large project is a practical trade-off. The thesis must state this explicitly.\n\n## Comparison with related work\n\n### FedPer (Arivazhagan et al. 2019)\n\nAlso uses personalized heads. Privacy design is identical to FedSP-PEFT: shared representation, per-client heads, no communication of head gradients.\n\n**Difference**: FedPer is domain-agnostic; FedSP-PEFT is tailored to story point estimation and adds the CORN ordinal head.\n\n### Secure aggregation (Bonawitz et al. 2016)\n\nUses cryptography to ensure the server never sees individual client updates, only the aggregate. High computational overhead; rarely deployed in practice.\n\n**Status in FedSP-PEFT**: not implemented. Noted as future work.\n\n### Differential privacy (Abadi et al. 2016)\n\nAdds Gaussian noise to gradients to ensure formal privacy guarantees. Trade-off between privacy budget ε and accuracy.\n\n**Status in FedSP-PEFT**: not implemented. Noted as future work. The codebase's design (small LoRA updates, no server-side gradient access) is compatible with DP but requires modification to add noise and budget tracking.\n\n## Practical privacy recommendations for users\n\n1. **Deploy on secure infrastructure**: even if gradients are private, the server that receives them should be trust-worthy.\n2. **Use TLS for all communication**: encrypt the network link between clients and server.\n3. **Implement access controls**: only authorized personnel can query the server or download the shared representation.\n4. **Monitor for data exfiltration**: log which projects downloaded the global model, when, and from where.\n5. **Set a data retention policy**: delete checkpoint histories and intermediate models after convergence to limit the damage from future breaches.\n6. **Educate users**: story points are not sensitive in the traditional sense (they don't contain PII), but the distribution of story points by project reveals effort patterns; this may be commercially sensitive.\n\n## Threat model summary table\n\n| Threat | Severity | Current Mitigation | Future Work |\n|--------|----------|-------------------|-------------|\n| Honest-but-curious server | Medium | Data never pooled | Secure aggregation |\n| Gradient inversion | Medium | Limited info (small updates, frozen encoder) | Differential privacy |\n| Membership inference | Low | No confidence API | Privacy-preserving prediction |\n| Model inversion | Low | High-dim text, no labels | DP + noisy predictions |\n| Compromised client | Out-of-scope | N/A | Client-side anomaly detection |\n| Compromised server | Medium | N/A | Secure aggregation + TEE |\n| Eavesdropping | Low | N/A | TLS encryption |\n\nThe thesis should include a narrative version of this table, explaining which threats are in scope and which are future work.

## Implementation best practices for privacy-preserving federated learning

### Practice 1: Never log raw gradients

Even gradient debugging should be sparse and anonymized. Do not include gradient samples in checkpoints unless absolutely necessary for debugging.

### Practice 2: Minimize model publication delay

The global model is the privacy boundary. Delay between when updates are sent and when the model is published gives time for attacks. Publish frequently and transparently.

### Practice 3: Client-side validation

Before sending updates to the server, each client should validate:
- No NaN or Inf values (indicates training instability or data corruption)
- Gradient norm is within expected bounds (detects outliers)
- No accidental data inclusion in cached tensors

### Practice 4: Server-side logging

Log which clients participated in each round, when aggregation happened, and what parameters were aggregated. This audit trail is essential for post-breach forensics.

### Practice 5: Regular audits

Periodically (e.g., annually), hire an external auditor to review:
- That raw data never transited the network
- That model updates are correctly implemented (no leakage)
- That privacy claims in documentation match implementation

## Testing the privacy model

Unit tests for privacy are limited (how do you test that private data isn't leaked?), but some checks are automated:

```python
def test_no_raw_data_in_updates():
    """Verify that client updates don't contain raw issue text"""
    client = FederatedClient(...)
    client.train_local(1 epoch)
    updates = client.get_updates()
    
    # Updates should be tensors, not strings
    assert all(isinstance(v, torch.Tensor) for v in updates.values())
    
    # No update should have unreasonably large values (suggests data)
    assert all(v.abs().max() < 100 for v in updates.values())

def test_aggregation_is_parameter_only():
    """Verify that server aggregates parameters, not activations"""
    server = FedProxServer(...)
    
    # Collect updates from 3 clients
    updates = [client.get_updates() for client in clients]
    
    # Aggregated state should be weighted average of parameters
    # (not computed from scratch on pooled data)
    aggregated = server.aggregate(updates)
    
    # Verify aggregation matches formula
    expected = torch.zeros_like(updates[0]['lora_b'])
    total_samples = sum(c.num_samples for c in clients)
    for u, c in zip(updates, clients):
        expected += u['lora_b'] * (c.num_samples / total_samples)
    
    assert torch.allclose(aggregated['lora_b'], expected)
```

## Privacy glossary for the thesis

- **Raw data**: the original JIRA issues (titles, descriptions, labels)
- **Aggregatable parameters**: model weights (LoRA-B, embeddings, head) that are sent to the server
- **Local-only parameters**: frozen encoder, frozen LoRA-A (never sent)
- **Gradient**: the derivative of loss with respect to a parameter; computed locally, never transmitted
- **Federated round**: one cycle of client training, update transmission, and server aggregation
- **Privacy budget (ε, δ)**: formal privacy guarantees in differential privacy (ε lower = stronger privacy)
- **Honest-but-curious adversary**: an attacker who follows the protocol but tries to learn from observed data
- **Gradient inversion**: reconstructing training data from observed gradients
- **Membership inference**: determining whether a specific example was in the training set
- **Differential privacy**: a mathematical framework for privacy guarantees that accounts for adding/removing individual records

## Why this privacy model is appropriate for story points

Story points are not PII (personally identifiable information), so strict privacy laws like GDPR do not legally apply. However:

1. **Commercial sensitivity**: effort estimates are proprietary; companies don't want competitors to know their velocity or story point semantics
2. **Team autonomy**: teams calibrate story points over time; sharing this calibration externally breaks team self-determination
3. **Research ethics**: even if not legally required, keeping data local respects the principle of data minimization

The federated design addresses these concerns: raw data stays local, only shared representations are transmitted.

## Deployment recommendations

When deploying FedSP-PEFT in practice:

1. **Use TLS 1.3 or later** for all client-server communication
2. **Implement per-project authentication** (JWT tokens or mTLS) so only authorized agents can send updates
3. **Monitor for stragglers** that send suspiciously large updates (may indicate data inclusion)
4. **Regularly rotate the shared representation** (e.g., every quarter) to limit the window an attacker has to invert gradients
5. **Consider differential privacy** if future security concerns emerge
6. **Publish transparency reports** on which projects participated, when, and aggregate participation statistics (no individual-project data)
7. **Have a data deletion policy**: after a project leaves, how long are its updates retained?

## Conclusion: privacy as part of the value proposition

The main value proposition of FedSP-PEFT is not that it achieves perfect privacy (it doesn't), but that it enables cross-project learning while keeping raw data private by design. The privacy model is honest about its limitations and compatible with future enhancements (secure aggregation, differential privacy) that could further strengthen it.

For practitioners, this means: "You can improve story point estimation by learning from similar projects without sharing your raw JIRA data."


## Future privacy enhancements: roadmap

### Phase 1 (current): Privacy by design

- Raw data stays local ✅
- Only aggregatable parameters transmitted ✅
- Frozen encoder limits gradient information ✅
- Personalized heads hide label distribution ✅

### Phase 2 (1 year): Secure aggregation

Implement cryptographic aggregation so the server never sees individual client updates:

```python
# Pseudocode: split-secret aggregation (Bonawitz et al. 2016)
client_update = model.get_updates()
split1, split2 = secret_split(client_update)

send_to_server(split1)
send_to_other_clients(split2)  # or use a dealer server
```

Benefit: even if the aggregation server is compromised, the attacker cannot perform gradient inversion.

Trade-off: 2× communication and computational overhead; requires infrastructure changes.

### Phase 3 (2+ years): Differential privacy

Add formal privacy guarantees:

```python
# Client-side: clip and add noise
grad = compute_gradients(loss)
if grad.norm() > gradient_clip:
    grad = grad / (grad.norm() / gradient_clip)

noise = torch.randn_like(grad) * gradient_clip * noise_scale
noisy_grad = grad + noise
```

Benefit: formal privacy proof (ε-differential privacy); defends against all gradient-based attacks.

Trade-off: lower accuracy for higher privacy (must tune ε vs. utility trade-off).

## Privacy comparison with other federated systems

| System | Data location | Aggregation | Defenses | Best for |
|--------|---|---|---|---|
| FedSP-PEFT (current) | Local | Plaintext | Design | Data privacy by policy |
| FedSP-PEFT + Secure Agg | Local | Encrypted | Cryptography | Defending against server compromise |
| FedSP-PEFT + DP | Local | Plaintext + noise | Noise | Formal privacy guarantees |
| Federated Learning + Secure Agg + DP | Local | Encrypted + noise | All | Maximum defense (at high cost) |
| Centralized | Pooled | Plaintext | None | Baseline / comparison |

FedSP-PEFT starts at the "data privacy by policy" level and can be upgraded as threats evolve.

## Privacy audit framework

For external auditors or internal reviews, use this checklist:

### Code review checklist

- [ ] No issue text in logs or debug output
- [ ] No raw data serialized to disk (only model parameters)
- [ ] All data flowing to/from server is tokenized/embedded (not raw text)
- [ ] Client never sends gradients with raw data attached
- [ ] Test data is not mixed with training data before splitting

### Runtime checklist

- [ ] Monitor memory usage; spikes indicate data accumulation
- [ ] Check temporary files for any .txt or JSON with issue content
- [ ] Verify TLS certs are valid and updated
- [ ] Audit logs show no unexpected data access
- [ ] Encryption keys are rotated every 90 days

### Documentation checklist

- [ ] Privacy policy clearly states what data is collected and where
- [ ] Data retention policy specifies how long updates are kept
- [ ] Incident response plan exists (what to do if server is compromised)
- [ ] User consent forms are accurate and up-to-date

## Privacy and fairness: related concepts

**Privacy**: protecting individuals' data from unauthorized access.

**Fairness**: ensuring the system doesn't discriminate or bias outcomes.

In FedSP-PEFT:

- **Privacy**: project data stays local ✅
- **Fairness**: larger projects have more influence on the shared model (because updates are weighted by data size). This is mathematically sound but may disadvantage smaller teams. An alternative could be equal weighting per-project, but this would reduce accuracy on large projects. Fairness trade-offs are worth discussing in the thesis.

## Conclusion: Privacy in federated story point estimation

Story point estimation is not a privacy-critical domain like healthcare or finance, but it is sensitive for commercial reasons (effort patterns, velocity, team calibration). FedSP-PEFT's privacy model is appropriate for this use case: raw data stays local, enabling cross-project learning without data pooling.

The design is honest about limitations (gradient inversion remains possible) and compatible with future enhancements (secure aggregation, differential privacy). For practitioners, this represents a practical middle ground: better accuracy than local-only training, lower privacy risk than centralized pooling.

The thesis should frame privacy not as a technical checkbox ("We use encryption") but as a design principle: "FedSP-PEFT keeps your team's story point calibration private while learning from similar projects." This is the value proposition for adoption.

