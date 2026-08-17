# Research Questions

> **Premise (applies to all three RQs):** each project is a client that never shares its
> raw issue data, and story-point labels are project-specific, so the clients' data is
> non-IID.

1. **RQ1** — How does FedSP-PEFT compare to centralized pooling and to per-project local
   training for story point estimation?
2. **RQ2** — Does story point estimation improve when each project keeps its own prediction
   head, instead of federating the head along with the representation?
3. **RQ3** — How does federating only LoRA adapters compare with fully fine-tuning and
   federating the whole encoder, in estimation quality and in communication cost?

*New-project onboarding (leave-one-project-out, head-only adaptation over a history budget)
is reported as a case study, not an RQ — a single holdout cannot support a general claim.*

---

## Optional RQs

Not committed to. Every one below is already implemented and reachable with a **single
flag** — no code changes — and each costs roughly one extra root (~3 GPU-h at 3 seeds)
unless noted. Ordered by how much they would strengthen the thesis.

**O1** — Does freezing the LoRA A matrices, so that averaging client updates is exact,
change estimation quality compared with averaging both A and B separately?
*`--no-ffa-lora`. The exactness of `avg(B·A) = avg(B)·A` is the argument the method rests
on, and it is currently asserted from the literature rather than shown here.*

**O2** — How much of the benefit comes from adapting the encoder at all, rather than from
training only the head and the categorical embeddings on top of a frozen one?
*`--no-lora --freeze-encoder`. Adds a third point to RQ3's axis (frozen → LoRA → full
fine-tuning) and answers the obvious challenge that the adapters may be doing nothing. By
far the cheapest run in the plan: gradients stop at the head, so the encoder is
forward-only.*

**O3** — Does treating the five story points as ordered thresholds estimate better than
treating them as unordered classes?
*`--head-type ce` against the locked CORN head. Worth reporting even if macro-F1 is flat —
the ordinal structure should show up in MAE and quadratic κ first, and that asymmetry is
itself the finding.*

**O4** — Does centralized pre-training on one large project earn its place, given that it
is the one step where a project's data is pooled?
*`--run-no-warmstart-fl`. Runs both arms in one invocation with matched client selection,
so it is a clean comparison — but it roughly doubles that run's federated time. The privacy
framing is the interesting part: the warm-start is the only place the design pools data.*

**O5** — Does estimation quality hold up when only a fraction of the projects take part in
each round, as it would with real stragglers and dropouts?
*`--clients-per-round-fraction 0.5`. Standard practice in the FL literature; full
participation is currently an unexamined assumption. Roughly halves the per-round cost.*

**O6** — Does more local computation between synchronizations help or hurt under this
degree of client heterogeneity?
*`--local-epochs 3` / `5`. The drift question from the other direction than μ: μ restrains
drift by regularizing, local epochs increase the opportunity for it.*

**O7** — What does input truncation cost, and does it fall hardest on the high story-point
issues that tend to be the longest?
*`--max-length 128` against the locked 256. p90 token length is ≈164, so 128 truncates a
substantial tail — and those issues are disproportionately the complex, high-point ones.*
