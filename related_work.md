# Related work: audit, bibliography, draft section

Verification method: every entry below was checked by opening the arXiv / ACL Anthology / ICLR / PMLR / NeurIPS / JMLR page (WebFetch) or, where noted, from a WebSearch result. OpenReview pages were blocked by a bot check in this environment, so entries that only exist on OpenReview are marked accordingly. Checked 2026-10-04.

## Section A: citation audit of project_idea.md

project_idea.md contains 8 linked citations (all links carry a `?utm_source=chatgpt.com` tracking suffix, which should be stripped).

| Citation as written | Status | Correct reference / note | Link |
|---|---|---|---|
| Gurnee and Tegmark, space/time representations in LLMs (ICLR Proceedings) | Verified | "Language Models Represent Space and Time", Gurnee and Tegmark, ICLR 2024. Abstract: linear representations of space and time in Llama-2 models. Claim matches. | https://proceedings.iclr.cc/paper_files/paper/2024/hash/0a6059857ae5c82ea9726ee9282a7145-Abstract-Conference.html ; https://arxiv.org/abs/2310.02207 |
| _Function Vectors_ (ICLR Proceedings) | Verified | "Function Vectors in Large Language Models", Todd, Li, Sen Sharma, Mueller, Wallace, Bau, ICLR 2024. Compact causal vectors representing input-output functions; claim matches. | https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ae163cb8788970e53b4fd9578141139-Abstract-Conference.html ; https://arxiv.org/abs/2310.15213 |
| Tao et al., inference and verbalization functions (ACL Anthology) | Verified (minor note) | "Inference and Verbalization Functions During In-Context Learning", Tao, Chen, Liu, Findings of EMNLP 2024. Abstract says "layer-wise intervention experiments"; I could not confirm the specific phrase "interchange interventions" from the abstract alone. | https://aclanthology.org/2024.findings-emnlp.957/ |
| _From Word Models to World Models_ (arXiv 2306.12672) | Verified | Wong, Grand, Lew, Goodman, Mansinghka, Andreas, Tenenbaum, June 2023. Language is translated to probabilistic programs ("probabilistic language of thought") and uses symbolic inference and physics simulators. The characterisation matches the abstract. | https://arxiv.org/abs/2306.12672 |
| _Learning Latent Causal Semantics from Text_ (OpenReview wsjNCPqziJ) | Exists, page not opened | A WebSearch result lists "Learning Latent Causal Semantics from Text: An Empirical Study of Next-Token Predictors Trained on Programs", Charles Jin and Martin Rinard, with the abstract described in the project file (grid-world DSL, linear probe recovers program states). The search result's venue ("ICLR 2024") could NOT be confirmed because OpenReview was blocked; do not cite a venue until checked. Same authors' verified related papers: arXiv 2305.11169 (ICML 2024) and arXiv 2407.13765 (COLM 2024). | https://openreview.net/forum?id=wsjNCPqziJ |
| _What Can Latent World Models Know?_ (arXiv 2607.27017) | Verified | Real. "What Can Latent World Models Know? Physical Information in Multimodal Predictive Representations", Tan, Xu, Xu, Tao, Li, Hong, Feng, Du, Wang; v1 29 Jul 2026, v4 26 Sep 2026. PokeWorld simulated robot env with mass, drag, contact stiffness; findings depend on prediction targets/horizon. Project-file claim (mass, drag, stiffness; multimodal/interactive, not language-only) matches. Note the paper reports drag has weak latent readout yet still affects forecasts, which parallels our decodable-but-inert dissociation. | https://arxiv.org/abs/2607.27017 |
| **PhysLang** (OpenReview pdf 7476b9d2...) | **Not found (likely fabricated or mis-named)** | OpenReview PDF blocked by bot check. Searches for "PhysLang" (plain, with arXiv/OpenReview domain filters, and by the described content: language-specified mass/friction plus numerical state) returned no such paper. Closest hit is unrelated: "Physis-Lang: Self-Evolving Language as a Physical Representation for Video World Model" (arXiv 2609.40358), a video-captioning framework, which does not match the described paper. Do not cite until the PDF can be opened. | none (closest unrelated: https://arxiv.org/abs/2609.40358) |
| "2026 materials-mechanism work" (Paperlayer 2607.20058) | Verified (citation form wrong) | Real arXiv paper: "Reading and Steering Representations of Materials-Science Mechanisms in an Open-Weight Language Model", Markus J. Buehler, 22 Jul 2026 (rev. 3 Sep 2026). Gemma-4 models; 60 counterfactual laws; bidirectional interventions. Claim matches. Cite as arXiv, not the paperlayer.ai aggregator. | https://arxiv.org/abs/2607.20058 |

Tally: 6 verified (Gurnee-Tegmark, Function Vectors, Tao et al., Wong et al., arXiv 2607.27017, arXiv 2607.20058), 1 exists but not fully opened (Jin and Rinard, venue unconfirmed), 1 not found (PhysLang), 0 with wrong titles/authors among those opened. The two "suspicious" IDs from the brief (2607.27017 and 2607.20058) are real. Only PhysLang failed. Note the project file's "as of October 1, 2026" novelty statement is not independently checked here.

## Section B: verified bibliography

### World models and what LMs represent
- Vafa, Chang, Rambachan, Mullainathan (2025). What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models. ICML 2025 (PMLR 267). https://arxiv.org/abs/2507.06952 ; https://proceedings.mlr.press/v267/vafa25a.html
- Vafa, Chen, Rambachan, Kleinberg, Mullainathan (2024). Evaluating the World Model Implicit in a Generative Model. NeurIPS 2024. https://arxiv.org/abs/2406.03689
- Li, Nye, Andreas (2021). Implicit Representations of Meaning in Neural Language Models. ACL-IJCNLP 2021. https://aclanthology.org/2021.acl-long.143/ ; https://arxiv.org/abs/2106.00737
- Li, Hopkins, Bau, Viegas, Pfister, Wattenberg (2023). Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task. ICLR 2023. https://arxiv.org/abs/2210.13382
- Nanda, Lee, Wattenberg (2023). Emergent Linear Representations in World Models of Self-Supervised Sequence Models. BlackboxNLP 2023. https://arxiv.org/abs/2309.00941 ; https://aclanthology.org/2023.blackboxnlp-1.2/
- Gurnee, Tegmark (2024). Language Models Represent Space and Time. ICLR 2024. https://arxiv.org/abs/2310.02207
- Jin, Rinard (2024). Emergent Representations of Program Semantics in Language Models Trained on Programs. ICML 2024. https://arxiv.org/abs/2305.11169

### Language models and physics / dynamics
- Song, Bae, Kim, Jeong (2025, rev. 2026). Uncovering Spontaneous Physics Representations in In-Context Learning. arXiv. https://arxiv.org/abs/2508.12448 (LLM activations correlate with energy during in-context dynamics forecasting).
- Tan et al. (2026). What Can Latent World Models Know? Physical Information in Multimodal Predictive Representations. arXiv. https://arxiv.org/abs/2607.27017
- Buehler (2026). Reading and Steering Representations of Materials-Science Mechanisms in an Open-Weight Language Model. arXiv. https://arxiv.org/abs/2607.20058
- Wong et al. (2023). From Word Models to World Models: Translating from Natural Language to the Probabilistic Language of Thought. arXiv. https://arxiv.org/abs/2306.12672

### Causal abstraction, DAS, interchange interventions
- Geiger, Wu, Potts, Icard, Goodman (2024). Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations (DAS). CLeaR 2024, PMLR 236. https://proceedings.mlr.press/v236/geiger24a.html ; https://arxiv.org/abs/2303.02536
- Geiger, Ibeling, Zur, et al. (2025). Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability. JMLR 26(83). https://www.jmlr.org/papers/v26/23-0058.html ; https://arxiv.org/abs/2301.04709
- Wu, Geiger, Icard, Potts, Goodman (2023). Interpretability at Scale: Identifying Causal Mechanisms in Alpaca (Boundless DAS). NeurIPS 2023. https://arxiv.org/abs/2305.08809
- Tao, Chen, Liu (2024). Inference and Verbalization Functions During In-Context Learning. Findings of EMNLP 2024. https://aclanthology.org/2024.findings-emnlp.957/

### Probing critiques
- Hewitt, Liang (2019). Designing and Interpreting Probes with Control Tasks. EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1275/
- Elazar, Ravfogel, Jacovi, Goldberg (2021). Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals. TACL 9:160-175. https://aclanthology.org/2021.tacl-1.10/
- Ravichander, Belinkov, Hovy (2021). Probing the Probing Paradigm: Does Probing Accuracy Entail Task Relevance? EACL 2021. https://aclanthology.org/2021.eacl-main.295/

### Illusions of subspace patching (and the reply)
- Makelov, Lange, Nanda (2024). Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching. ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/70b8505ac79e3e131756f793cd80eb8d-Abstract-Conference.html ; https://arxiv.org/abs/2311.17030
- Wu, Geiger, Huang, Arora, Icard, Potts, Goodman (2024). A Reply to Makelov et al. (2023)'s "Interpretability Illusion" Arguments. arXiv. https://arxiv.org/abs/2401.12631

### Steering and representation engineering
- Turner, Thiergart, Leech, Udell, Vazquez, Mini, MacDiarmid (2023, rev. 2024). Steering Language Models With Activation Engineering (ActAdd). arXiv. https://arxiv.org/abs/2308.10248
- Zou et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. arXiv. https://arxiv.org/abs/2310.01405
- Todd et al. (2024). Function Vectors in Large Language Models. ICLR 2024. https://arxiv.org/abs/2310.15213

### Compositional generalization
- Lake, Baroni (2018). Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks (SCAN). ICML 2018. https://arxiv.org/abs/1711.00350
- Hupkes, Dankers, Mul, Bruni (2019/2020). Compositionality Decomposed: How Do Neural Networks Generalise? arXiv. https://arxiv.org/abs/1908.08351
- Dziri et al. (2023). Faith and Fate: Limits of Transformers on Compositionality. arXiv. https://arxiv.org/abs/2305.18654

### Unverified (do not cite yet)
- Jin, Rinard. Learning Latent Causal Semantics from Text. OpenReview wsjNCPqziJ (venue unconfirmed).
- "PhysLang" (not found).

## Section C: draft Related work section

**Related work**

*World models in sequence models.* Whether next-token predictors recover the structure of the process that generated their data has been studied in games, programs, geography and physics. Othello-GPT models were shown to contain a board-state representation that can be intervened on (Li et al., 2023), later found to be linear in a "mine/theirs" basis (Nanda et al., 2023). Li, Nye and Andreas (2021) showed that language-model representations support linear readout of entity state and that editing them changes generated text, and Gurnee and Tegmark (2024) found linear space and time representations in Llama-2. Jin and Rinard (2024) trained on programs and used an interventional baseline to separate what a probe learns from what the model represents. Closest to our setting, Vafa et al. (2025) report that foundation models trained on orbital trajectories fit the sequences but do not apply Newtonian mechanics when adapted to new tasks, and Vafa et al. (2024) show that apparent world models can be incoherent under automaton-based metrics. Song et al. (2025) find energy-correlated activations when LLMs forecast dynamics in context, and Tan et al. (2026) show in an action-conditioned simulator that whether mass, drag or stiffness is decodable depends on the training targets. We differ in the data: the only observation channel is text (numbers or English) describing a one-dimensional cart, and we ask not only whether hidden mass and friction are decodable but whether they are used.

*Decodable is not causal.* The gap between probe success and use is established in the NLP literature (Hewitt and Liang, 2019; Ravichander et al., 2021; Elazar et al., 2021), and Tan et al. (2026) report a physical quantity with weak latent readout that still affects forecasts. Our first result (R1), that the ridge-probe direction for mass is decodable yet causally inert, is therefore an instance of a known phenomenon, not a new one; what we add is a controlled physical domain in which the ground-truth variables are known and the dissociation can be measured against a causally effective alternative.

*Distributed causal channels.* Distributed alignment search (Geiger et al., 2024), within the causal abstraction framework (Geiger et al., 2025), learns a rotated subspace in which interchange interventions realise a high-level variable. We use it to find a 32-dimensional middle-layer subspace where writing a value yields the physically correct counterfactual velocity. Makelov et al. (2024) show that subspace patching can succeed by activating a dormant pathway disconnected from the model's normal computation, so a successful DAS subspace need not be the representation the model reads; Wu et al. (2024) dispute that reading and attribute the cases to training and evaluation choices. We treat the debate as live. Our "write channel" (R3) is a claim about sufficiency of the written value under intervention. The illusion concern bites hardest on the weak single-layer transfer of natural activations between episodes (R4). It is eased by two results: writes bounded to the natural range still work (R3), and swapping the model's own activations at four layers transfers about half the effect (R4). We still do not claim the subspace is the model's only natural storage site. Steering vectors (Turner et al., 2023; Zou et al., 2023; Todd et al., 2024) show that adding directions changes behaviour, but they are typically derived from contrasts, not tested against a known simulator.

*Methodological point.* We found that interchange-intervention accuracy is inflated by regression to the mean when a patch merely erases a value, and use a source-difference metric. None of the sources opened for this review discuss this specific artifact, but we have not searched exhaustively, and it should be described as a caveat on the metric, not a claim of priority.

*Computation versus lookup, and composition.* R5 (held-out mass-by-force combinations are still computed correctly) is evidence against pure memorisation in a setting where Vafa et al. (2025) found heuristic solutions, but it concerns interpolation within seen factors. R6, failure when friction and force act together after training on each separately, fits prior evidence of weak systematic generalisation in neural sequence models (Lake and Baroni, 2018; Hupkes et al., 2020; Dziri et al., 2023). It differs in that the failing composition is of physical mechanisms with a known correct answer.

**Positioning (new versus known).** Known: decodability does not imply use; world-model-like representations can emerge from prediction; DAS and steering produce causal effects; subspace patches can mislead; compositional generalisation is weak. Plausibly new, pending a broader search: a text-only continuous physical domain with known hidden parameters in which probe-inert and DAS-causal directions coexist; the held-out-combination and friction-under-force composition tests in this domain; and the regression-to-the-mean correction for IIA.
