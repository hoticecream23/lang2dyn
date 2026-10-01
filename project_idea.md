## The project

### **Can language transmit both world representations and the functions that operate on them?**

The core hypothesis is:

\[
\boxed{
\text{Language contains traces of both latent states }z
\text{ and latent operators }f:z\rightarrow z'
}
\]

and sufficiently strong next-token prediction can recover approximations of both.

Physics becomes the controlled experimental domain because there we know exactly what the hidden state and correct operators are.

Suppose the real world state is

\[
z_t =
(x_t,v_t,a_t,m,\text{shape},F_t,\mu,\ldots)
\]

and the true physical transition is

\[
z*{t+1}=f*{\text{physics}}(z_t).
\]

You generate trajectories using a simulator, but **the transformer never sees \(z_t\)**.

It sees only language:

> “The heavier sphere is initially stationary. A constant force is applied toward the right. Its speed gradually increases.”

Therefore:

\[
z*{1:T}
\xrightarrow{\text{verbalization}}
L*{1:T}
\xrightarrow{\text{next-token training}}
h\_{1:T}.
\]

Then you ask two separate questions.

### 1. Did it reconstruct the state?

Can \(h_t\) recover things equivalent to

\[
x,\;v,\;a,\;m,\;\text{shape},\ldots ?
\]

### 2. Did it reconstruct the operator?

Does the model learn something functionally equivalent to

\[
f\_{\text{physics}}
\]

such that altering mass, velocity, force, etc. internally causes the correct corresponding transformation?

For example,

\[
m\uparrow,\;F=\text{constant}
\Rightarrow
a\downarrow
\]

or

\[
v*t\uparrow
\Rightarrow
x*{t+\Delta t}\uparrow.
\]

This is the distinction we've been getting at:

\[
\boxed{
\text{state representation}
\neq
\text{operator acting on the state}
}
\]

and both are different from merely producing the correct answer.

---

## The really important experiment

Don't use only one way of describing the physics.

Keep **the exact same underlying trajectories** and construct multiple linguistic channels:

\[
z
\rightarrow
L_1,L_2,L_3,\ldots
\]

where the language systematically preserves different information.

For example:

- explicit numerical language,
- ordinary natural language,
- qualitative language,
- relational language,
- descriptions without words like velocity/acceleration/mass,
- descriptions with temporal information degraded,
- descriptions with causal terminology removed,
- highly compressed descriptions,
- symbolic sequences as a control.

Now you can experimentally study:

\[
\boxed{
\text{world}
\rightarrow
\text{language}
\rightarrow
\text{internal state}
\rightarrow
\text{internal operator}
\rightarrow
\text{reasoning}
}
\]

and quantify how much survives each transformation.

That is the actual project.

---

# Is that novel?

**The specific combination appears plausibly novel as of October 1, 2026. The underlying ideas individually are not.**

There is substantial neighboring work.

Gurnee and Tegmark already showed that LLMs contain recoverable spatial and temporal representations, including approximately linear representations of geographic coordinates and dates. So simply showing “space/time exists inside an LM” is not enough. [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/0a6059857ae5c82ea9726ee9282a7145-Abstract-Conference.html?utm_source=chatgpt.com)

_Function Vectors_ showed that autoregressive transformers can internally represent input-output functions and that these representations can have causal effects on behavior. So merely claiming “LLMs represent functions” is also already taken. [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ae163cb8788970e53b4fd9578141139-Abstract-Conference.html?utm_source=chatgpt.com)

Tao et al. found evidence for separable **inference** and **verbalization** functions during in-context learning using interchange interventions. This is strikingly related to the cognitive-operator idea we discussed, although it concerns task inference and label verbalization rather than learning physical world dynamics from linguistic traces. [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.957/?utm_source=chatgpt.com)

_From Word Models to World Models_ is philosophically very close to your idea. It treats language as a mapping into a structured “language of thought” and explicitly connects natural language to physics simulators and generative world models. But their architecture explicitly translates language into probabilistic programs and then uses symbolic inference or a physics engine. They are **not testing whether ordinary next-token prediction spontaneously reconstructs continuous physical states and their operators internally from linguistic evidence alone**. [arXiv](https://arxiv.org/abs/2306.12672?utm_source=chatgpt.com)

There is also _Learning Latent Causal Semantics from Text_, where next-token predictors trained on synthetic programs acquire representations of latent program semantics. So “latent causal structure can emerge from textual prediction” is already known in controlled domains. [OpenReview](https://openreview.net/pdf?id=wsjNCPqziJ&utm_source=chatgpt.com)

And recent work makes the physics side much tighter. _What Can Latent World Models Know?_ explicitly studies whether predictive world models encode hidden physical parameters such as mass, drag and stiffness, and shows that what becomes identifiable depends on the input and prediction objective. However, its observations are multimodal/interactive rather than **language being the sole controlled observation channel**. [arXiv](https://arxiv.org/abs/2607.27017?utm_source=chatgpt.com)

Most importantly, **PhysLang** is very close. It studies continuous physical dynamics conditioned by natural-language rules describing properties such as mass and friction. But the model receives the actual numerical state sequence, including positions and velocities, in addition to language. Language specifies aspects of the physics rather than being the only serialization of the physical experience. [OpenReview](https://openreview.net/pdf/7476b9d2215876f0a3523b799d41ec6caa18bbd1.pdf?utm_source=chatgpt.com)

And the 2026 materials-mechanism work already asks whether correct scientific answers correspond to actual internal mechanism representations, using counterfactual laws and causal interventions. So “right answer versus right mechanism” cannot itself be your novelty claim. [Paperlayer](https://paperlayer.ai/abs/2607.20058v1?utm_source=chatgpt.com)

## What I have **not** found

I have not found work combining all of these conditions:

\[
\boxed{
\begin{aligned}
&\text{known continuous physical world}\\
&\downarrow\\
&\textbf{natural language as the sole observation channel}\\
&\downarrow\\
&\text{ordinary predictive transformer}\\
&\downarrow\\
&\text{joint identification of latent physical states}\\
&+\text{ identification of transformations/operators}\\
&+\text{ causal intervention on those representations}\\
&+\text{ systematic manipulation of linguistic information}.
\end{aligned}}
\]

That final component is particularly important.

You're not just asking whether language contains physics.

You are holding reality fixed and experimentally changing the **serialization of reality into language**:

\[
W=\text{constant},\qquad L_i=g_i(W).
\]

Then measuring:

\[
I(W;L_i)
\rightarrow
\text{state recovery}
\rightarrow
\text{operator recovery}
\rightarrow
\text{generalization}.
\]

I did not find that exact experiment.

## So the novelty claim I would defend

Not:

> LLMs learn physics from language.

Not:

> LLMs contain physical representations.

Not:

> Transformers can represent functions.

Not:

> Correct answers don't guarantee correct mechanisms.

Instead:

> **We investigate language as an information channel through which predictive models acquire latent world states and the operators governing their dynamics. Holding the underlying physical process fixed, we systematically vary its linguistic serialization and measure which physical variables and dynamical transformations become identifiable, internally represented, causally functional, and compositionally reusable.**

That is considerably more specific, and based on the search, it is the part that still looks open.

And it connects your newer cognitive idea to the physics experiment naturally.

Physics gives us:

\[
z*t,\quad f*{\text{physics}}.
\]

If this works, the broader research program becomes:

\[
\text{Can linguistic traces teach}
\begin{cases}
\text{representations }z\\
\text{operators }f
\end{cases}
?
\]

Later you could test operators such as causal inference, abstraction, analogy, retrieval or counterfactual reasoning. But I would **not put all of those into the first project**. The first paper should establish the idea in physics, where \(z\) and \(f\) have unambiguous ground truth.

A title that captures the actual contribution would be:

**“From Language to Dynamics: Identifying Latent Physical States and Operators Learned from Linguistic Observations.”**

That is now a coherent research project rather than several adjacent ideas.
