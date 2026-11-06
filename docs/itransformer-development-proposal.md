# iTransformer candidates for a future development study

Status: methodological proposal, not an implementation, training authorization,
or change to the frozen DirectFlow experiment. No raw assessment arrays or checkpoint files were opened for this proposal;
it uses the current model/input source code and primary literature checked on
November 6, 2026. The research question is informed by the already known
assessment findings.

## What would change

iTransformer is worth testing, but the literature does not establish that it is
better suited to these gauges or to extreme discharge. Its central change is
the attention axis: each variable's whole lookback becomes one token, and
self-attention relates variable tokens. The feed-forward layers process their
representations. The original study evaluated multivariate time-series
benchmarks, not this issued-weather discharge experiment. Attention weights
remain associations learned by the model, not causal explanations.
[Original paper](https://arxiv.org/html/2310.06625v4)

The authors' reference implementation is encoder-only with a linear projection
to the requested forecast length. It does not contain a GRU decoder. Optional
per-window normalization and inverse normalization use the input series' own
mean and standard deviation; that is a separate design choice from our saved
train-only affine transformation.
[Official model](https://github.com/thuml/iTransformer/blob/main/model/iTransformer.py)

The current DirectFlow Transformer embeds each historical hour's Q, age and
masks, adds temporal positions, and attends across 168 history tokens. Its
Transformer horizon decoder has 18 issued-weather/soil tokens and cross-attends
to that history; all issued horizons can interact. GRU–Transformer and
Mamba2–Transformer replace the history encoder while retaining that decoder.
These facts are directly inspectable in
[directflow.py](../src/hydra_directflow/models/directflow.py) and
[training.py](../src/hydra_directflow/training.py).

## The input contract matters more than the name

Our per-gauge history contains **one physical time series, discharge**, plus
age metadata. A Q-only inverted encoder has one substantive variable token;
its cross-variable attention therefore has little to exploit. Treating the
deterministic age sequence or constant complete-case masks as additional
hydrologic variables would overstate the multivariate input information.

Issued HRRR weather is a separate 18-by-4 forecast block, not 168 hours of
observed weather. Soil is a separately timestamped initial state, not a future
soil series. A full-input adaptation could use a Q-history token, four
issued-weather tokens and a soil-state token, with separate length-appropriate
projections, type/time metadata and explicit masks. This is a **heterogeneous
inverted-attention adaptation**, not an unchanged reproduction of iTransformer.
Do not manufacture historical weather by padding, extend soil into a fictitious
forecast, or add cross-gauge histories without a new input study.

The official inverted embedding uses a linear map whose input dimension is the
configured lookback. Longer-lookback experiments do not mean a saved L168
checkpoint can simply accept L24 without a compatible architecture and fitting
plan. [Official embedding](https://github.com/thuml/iTransformer/blob/main/layers/Embed.py)

## iTransformer encoder with a GRU horizon decoder

This is a plausible new hypothesis: variable-level context might feed a gated
sequential horizon state, allowing continuity across H1–18. The original gated
RNN work motivates reset/update memory mechanisms, not superiority for this
hydrologic hybrid. [Original gated encoder–decoder](https://arxiv.org/html/1406.1078v3)

A conservative prototype would initialize the GRU from the encoded input
context, then consume the known issued weather, soil/time metadata and horizon
embedding at each step. A scalar head produces discharge at every horizon.
Only the hidden state carries across steps: **no observed future discharge,
target teacher forcing, NWM predictor or residual add-back**. A GRU fed only
weather up to step h would have a different information route from our current
full-horizon attention decoder. Access to all weather already known at C must
therefore be specified explicitly, for example through a common forecast
context, and held comparable in the relevant controls.

## First check the output representation

All four current families end in `Linear(LayerNorm(hidden))`. A linear final
layer does not by itself make this composition unbounded. For fixed parameters,
let `a = head_weight * norm_weight`, `c = head_bias + head_weight·norm_bias`
and width `d`. In exact arithmetic the standardized output obeys

`|prediction - c| <= sqrt(d) * ||a - mean(a)||₂`.

This follows from the normalized vector's zero mean and norm no larger than
`sqrt(d)`; positive epsilon tightens the norm. The saved affine inverse maps
this to a fixed physical-Q bound. This is a mathematical architecture property,
not a finite-precision certificate or evidence yet that the bound constrains a
particular observed peak. The actual parameter bounds and saved forecasts
should be checked before attributing attenuation to the attention axis.
[LayerNorm definition](https://docs.pytorch.org/docs/2.15/generated/torch.nn.LayerNorm.html)

The reference iTransformer also has an encoder normalization before its output
projection. Copying it while retaining a fixed global inverse transform could
retain an amplitude restriction. Its optional input-window inverse normalization
changes that situation, but a normalization change and an encoder change must
be isolated experimentally. Neither should be introduced as an assessment-era
repair to the saved models. These are ceilings for fixed learned parameters,
not universal numerical caps for an architecture family: newly learned gains
and head weights can change them.

Removing the terminal LayerNorm alone also does not guarantee unrestricted
amplitude. An ordinary LSTM hidden output is bounded by its gated tanh. A GRU
state remains within [-1,1] when initialized there, because its sigmoid update
gate mixes the previous state with a bounded tanh candidate. A fixed linear
head on those states has a corresponding finite range. An arbitrary unbounded
initial GRU state is an exception to that particular [-1,1] argument; the actual
initialization must be specified. Explicit amplitude carry or origin-dependent
inverse normalization would therefore need its own controls, not a claim that
switching to a GRU decoder automatically resolves extreme-flow extrapolation.
[Original GRU equations](https://arxiv.org/html/1406.1078v3),
[PyTorch LSTM equations](https://docs.pytorch.org/docs/2.8/generated/torch.nn.LSTM.html)

## Controlled next study, after the diagnostic report

| Comparison | What it could distinguish |
|---|---|
| Existing temporal encoder vs inverted-style encoder, same horizon decoder and head | History representation / attention-axis contribution |
| Inverted-style encoder + direct horizon head vs + GRU decoder, matched known inputs | Added gated horizon decoding |
| GRU encoder + same GRU decoder vs inverted-style encoder + GRU decoder | Hybrid contribution beyond the recurrent component |
| Same architecture with alternative output/normalization design | Amplitude representation versus encoder effects |

Use development-only rolling folds, identical admitted origins, equal tuning
opportunity and all declared seeds. Freeze each input/normalization/head choice,
loss and selection rule before fitting. Benchmark realistic CPU/MPS batches;
fewer attention tokens do not guarantee lower end-to-end cost, particularly
with a sequential GRU decoder. Record parameter counts and measured training
and inference time rather than assuming equal width means equal capacity.

Peak attenuation could also reflect rare-event support, distribution shift,
weather/soil information, spatial approximation or timing. Event diagnostics
can localize these possibilities but cannot establish that swapping architecture
will fix them. A favorable development comparison would support the candidate;
inconsistent seeds, no matched gain, clean-input degradation or persistent peak
attenuation would weaken the case.

The 2024–25 assessment has now informed research questions. Even if subsequent
weights use only development data, new models must not be selected or tuned on
that assessment, and it cannot become a fresh confirmatory holdout again merely
by issuing a new protocol hash. Confirmation of a later design needs genuinely
untouched evidence or an explicitly labeled exploratory follow-up. No new fits,
downloads, paid jobs, data-policy changes or persistence baseline are authorized
by this proposal.
