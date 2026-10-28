# Attribution and Responsible Portfolio Use

## Authorship boundary

The original paper context was collaborative. This repository must not be used to claim sole authorship of that paper, its shared research idea, its original manuscript, or work performed by other contributors.

The defensible personal portfolio claim is narrower:

> Independently reconstructed and audited the PEAD data and modeling pipeline, including event-key validation, transcript segmentation, fixed-model sentiment inference, event-time market features, point-in-time fundamentals, leakage-safe evaluation, automated tests, and evidence-based reinterpretation.

Only use that claim if you can explain the implementation and methodological choices in an interview. Be ready to discuss why event time matters, why the legacy SUE proxy was invalid, how train-only transformations prevent leakage, why clustered standard errors were used, and why feature importance is not causal evidence.

## AI-assisted development

The reconstruction used AI-assisted software development under human direction. The portfolio owner remains responsible for understanding, validating, and accurately describing the code and results. Do not represent generated implementation details as independently written from memory if an application or course explicitly requires disclosure of AI assistance.

## Data and model provenance

- Transcript records originate from an external Motley Fool earnings-call dataset. Redistribution rights must be checked at the original source before publishing raw text.
- Market, earnings, and share histories were obtained through Yahoo Finance via `yfinance`; cached responses are excluded from the public repository.
- Sentiment inference uses the `ProsusAI/finbert` model at the fixed revision recorded in the Phase 3 quality report. The model's own license and attribution requirements continue to apply.
- Collaborative manuscripts, personal records, legacy working tables, raw transcripts, downloaded market caches, and model weights are excluded by `.gitignore`.

## Licensing

No open-source license is assigned by default because the ownership and redistribution status of every inherited component has not been established. Publishing the repository without a license allows others to inspect it but does not grant broad reuse rights. Add a software license only after separating code that you are authorized to license from collaborative or third-party material.
