# GitHub Publishing Guide

## Recommendation

Publish the sanitized project as a separate repository named `auditable-pead-pipeline`. A public repository is useful for internship and graduate-school applications because reviewers can inspect the code, tests, audit trail, and honest negative findings. Use a private repository instead if the original project team has asked that derivative code remain non-public or if an application explicitly restricts public sharing.

Only publish the contents of `deliverables/github/pead-portfolio-github.zip`. Do not publish the whole working directory.

## Suggested repository settings

- Repository name: `auditable-pead-pipeline`
- Description: `Auditable earnings-call PEAD pipeline using FinBERT, event-date-filtered financial features, and chronological model validation.`
- Topics: `financial-nlp`, `finbert`, `event-study`, `machine-learning`, `reproducibility`, `model-audit`, `pead`
- Visibility: public, subject to the collaboration and data-licensing boundary above
- License: leave unlicensed until ownership of all inherited code and materials is confirmed

## Upload steps

1. Create a new empty GitHub repository without an auto-generated README or license.
2. Extract `pead-portfolio-github.zip` into a new local folder.
3. Confirm that the folder contains `README.md`, `PORTFOLIO.md`, `ATTRIBUTION.md`, source code, tests, requirements, and Markdown reports only.
4. Initialize Git in that extracted folder and push it to the empty repository.
5. Pin the repository on the GitHub profile and add the repository URL to the CV project entry.

Example commands, run inside the extracted folder after replacing the placeholder URL:

```powershell
git init
git add .
git commit -m "Publish auditable PEAD pipeline"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

## What belongs on GitHub

- Pipeline source code and tests
- Requirements files and fixed model configuration
- Markdown quality reports with aggregate counts and hashes
- Portfolio explanation, limitations, and attribution statement
- Reproduction instructions describing where users must provide their own data

## What must remain private

- Collaborative manuscript files and unpublished writing
- Raw or sectioned transcript text
- Legacy working tables and event-level generated CSV files
- Yahoo request caches and downloaded model weights
- Resumes, identification documents, and local machine paths
- Any file inside `private/` or `data/`

## Claims to use publicly

Describe the work as an independent reconstruction and model audit conducted in a collaborative research context. Do not claim sole authorship of the original paper. Do not describe portfolio-sort diagnostics as realizable investment returns, and do not claim that the ensemble universally outperforms its component models.
