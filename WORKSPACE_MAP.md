# Workspace Map

| Path | Purpose | Publish to GitHub |
|---|---|---|
| `src/pead_pipeline/` | Reusable event, transcript, FinBERT, market, fundamental, and modeling logic | Yes |
| `scripts/` | Phase runners and release builder | Yes, except local outputs |
| `tests/` | Automated validation suite | Yes |
| `reports/` | Local quality reports; the release includes sanitized Markdown copies only | Sanitized Markdown only |
| `data/raw/` | Source transcripts and provider caches | No |
| `data/interim/` | Audit checkpoints and temporary results | No |
| `data/processed/` | Event-level generated datasets and predictions | No |
| `data/models/` | Downloaded FinBERT weights | No |
| `private/manuscripts/` | Collaborative manuscript files | No |
| `private/legacy/` | Legacy comparison tables and obsolete script | No |
| `private/archive/` | Superseded or privacy-audit-failed release copies | No |
| `deliverables/github/pead-portfolio/` | Inspectable sanitized repository copy | Use the ZIP instead |
| `deliverables/github/pead-portfolio-github.zip` | Final GitHub upload package | Yes |

The project root is the private working directory, not the public repository. When publishing, extract the final ZIP into a new folder and initialize Git there.

