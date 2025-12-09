BuildPoolDatasets: Per-pool dataset builder

Overview
- Generates CSVs for each training pool using the same default sizes as epoch_sampler.
- Writes outputs to `TrainingData/pools/<pool>.csv`.
- Loads the think model once and inserts assistant `<think>` turns when `target` lacks `<think>`.
- Verbose logging and a simple progress bar per pool.

Adjusting sizes
- Edit the `POOL_SIZES` dict at the top of `TrainingScripts/BuildPoolDatasets.py` for absolute counts.
- Optionally tune per-pool percentages via `POOL_PCT` (0-100) without using CLI.

Pools
- `hh`: Anthropic HH identity-style mapping
- `self_cognition`: Modelscope self-cognition
- `open_thoughts`: OpenThoughts-114k (thinking)
- `rag_bespoke`: Bespoke-Stratos-17k (RAG-style)
- `tools_glaive`: Glaive function-calling v2
- `intel_magicoder`: Magicoder Evol Instruct
- `intel_reclor`: ReClor (logic_text mapping)
- `intel_openmix`: Infinity-Instruct + Open-Platypus (half/half)
- `create_dolphin`: Dolphin style data
- `create_airoboros`: Airoboros stream
- `enigmata`: Enigmata Data
- `empathetic`: Facebook empathetic_dialogues

Outputs
- CSV columns: `messages`, `target`, `source`.
- Assistant `<think>` content is generated and inserted into `messages` when missing.

Notes
- Errors are surfaced; there are no silent fallbacks.
- Large pools may take time; the progress bar shows processed/total and ETA.
