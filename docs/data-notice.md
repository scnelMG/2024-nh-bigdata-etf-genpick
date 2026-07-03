# Data Notice

This repository is a public-safe portfolio version of a 2024 NH Investment & Securities Big Data Competition project. The original work used NH-provided ETF and stock tables. Those raw source tables are not redistributed here.

## Public Files Included

The repository includes:

- analysis and pipeline code,
- selected derived CSV outputs under `results/`,
- presentation-derived images under `assets/`,
- existing presentation, report, and demo assets,
- method and portfolio documentation.

These files are intended for project inspection. They are not a complete public dataset.

## Excluded Materials

The following materials must not be published from local Drive or contest folders:

- original NH source tables and raw contest data,
- NH security pledge PDFs,
- NH data destruction pledge PDFs,
- signed agreements, private PDFs, or personal/team documents,
- raw source bundles, archive folders, or large zip files,
- `.env` files, private keys, real credentials, or service access values,
- unreviewed Drive folders or copied `.git` directories.

## Credential Boundary

The generative AI step requires a locally configured Azure OpenAI-compatible credential. The repository may contain placeholder strings or environment-variable names so the code can explain how to run, but no real credential should be committed.

Any value that grants access to a service must remain outside Git and should be supplied only through the local runtime environment.

## Reuse Boundary

The derived CSV outputs can be inspected to understand the method and evidence. They should not be treated as a substitute for the original NH data, and they should not be used to make investment decisions.

Before any external publication, re-run the safety checks for forbidden file names, large files, and credential patterns, then review the assets for contest or personal information.
