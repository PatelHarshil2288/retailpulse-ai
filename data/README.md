# Data Directory

This directory documents the datasets used by RetailPulse AI.

## Primary dataset

**UCI Online Retail**

Source: https://archive.ics.uci.edu/dataset/352/online+retail

The dataset contains historical retail transactions used for sales analysis, customer segmentation, and product analytics.

## Data handling rules

- Keep large raw dataset files out of GitHub.
- Do not upload private, confidential, or personally identifiable information.
- Document data cleaning decisions and missing-value handling.
- Record the source and applicable license terms.
- Keep raw data separate from processed analytical outputs.

Run `python prepare_deployment.py` from the repository root to download the dataset when needed and generate the processed CSV files. The Docker image runs the same preparation script during its build.
