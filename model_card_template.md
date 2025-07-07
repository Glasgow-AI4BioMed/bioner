---
task: token-classification
tags:
- biomedical
- bionlp
license: mit
base_model: {base_model}
---

# {model_name}

This is a named entity recognition model fine-tuned from the [{base_model}](https://huggingface.co/{base_model}) model. {label_explanation}

The code used for training this model can be found at https://github.com/Glasgow-AI4BioMed/bioner along with links to other biomedical NER models trained on well-known biomedical corpora. The source dataset information is below.

## Example Usage

The code below will load up the model and apply it to the provided text. It uses a simple aggregation strategy to post-process the individual tokens into larger multi-token entities where needed.

```python
from transformers import pipeline

# Load the model as part of an NER pipeline
ner_pipeline = pipeline("token-classification", 
                        model="Glasgow-AI4BioMed/{model_name}",
                        aggregation_strategy="max")

# Apply it to some text
ner_pipeline("EGFR T790M mutations have been known to affect treatment outcomes for NSCLC patients receiving erlotinib.")

# Output:
{example_output}
```

## Dataset Info

{dataset_info}

## Performance

The span-level performance on the test split for the different labels are shown in the tables below. The full performance results are available in the model repo in Markdown format for viewing and JSON format for easier loading. These include the performance at token level (with individual B- and I- labels as the token classifier uses IOB2 token labelling).

{test_span_report}

## Hyperparameters

Hyperparameter tuning was done with [optuna](https://optuna.org/) and the [hyperparameter_search](https://huggingface.co/docs/transformers/en/hpo_train) functionality. {n_trials} trials were run. Early stopping was applied during training. The best performing model was selected using the macro F1 performance on the validation set. The selected hyperparameters are in the table below.

{hyperparameter_table}
