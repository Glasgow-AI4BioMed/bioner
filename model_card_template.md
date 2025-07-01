---
task: token-classification
tags:
- biomedical
- bionlp
license: mit
base_model: {base_model}
---

# {model_name}

This is a named entity recognition model fine-tuned from the [{base_model}](https://huggingface.co/{base_model}) model. It predicts spans with {label_count} possible labels. The labels are **{nice_labels}**.

The code used for training this model can be found at https://github.com/Glasgow-AI4BioMed/bioner along with links to other biomedical NER models trained on well-known biomedical corpora. The source dataset information is below.

## Example Usage

The code below will load up the model and apply it to the provided text. Notably, it uses an aggregation strategy to 

```python
from transformers import pipeline

# Load the model as part of an NER pipeline
ner_pipeline = pipeline("token-classification", 
                        model="Glasgow-AI4BioMed/{model_name}",
                        aggregation_strategy="simple")

# Apply it to some text
ner_pipeline("EGFR T790M mutations have been known to affect treatment outcomes for NSCLC patients receiving erlotinib.")
```

## Dataset Info

{dataset_info}

## Performance

The performance on the test split for the different labels are shown in the table below. This shows the individual B- (begin) and I- (inside) token-level labels with IOB2 labelling.

{test_report}

The performance on the training and validation splits are available in the corresponding files in the model repo.

## Hyperparameters

Hyperparameter tuning was done with [optuna](https://optuna.org/) and the [hyperparameter_search](https://huggingface.co/docs/transformers/en/hpo_train) functionality. {n_trials} trials were run. The best performing model was selected using the macro F1 performance on the validation set. The selected hyperparameters are in the table below.

{hyperparameter_table}
