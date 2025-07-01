# A selection of biomedical NER models

This repo contains code for training NER models on a variety of well-known biomedical named entity recognition datasets. The models are fine-tuned token classification models that are available through Hugging Face Hub.

## Example Usage

The code below will load up the model and apply it to the provided text. Notably, it uses an aggregation strategy to 

```python
from transformers import pipeline

# Load the model as part of an NER pipeline
ner_pipeline = pipeline("token-classification", 
                        model="Glasgow-AI4BioMed/bioner_medmentions_st21pv",
                        aggregation_strategy="simple")

# Apply it to some text
ner_pipeline("EGFR T790M mutations have been known to affect treatment outcomes for NSCLC patients receiving erlotinib.")
```

## Available Models

| Model | Entity Types | Entity Count |
|-------|--------------|--------------|
| [medmentions_st21pv](https://huggingface.co/Glasgow-AI4BioMed/bioner_medmentions_st21pv) | A variety of broad biomedical concept categories |              |
| [medmentions_st21pv_finegrain](https://huggingface.co/Glasgow-AI4BioMed/bioner_medmentions_st21pv_finegrain)      | A large number of specific biomedical categories |              |
| [ncbi_disease](https://huggingface.co/Glasgow-AI4BioMed/bioner_ncbi_disease) | Diseases |              |
| [nlmchem](https://huggingface.co/Glasgow-AI4BioMed/bioner_nlmchem) | Chemicals |              |
| [bc5cdr](https://huggingface.co/Glasgow-AI4BioMed/bioner_bc5cdr) | Chemicals and diseases |              |
| [tmvar](https://huggingface.co/Glasgow-AI4BioMed/bioner_tmvar) | Mutations (plus genes, species, etc) |              |
| [gnormplus](https://huggingface.co/Glasgow-AI4BioMed/bioner_gnormplus) | Genes and gene families |              |

## Building the Models

The models can be built with a moderate GPU. The commands below outline what's needed to get the datasets, preprocess them and fine-tune the models.

### Prerequisites

Building the models requires several libraries including transformers and are listed in the [requirements.txt](https://github.com/Glasgow-AI4BioMed/bioner/blob/main/requirements.txt) file. These can be installed through pip with:

```bash
pip install -r requirements.txt
```

### Getting the data

The various datasets/corpora used to train the models can be downloaded used the [fetch_corpora.sh](https://github.com/Glasgow-AI4BioMed/bioner/blob/main/fetch_corpora.sh) script:

```bash
bash fetch_corpora.sh
```

### Preprocessing and training

The sections below provide the commands to preprocess and tune the model. More details are available for each model on their model page including model performance and selected hyperparameters.

#### MedMentions ST21pv

```bash
# Preprocess the data
python prepare_medmentions.py --medmentions_dir corpora_sources/medmentions/st21pv --semantic_groups corpora_sources/medmentions/SemGroups.txt --out_train datasets/medmentions_st21pv_train.bioc.xml.gz --out_val datasets/medmentions_st21pv_val.bioc.xml.gz --out_test datasets/medmentions_st21pv_test.bioc.xml.gz

# Tune the model and save it
python tune_ner.py --train_corpus datasets/medmentions_st21pv_train.bioc.xml.gz --val_corpus datasets/medmentions_st21pv_val.bioc.xml.gz --test_corpus datasets/medmentions_st21pv_test.bioc.xml.gz --n_trials 10 --model_name bioner_medmentions_st21pv --model_card_template model_card_template.md --dataset_info dataset_info/medmentions_st21pv.md
```

#### MedMentions ST21pv (finegrain)

```bash
# Preprocess the data
python prepare_medmentions.py --medmentions_dir corpora_sources/medmentions/st21pv --semantic_groups corpora_sources/medmentions/SemGroups.txt --out_train datasets/medmentions_st21pv_finegrain_train.xml.gz --out_val datasets/medmentions_st21pv_finegrain_val.bioc.xml.gz --out_test datasets/medmentions_st21pv_finegrain_test.bioc.xml.gz --finegrain

# Tune the model and save it
python tune_ner.py --train_corpus datasets/medmentions_st21pv_finegrain_train.bioc.xml.gz --val_corpus datasets/medmentions_st21pv_finegrain_val.bioc.xml.gz --test_corpus datasets/medmentions_st21pv_finegrain_test.bioc.xml.gz --n_trials 10 --model_name bioner_medmentions_st21pv_finegrain --model_card_template model_card_template.md --dataset_info dataset_info/medmentions_st21pv_finegrain.md
```

#### NCBI Disease

```bash
# Preprocess the data
python prepare_ncbi_disease.py --ncbidisease_dir corpora_sources/NCBI-disease --out_train datasets/ncbi_disease_train.bioc.xml.gz --out_val datasets/ncbi_disease_val.bioc.xml.gz --out_test datasets/ncbi_disease_test.bioc.xml.gz

# Tune the model and save it
python tune_ner.py --train_corpus datasets/ncbi_disease_train.bioc.xml.gz --val_corpus datasets/ncbi_disease_val.bioc.xml.gz --test_corpus datasets/ncbi_disease_test.bioc.xml.gz --n_trials 10 --model_name bioner_ncbi_disease --model_card_template model_card_template.md --dataset_info dataset_info/ncbi_disease.md
```

#### NLM-Chem

```bash
# Preprocess the data
python prepare_nlmchem.py --nlmchem_dir corpora_sources/NLM-Chem --out_train datasets/nlmchem_train.bioc.xml.gz --out_val datasets/nlmchem_val.bioc.xml.gz --out_test datasets/nlmchem_test.bioc.xml.gz

# Tune the model and save it
python tune_ner.py --train_corpus datasets/nlmchem_train.bioc.xml.gz --val_corpus datasets/nlmchem_val.bioc.xml.gz --test_corpus datasets/nlmchem_test.bioc.xml.gz --n_trials 10 --model_name bioner_nlmchem --model_card_template model_card_template.md --dataset_info dataset_info/nlmchem.md
```

#### BC5CDR

```bash
# Preprocess the data
python prepare_bc5cdr.py --bc5cdr_dir corpora_sources/CDR_Data/CDR.Corpus.v010516 --out_train datasets/bc5cdr_train.bioc.xml.gz --out_val datasets/bc5cdr_val.bioc.xml.gz --out_test datasets/bc5cdr_test.bioc.xml.gz

# Tune the model and save it
python tune_ner.py --train_corpus datasets/bc5cdr_train.bioc.xml.gz --val_corpus datasets/bc5cdr_val.bioc.xml.gz --test_corpus datasets/bc5cdr_test.bioc.xml.gz --n_trials 10 --model_name bioner_bc5cdr --model_card_template model_card_template.md --dataset_info dataset_info/bc5cdr.md
```

#### tmVar

```bash
# Preprocess the data
python prepare_tmvar.py --tmvar_corpus corpora_sources/tmVar3Corpus.txt --out_train datasets/tmvar3_train.bioc.xml.gz --out_val datasets/tmvar3_val.bioc.xml.gz --out_test datasets/tmvar3_test.bioc.xml.gz

# Tune the model and save it
python tune_ner.py --train_corpus datasets/tmvar3_train.bioc.xml.gz --val_corpus datasets/tmvar3_val.bioc.xml.gz --test_corpus datasets/tmvar3_test.bioc.xml.gz --n_trials 10 --model_name bioner_tmvar3 --model_card_template model_card_template.md --dataset_info dataset_info/tmvar3.md

```

#### GNormPlus

```bash
# Preprocess the data
python prepare_gnormplus.py --gnormplus_dir corpora_sources/GNormPlusCorpus --out_train datasets/gnormplus_train.bioc.xml.gz --out_val datasets/gnormplus_val.bioc.xml.gz --out_test datasets/gnormplus_test.bioc.xml.gz

# Tune the model and save it
python tune_ner.py --train_corpus datasets/gnormplus_train.bioc.xml.gz --val_corpus datasets/gnormplus_val.bioc.xml.gz --test_corpus datasets/gnormplus_test.bioc.xml.gz --n_trials 10 --model_name bioner_gnormplus --model_card_template model_card_template.md --dataset_info dataset_info/gnormplus.md
```

