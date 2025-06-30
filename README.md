# bioner
A selection of biomedical NER models

```
bash fetch_corpora.sh
```

```
mkdir datasets

python prepare_medmentions.py --medmentions_dir corpora_sources/medmentions/st21pv --semantic_groups corpora_sources/medmentions/SemGroups.txt --out_train datasets/medmentions_st21pv_train.bioc.xml.gz --out_val datasets/medmentions_st21pv_val.bioc.xml.gz --out_test datasets/medmentions_st21pv_test.bioc.xml.gz

python prepare_medmentions.py --medmentions_dir corpora_sources/medmentions/st21pv --semantic_groups corpora_sources/medmentions/SemGroups.txt --out_train datasets/medmentions_st21pv_finegrain_train.xml.gz --out_val datasets/medmentions_st21pv_finegrain_val.bioc.xml.gz --out_test datasets/medmentions_st21pv_finegrain_test.bioc.xml.gz --finegrain

python prepare_ncbi_disease.py --ncbidisease_dir corpora_sources/NCBI-disease --out_train datasets/ncbi_disease_train.bioc.xml.gz --out_val datasets/ncbi_disease_val.bioc.xml.gz --out_test datasets/ncbi_disease_test.bioc.xml.gz

python prepare_nlmchem.py --nlmchem_dir corpora_sources/NLM-Chem --out_train datasets/nlmchem_train.bioc.xml.gz --out_val datasets/nlmchem_val.bioc.xml.gz --out_test datasets/nlmchem_test.bioc.xml.gz

python prepare_bc5cdr.py --bc5cdr_dir corpora_sources/CDR_Data/CDR.Corpus.v010516 --out_train datasets/bc5cdr_train.bioc.xml.gz --out_val datasets/bc5cdr_val.bioc.xml.gz --out_test datasets/bc5cdr_test.bioc.xml.gz

python prepare_tmvar.py --tmvar_corpus corpora_sources/tmVar3Corpus.txt --out_train datasets/tmvar_train.bioc.xml.gz --out_val datasets/tmvar_val.bioc.xml.gz --out_test datasets/tmvar_test.bioc.xml.gz

python prepare_gnormplus.py --gnormplus_dir corpora_sources/GNormPlusCorpus --out_train datasets/gnormplus_train.bioc.xml.gz --out_val datasets/gnormplus_val.bioc.xml.gz --out_test datasets/gnormplus_test.bioc.xml.gz
```



```
python tune_ner.py --train_corpus datasets/medmentions_st21pv_train.bioc.xml.gz --val_corpus datasets/medmentions_st21pv_val.bioc.xml.gz --test_corpus datasets/medmentions_st21pv_test.bioc.xml.gz --n_trials 10 --output_dir bioner_medmentions_st21pv

python tune_ner.py --train_corpus datasets/medmentions_st21pv_finegrain_train.bioc.xml.gz --val_corpus datasets/medmentions_st21pv_finegrain_val.bioc.xml.gz --test_corpus datasets/medmentions_st21pv_finegrain_test.bioc.xml.gz --n_trials 10 --output_dir bioner_medmentions_st21pv_finegrain

python tune_ner.py --train_corpus datasets/ncbi_disease_train.bioc.xml.gz --val_corpus datasets/ncbi_disease_val.bioc.xml.gz --test_corpus datasets/ncbi_disease_test.bioc.xml.gz --n_trials 10 --output_dir bioner_ncbi_disease

python tune_ner.py --train_corpus datasets/nlmchem_train.bioc.xml.gz --val_corpus datasets/nlmchem_val.bioc.xml.gz --test_corpus datasets/nlmchem_test.bioc.xml.gz --n_trials 10 --output_dir bioner_nlmchem

python tune_ner.py --train_corpus datasets/bc5cdr_train.bioc.xml.gz --val_corpus datasets/bc5cdr_val.bioc.xml.gz --test_corpus datasets/bc5cdr_test.bioc.xml.gz --n_trials 10 --output_dir bioner_bc5cdr

python tune_ner.py --train_corpus datasets/tmvar_train.bioc.xml.gz --val_corpus datasets/tmvar_val.bioc.xml.gz --test_corpus datasets/tmvar_test.bioc.xml.gz --n_trials 10 --output_dir bioner_tmvar

python tune_ner.py --train_corpus datasets/gnormplus_train.bioc.xml.gz --val_corpus datasets/gnormplus_val.bioc.xml.gz --test_corpus datasets/gnormplus_test.bioc.xml.gz --n_trials 10 --output_dir bioner_gnormplus
```

