# bioner
A selection of biomedical NER models

```
bash fetch_corpora.sh
```

```
python prepare_medmentions.py --medmentions_dir corpora_sources/medmentions/st21pv --semantic_groups corpora_sources/medmentions/SemGroups.txt --out_train medmentions_st21pv_train.bioc.xml.gz --out_val medmentions_st21pv_val.bioc.xml.gz --out_test medmentions_st21pv_test.bioc.xml.gz

python prepare_medmentions.py --medmentions_dir corpora_sources/medmentions/st21pv --semantic_groups corpora_sources/medmentions/SemGroups.txt --out_train medmentions_st21pv_finegrain_train.xml.gz --out_val medmentions_st21pv_finegrain_val.bioc.xml.gz --out_test medmentions_st21pv_finegrain_test.bioc.xml.gz --finegrained

python prepare_ncbi_disease.py --ncbidisease_dir corpora_sources/NCBI-disease --out_train ncbi_disease_train.bioc.xml.gz --out_val ncbi_disease_val.bioc.xml.gz --out_test ncbi_disease_test.bioc.xml.gz

python prepare_nlmchem.py --nlmchem_dir corpora_sources/NLM-Chem --out_train nlmchem_train.bioc.xml.gz --out_val nlmchem_val.bioc.xml.gz --out_test nlmchem_test.bioc.xml.gz

python prepare_bc5cdr.py --bc5cdr_dir corpora_sources/CDR_Data/CDR.Corpus.v010516 --out_train bc5cdr_train.bioc.xml.gz --out_val bc5cdr_val.bioc.xml.gz --out_test bc5cdr_test.bioc.xml.gz

python prepare_tmvar.py --tmvar_corpus corpora_sources/tmVar3Corpus.txt --out_train tmvar_train.bioc.xml.gz --out_val tmvar_val.bioc.xml.gz --out_test tmvar_test.bioc.xml.gz

python prepare_gnormplus.py --gnormplus_dir corpora_sources/GNormPlusCorpus --out_train gnormplus_train.bioc.xml.gz --out_val gnormplus_val.bioc.xml.gz --out_test gnormplus_test.bioc.xml.gz
```



```
python tune_ner.py --train_corpus medmentions_st21pv_train.bioc.xml.gz --val_corpus medmentions_st21pv_val.bioc.xml.gz --test_corpus medmentions_st21pv_test.bioc.xml.gz --n_trials 10 --output_dir bioner_medmentions_st21pv

python tune_ner.py --train_corpus medmentions_st21pv_finegrain_train.bioc.xml.gz --val_corpus medmentions_st21pv_finegrain_val.bioc.xml.gz --test_corpus medmentions_st21pv_finegrain_test.bioc.xml.gz --n_trials 10 --output_dir bioner_medmentions_st21pv_finegrain

python tune_ner.py --train_corpus ncbi_disease_train.bioc.xml.gz --val_corpus ncbi_disease_val.bioc.xml.gz --test_corpus ncbi_disease_test.bioc.xml.gz --n_trials 10 --output_dir bioner_ncbi_disease

python tune_ner.py --train_corpus nlmchem_train.bioc.xml.gz --val_corpus nlmchem_val.bioc.xml.gz --test_corpus nlmchem_test.bioc.xml.gz --n_trials 10 --output_dir bioner_nlmchem

python tune_ner.py --train_corpus bc5cdr_train.bioc.xml.gz --val_corpus bc5cdr_val.bioc.xml.gz --test_corpus bc5cdr_test.bioc.xml.gz --n_trials 10 --output_dir bioner_bc5cdr

python tune_ner.py --train_corpus tmvar_train.bioc.xml.gz --val_corpus tmvar_val.bioc.xml.gz --test_corpus tmvar_test.bioc.xml.gz --n_trials 10 --output_dir bioner_tmvar

python tune_ner.py --train_corpus gnormplus_train.bioc.xml.gz --val_corpus gnormplus_val.bioc.xml.gz --test_corpus gnormplus_test.bioc.xml.gz --n_trials 10 --output_dir bioner_gnormplus
```

