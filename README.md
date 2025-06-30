# bioner
A selection of biomedical NER models

```
bash fetch_corpora.sh
```

```
python prepare_medmentions.py --medmentions_dir corpora_sources/medmentions/st21pv --umls_dir ~/umls/2017AA-full/META/ --out_train medmentions_st21pv_train.xml.gz --out_val medmentions_st21pv_val.bioc.xml.gz --out_test medmentions_st21pv_test.bioc.xml.gz
```

```
python tune_ner.py --train_corpus medmentions_st21pv_train.bioc.xml.gz --val_corpus medmentions_st21pv_val.bioc.xml.gz --test_corpus medmentions_st21pv_test.bioc.xml.gz --n_
trials 1 --output_dir bioner_medmentions_st21pv
```
