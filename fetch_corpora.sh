#!/bin/bash
set -ex

mkdir -p corpora_sources
cd corpora_sources

# MedMentions (full)
mkdir -p medmentions/full
wget -O medmentions/full/corpus_pubtator.txt.gz https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator.txt.gz
wget -O medmentions/full/corpus_pubtator_pmids_dev.txt https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator_pmids_dev.txt
wget -O medmentions/full/corpus_pubtator_pmids_test.txt https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator_pmids_test.txt
wget -O medmentions/full/corpus_pubtator_pmids_trng.txt https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator_pmids_trng.txt

# MedMentions (st21pv)
mkdir -p medmentions/st21pv
wget -O medmentions/st21pv/corpus_pubtator.txt.gz https://github.com/chanzuckerberg/MedMentions/raw/refs/heads/master/st21pv/data/corpus_pubtator.txt.gz
wget -O medmentions/st21pv/corpus_pubtator_pmids_dev.txt https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator_pmids_dev.txt
wget -O medmentions/st21pv/corpus_pubtator_pmids_test.txt https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator_pmids_test.txt
wget -O medmentions/st21pv/corpus_pubtator_pmids_trng.txt https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator_pmids_trng.txt

wget -O medmentions/SemGroups.txt https://www.nlm.nih.gov/research/umls/knowledge_sources/semantic_network/SemGroups.txt

# GNormPlus
rm -fr GNormPlusCorpus
wget -O GNormPlusCorpus.zip https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/download/GNormPlus/GNormPlusCorpus.zip
unzip -o GNormPlusCorpus.zip
rm GNormPlusCorpus.zip

# NLM-Chem
rm -fr NLM-Chem
wget -O NLM-Chem-corpus.zip https://ftp.ncbi.nlm.nih.gov/pub/lu/NLMChem/NLM-Chem-corpus.zip
unzip -o NLM-Chem-corpus.zip
mv FINAL_v1 NLM-Chem
rm NLM-Chem-corpus.zip

# tmVar
wget -O tmVar3Corpus.txt https://ftp.ncbi.nlm.nih.gov/pub/lu/tmVar3/tmVar3Corpus.txt

# BC5CDR
rm -fr CDR_Data
wget -O CDR_Data.zip https://ftp.ncbi.nlm.nih.gov/pub/lu/BC5CDR/CDR_Data.zip
unzip -o CDR_Data.zip
rm CDR_Data.zip

# NCBI disease
rm -fr NCBI-disease
wget -O NCBItrainset_corpus.zip https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItrainset_corpus.zip
wget -O NCBIdevelopset_corpus.zip https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBIdevelopset_corpus.zip
wget -O NCBItestset_corpus.zip https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItestset_corpus.zip
mkdir -p NCBI-disease
unzip -d NCBI-disease -o NCBItrainset_corpus.zip
unzip -d NCBI-disease -o NCBIdevelopset_corpus.zip
unzip -d NCBI-disease -o NCBItestset_corpus.zip
rm NCBItrainset_corpus.zip NCBIdevelopset_corpus.zip NCBItestset_corpus.zip

rm -fr __MACOSX
