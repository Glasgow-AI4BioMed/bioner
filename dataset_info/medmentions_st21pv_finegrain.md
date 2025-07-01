**Source:** The ST21pv version of MedMentions was downloaded from: https://github.com/chanzuckerberg/MedMentions/tree/master/st21pv

The dataset should be cited with: Mohan, Sunil, and Donghui Li. "MedMentions: A Large Biomedical Corpus Annotated with UMLS Concepts." Automated Knowledge Base Construction (AKBC), 2019, https://openreview.net/forum?id=SylxCx5pTQ. DOI: [10.24432/C5G59C](https://doi.org/10.24432/C5G59C)

An overview of semantic types can be found at: https://www.nlm.nih.gov/research/umls/META3_current_semantic_types.html

**Preprocessing:** The training, validation and test splits were maintained from the original dataset. Annotations were mapped to specific *semantic types* using the Semantic Groups file available at: https://www.nlm.nih.gov/research/umls/knowledge_sources/semantic_network/index.html. This contrasts with the finegrained version that mapped annotations to *semantic groups*. The preprocessing script for this dataset is [prepare_medmentions.py](https://github.com/Glasgow-AI4BioMed/bioner/blob/main/prepare_medmentions.py.py) with the --finegrain flag.

