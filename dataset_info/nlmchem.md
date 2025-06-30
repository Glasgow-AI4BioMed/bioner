**Source:** The NLM-Chem dataset was downloaded from: https://ftp.ncbi.nlm.nih.gov/pub/lu/NLMChem/

The dataset should be cited with: Islamaj, Rezarta, et al. "NLM-Chem, a new resource for chemical entity recognition in PubMed full text literature." Scientific data 8.1 (2021): 91. DOI: [10.1038/s41597-021-00875-16](https://doi.org/10.1038/s41597-021-00875-1)

**Preprocessing:** The training/validation/test split was maintained from the original dataset. The annotations were filtered down to only 'Chemical'. The preprocessing script for this dataset is [prepare_nlmchem.py](https://github.com/Glasgow-AI4BioMed/bioner/blob/main/prepare_nlmchem.py).
