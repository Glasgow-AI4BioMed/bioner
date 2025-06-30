**Source:** The NCBI Disease dataset was downloaded from: https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/

The dataset should be cited with: Doğan, Rezarta Islamaj, Robert Leaman, and Zhiyong Lu. "NCBI disease corpus: a resource for disease name recognition and concept normalization." Journal of biomedical informatics 47 (2014): 1-10. DOI: [10.1016/j.jbi.2013.12.006](https://doi.org/10.1016/j.jbi.2013.12.006)

**Preprocessing:** The training/validation/test split was maintained from the original dataset. The annotations were filtered down to only 'DiseaseClass' and 'SpecificDisease'. The preprocessing script for this dataset is [prepare_ncbi_disease.py](https://github.com/Glasgow-AI4BioMed/bioner/blob/main/prepare_ncbi_disease.py).
