from bioc import biocxml
import argparse
import os

from utils import save_bioc_docs

def main():
    parser = argparse.ArgumentParser(description='Convert NLM-Chem corpus to BioCXML')
    parser.add_argument('--nlmchem_dir',required=True,type=str,help='Directory with source NLM-Chem corpus files')
    parser.add_argument('--out_train',required=True,type=str,help='Output Gzipped BioC XML for training data')
    parser.add_argument('--out_val',required=True,type=str,help='Output Gzipped BioC XML for validation data')
    parser.add_argument('--out_test',required=True,type=str,help='Output Gzipped BioC XML for test data')
    args = parser.parse_args()

    assert os.path.isdir(args.nlmchem_dir)

    print("Loading data split...")
    with open(f'{args.nlmchem_dir}/pmcids_train.txt') as f:
        train_pmcids = [ line.strip() for line in f ]
    with open(f'{args.nlmchem_dir}/pmcids_dev.txt') as f:
        val_pmcids = [ line.strip() for line in f ]
    with open(f'{args.nlmchem_dir}/pmcids_test.txt') as f:
        test_pmcids = [ line.strip() for line in f ]

    print("Loading documents...")
    train_docs, val_docs, test_docs = [], [], []
    for filename in os.listdir(f"{args.nlmchem_dir}/ALL"):
        with open(f"{args.nlmchem_dir}/ALL/{filename}") as fp:
            collection = biocxml.load(fp)
            for doc in collection.documents:
                if doc.id in train_pmcids:
                    train_docs.append(doc)
                elif doc.id in val_pmcids:
                    val_docs.append(doc)
                elif doc.id in test_pmcids:
                    test_docs.append(doc)
                else:
                    raise RuntimeError(f"{doc.id=} is not in one of the train/val/test groupings")

    print("Filtering to Chemical...")
    for doc in train_docs+val_docs+test_docs:
        for passage in doc.passages:
             passage.annotations = [ anno for anno in passage.annotations if anno.infons['type'] == 'Chemical']
             for anno in passage.annotations:
                 anno.infons = { 'label': anno.infons['type'] }
    
    print("Saving documents...")
    save_bioc_docs(train_docs, args.out_train)
    save_bioc_docs(val_docs, args.out_val)
    save_bioc_docs(test_docs, args.out_test)

    print(f"{len(train_docs)=} {len(val_docs)=} {len(test_docs)=}")
    print("Done")
	
if __name__ == '__main__':
	main()


