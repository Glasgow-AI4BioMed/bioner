from bioc import biocxml
import argparse
import os

from sklearn.model_selection import train_test_split
from utils import pubtator_to_bioc, save_bioc_docs

def main():
    parser = argparse.ArgumentParser(description='Convert GNormPlus to BioCXML')
    parser.add_argument('--gnormplus_dir',required=True,type=str,help='Directory with source GNormPlus corpus files')
    parser.add_argument('--out_train',required=True,type=str,help='Output Gzipped BioC XML for training data')
    parser.add_argument('--out_val',required=True,type=str,help='Output Gzipped BioC XML for validation data')
    parser.add_argument('--out_test',required=True,type=str,help='Output Gzipped BioC XML for test data')
    args = parser.parse_args()

    assert os.path.isdir(args.gnormplus_dir)

    print("Loading documents...")
    with open(f"{args.gnormplus_dir}/BC2GNtrain.BioC.xml") as fp:
        train_collection = biocxml.load(fp)
    with open(f"{args.gnormplus_dir}/BC2GNtest.BioC.xml") as fp:
        test_collection = biocxml.load(fp)

    train_docs = train_collection.documents
    test_docs = test_collection.documents

    train_docs, val_docs = train_test_split(train_docs, train_size=0.75, random_state=42)
       
    print("Saving documents...")
    save_bioc_docs(train_docs, args.out_train)
    save_bioc_docs(val_docs, args.out_val)
    save_bioc_docs(test_docs, args.out_test)

    print(f"{len(train_docs)=} {len(val_docs)=} {len(test_docs)=}")
    print("Done")
	
if __name__ == '__main__':
	main()
