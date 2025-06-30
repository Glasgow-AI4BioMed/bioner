from bioc import pubtator
import gzip
import argparse
import csv
from tqdm.auto import tqdm
import os
import json

from utils import pubtator_to_bioc, save_bioc_docs

def main():
    parser = argparse.ArgumentParser(description='Convert NCBI Disease to BioCXML')
    parser.add_argument('--ncbidisease_dir',required=True,type=str,help='Directory with source NCBI Disease corpus files')
    parser.add_argument('--out_train',required=True,type=str,help='Output Gzipped BioC XML for training data')
    parser.add_argument('--out_val',required=True,type=str,help='Output Gzipped BioC XML for validation data')
    parser.add_argument('--out_test',required=True,type=str,help='Output Gzipped BioC XML for test data')
    args = parser.parse_args()

    assert os.path.isdir(args.ncbidisease_dir)

    print("Loading documents...")
    with open(f"{args.ncbidisease_dir}/NCBItrainset_corpus.txt") as fp:
        train_docs = pubtator.load(fp)
    with open(f"{args.ncbidisease_dir}/NCBIdevelopset_corpus.txt") as fp:
        val_docs = pubtator.load(fp)
    with open(f"{args.ncbidisease_dir}/NCBItestset_corpus.txt") as fp:
        test_docs = pubtator.load(fp)

    train_docs = [ pubtator_to_bioc(doc) for doc in train_docs ]
    val_docs = [ pubtator_to_bioc(doc) for doc in val_docs ]
    test_docs = [ pubtator_to_bioc(doc) for doc in test_docs ]

    filtered_labels = ['DiseaseClass', 'SpecificDisease']
    print(f"Filtering annotations to {filtered_labels}")
    for doc in train_docs+val_docs+test_docs:
        for passage in doc.passages:
            passage.annotations = [ anno for anno in passage.annotations if anno.infons['label'] in filtered_labels ]
       
    print("Saving documents...")
    save_bioc_docs(train_docs, args.out_train)
    save_bioc_docs(val_docs, args.out_val)
    save_bioc_docs(test_docs, args.out_test)

    print(f"{len(train_docs)=} {len(val_docs)=} {len(test_docs)=}")
    print("Done")
	
if __name__ == '__main__':
	main()
