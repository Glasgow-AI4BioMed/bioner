from bioc import biocxml
import gzip
import argparse
import os
import json
import xml.etree.ElementTree as ET

from entitytools.file_formats import save_bioc_docs

def main():
    parser = argparse.ArgumentParser(description='Convert BC5CDR corpus to BioCXML and set up a matching MeSH ontology')
    parser.add_argument('--bc5cdr_dir',required=True,type=str,help='Directory with source NCBI Disease corpus files')
    parser.add_argument('--out_train',required=True,type=str,help='Output Gzipped BioC XML for training data')
    parser.add_argument('--out_val',required=True,type=str,help='Output Gzipped BioC XML for validation data')
    parser.add_argument('--out_test',required=True,type=str,help='Output Gzipped BioC XML for test data')
    args = parser.parse_args()

    assert os.path.isdir(args.bc5cdr_dir)

    print("Loading documents...")
    with open(f'{args.bc5cdr_dir}/CDR_TrainingSet.BioC.xml') as f:
        train_collection = biocxml.load(f)
    with open(f'{args.bc5cdr_dir}/CDR_DevelopmentSet.BioC.xml') as f:
        val_collection = biocxml.load(f)
    with open(f'{args.bc5cdr_dir}/CDR_TestSet.BioC.xml') as f:
        test_collection = biocxml.load(f)

    print("Reformatting BioC XML annotations...")
    for doc in train_collection.documents + val_collection.documents + test_collection.documents:
        for passage in doc.passages:
            #passage.annotations = [ anno for anno in passage.annotations if anno.infons['MESH'] != '-1' and '|' not in anno.infons['MESH'] ]
            for anno in passage.annotations:
                anno.infons = { 'label': f"MESH:{anno.infons['type']}" }

    print("Removing non-contiguous annotations...")
    for doc in train_collection.documents + val_collection.documents + test_collection.documents:
        for passage in doc.passages:
            passage.annotations = [ anno for anno in passage.annotations if len(anno.locations) == 1 ]
        
    print("Saving documents...")    
    with gzip.open(args.out_train, 'wt', encoding='utf8') as f:
        biocxml.dump(train_collection, f)
    with gzip.open(args.out_val, 'wt', encoding='utf8') as f:
        biocxml.dump(val_collection, f)
    with gzip.open(args.out_test, 'wt', encoding='utf8') as f:
        biocxml.dump(test_collection, f)

    print(f"{len(train_collection.documents)=} {len(val_collection.documents)=} {len(test_collection.documents)=}")
    print("Done")
	
if __name__ == '__main__':
	main()


