import bioc
from bioc import biocxml, pubtator
import argparse
import gzip

from utils import pubtator_to_bioc, save_bioc_docs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--medmentions_dir',type=str,required=True,help='Directory with MedMentions files')
    parser.add_argument('--semantic_groups',type=str,required=True,help='Semantic groups file')
    parser.add_argument('--out_train',type=str,required=True,help='Output Gzipped BioC XML file for training set')
    parser.add_argument('--out_val',type=str,required=True,help='Output Gzipped BioC XML file for validation set')
    parser.add_argument('--out_test',type=str,required=True,help='Output Gzipped BioC XML file for test set')
    parser.add_argument('--finegrain',action='store_true',help='Use the semantic types instead of more general groups')
    args = parser.parse_args()

    print("Loading MedMentions...")
    with open(f"{args.medmentions_dir}/corpus_pubtator_pmids_trng.txt") as f:
        train_pmids = set( line.strip() for line in f )
    with open(f"{args.medmentions_dir}/corpus_pubtator_pmids_dev.txt") as f:
        val_pmids = set( line.strip() for line in f )
    with open(f"{args.medmentions_dir}/corpus_pubtator_pmids_test.txt") as f:
        test_pmids = set( line.strip() for line in f )
        
    print(f"{len(train_pmids)=} {len(val_pmids)=} {len(test_pmids)=}")

    train_docs, val_docs, test_docs = [], [], []
    with gzip.open(f"{args.medmentions_dir}/corpus_pubtator.txt.gz", 'rt',encoding='utf8') as fp:
        pubtator_docs = pubtator.load(fp)

    print("Distributing documents to train/val/test splits...")
    for pubtator_doc in pubtator_docs:
        bioc_doc = pubtator_to_bioc(pubtator_doc)
        
        if bioc_doc.id in train_pmids:
            train_docs.append(bioc_doc)
        elif bioc_doc.id in val_pmids:
            val_docs.append(bioc_doc)
        elif bioc_doc.id in test_pmids:
            test_docs.append(bioc_doc)
        else:
            raise RuntimeError(f"ID not assigned to a split. {bioc_doc.id=}")

    print("Loading semantic types from UMLS...")
    semantic_types = {}
    with open(args.semantic_groups) as f:
        for line in f:
            group_id,group_name,type_id,type_name = line.strip('\n').split('|')

            if args.finegrain:
                semantic_types[type_id] = type_name
            else:
                semantic_types[type_id] = group_name
            
    print("Labelling annotations with semantic information...")
    for doc in train_docs + val_docs + test_docs:
        for passage in doc.passages:
            for anno in passage.annotations:
                type_name = semantic_types[anno.infons['label']]
                anno.infons['label'] = type_name
            passage.annotations = [ anno for anno in passage.annotations if anno.infons['label'] ]

    print("Saving...")
    save_bioc_docs(train_docs, args.out_train)
    save_bioc_docs(val_docs, args.out_val)
    save_bioc_docs(test_docs, args.out_test)

    print("Done.")

if __name__ == '__main__':
    main()
