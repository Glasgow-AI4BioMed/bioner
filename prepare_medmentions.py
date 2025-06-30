import bioc
from bioc import biocxml, pubtator
import argparse
import gzip

def pubtator_to_bioc(doc):
	bioc_doc = bioc.BioCDocument()
	bioc_doc.id = doc.pmid
	bioc_passage = bioc.BioCPassage()
	bioc_passage.text = doc.text
	bioc_passage.offset = 0
	bioc_doc.add_passage(bioc_passage)

	title = doc.text.split('\n')[0]
	bioc_doc.infons['title'] = title

	for a in doc.annotations:
		bioc_anno = bioc.BioCAnnotation()
		bioc_anno.infons['concept_id'] = a.id
		bioc_anno.text = a.text
		bioc_loc = bioc.BioCLocation(a.start,a.end-a.start)
		bioc_anno.add_location(bioc_loc)
		bioc_passage.add_annotation(bioc_anno)

	return bioc_doc

def save_bioc_docs(docs, filename):
	collection = bioc.BioCCollection.of_documents(*docs)
	with gzip.open(filename, 'wt', encoding='utf8') as f:
		biocxml.dump(collection, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--medmentions_dir',type=str,required=True,help='Directory with MedMentions files')
    parser.add_argument('--umls_dir',type=str,required=True,help='Directory with UMLS files')
    parser.add_argument('--out_train',type=str,required=True,help='Output Gzipped BioC XML file for training set')
    parser.add_argument('--out_val',type=str,required=True,help='Output Gzipped BioC XML file for validation set')
    parser.add_argument('--out_test',type=str,required=True,help='Output Gzipped BioC XML file for test set')
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
    with open(f'{args.umls_dir}/MRSTY.RRF') as f:
        for line in f:
            split = line.strip().split('|')
            cui = split[0]
            type_name = split[3]
    
            semantic_types[cui] = type_name

    print("Labelling annotations with semantic types...")
    for doc in train_docs + val_docs + test_docs:
        for passage in doc.passages:
            for anno in passage.annotations:
                cui = anno.infons['concept_id'].removeprefix('UMLS:')
                type_name = semantic_types.get(cui)
                anno.infons['semantic_type'] = type_name
            passage.annotations = [ anno for anno in passage.annotations if anno.infons['semantic_type'] ]

    print("Saving...")
    save_bioc_docs(train_docs, args.out_train)
    save_bioc_docs(val_docs, args.out_val)
    save_bioc_docs(test_docs, args.out_test)

    print("Done.")

if __name__ == '__main__':
    main()
