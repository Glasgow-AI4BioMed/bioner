from bioc import pubtator
import argparse
from sklearn.model_selection import train_test_split

from utils import pubtator_to_bioc, save_bioc_docs

def main():
    parser = argparse.ArgumentParser(description='Convert tmVar corpus to BioCXML')
    parser.add_argument('--tmvar_corpus',required=True,type=str,help='Directory with source tmVar corpus files')
    parser.add_argument('--out_train',required=True,type=str,help='Output Gzipped BioC XML for training data')
    parser.add_argument('--out_val',required=True,type=str,help='Output Gzipped BioC XML for validation data')
    parser.add_argument('--out_test',required=True,type=str,help='Output Gzipped BioC XML for test data')
    args = parser.parse_args()
    
    with open(args.tmvar_corpus) as fp:
        pubtator_docs = pubtator.load(fp)

    docs = [ pubtator_to_bioc(pubtator_doc) for pubtator_doc in pubtator_docs ]

    train_docs, temp_docs = train_test_split(docs, train_size=0.6, random_state=42)
    val_docs, test_docs = train_test_split(temp_docs, train_size=0.5, random_state=42)
   
    print("Saving documents...")
    save_bioc_docs(train_docs, args.out_train)
    save_bioc_docs(val_docs, args.out_val)
    save_bioc_docs(test_docs, args.out_test)

    print(f"{len(train_docs)=} {len(val_docs)=} {len(test_docs)=}")
    print("Done")
	
if __name__ == '__main__':
	main()


