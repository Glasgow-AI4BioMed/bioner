import bioc
from bioc import biocxml, pubtator
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
		bioc_anno.infons['label'] = a.type
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
