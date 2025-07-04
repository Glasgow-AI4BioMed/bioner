from transformers import pipeline
import json
import os

import pandas as pd
from tabulate import tabulate

def make_spans(doc_idx, passage):
    spans = []
    for anno in passage.annotations:
        loc = anno.total_span
        start,end = loc.offset-passage.offset, loc.offset+loc.length-passage.offset
        spans.append( (doc_idx,start,end,anno.infons['label']) )
    return spans

from tqdm.auto import tqdm

def evaluate_at_span_level(ner_pipeline, collection):
    labels = sorted(set( anno.infons['label'] for doc in collection.documents for passage in doc.passages for anno in passage.annotations ) )

    gold_spans, predicted_spans = [], []
    for doc_idx,doc in enumerate(tqdm(collection.documents)):
        for passage in doc.passages:
            gold_spans += make_spans(doc_idx, passage)
    
            output = ner_pipeline(passage.text)
            predicted_spans += [ (doc_idx, x['start'],x['end'],x['entity_group']) for x in output ]

    report_dict = {}
    for label in labels:
        gold_for_label = set( (d,s,e,l) for d,s,e,l in gold_spans if l == label )
        pred_for_label = set( (d,s,e,l) for d,s,e,l in predicted_spans if l == label )
                
        TP = len(gold_for_label.intersection(pred_for_label))
        FP = len(pred_for_label.difference(gold_for_label))
        FN = len(gold_for_label.difference(pred_for_label))

        precision = TP / (TP+FP) if (TP+FP) > 0 else 0
        recall = TP / (TP+FN) if (TP+FN) > 0 else 0
        f1 = 2 * (precision*recall) / (precision+recall)  if (precision+recall) > 0 else 0

        report_dict[label] = {'precision':precision, 'recall':recall, 'f1-score':f1, 'support':len(gold_for_label)}
        
    total_support = sum( report_dict[label]['support'] for label in labels )
    
    report_dict['macro avg'] = { 
        'precision': sum( report_dict[label]['precision'] for label in labels ) / len(labels) ,
        'recall': sum( report_dict[label]['recall'] for label in labels ) / len(labels) ,
        'f1-score': sum( report_dict[label]['f1-score'] for label in labels ) / len(labels) ,
        'support': total_support
    }
    
    report_dict['weighted avg'] = { 
        'precision': sum( report_dict[label]['support'] * report_dict[label]['precision'] for label in labels ) / total_support ,
        'recall': sum( report_dict[label]['support'] * report_dict[label]['recall'] for label in labels ) / total_support ,
        'f1-score': sum( report_dict[label]['support'] * report_dict[label]['f1-score'] for label in labels ) / total_support ,
        'support': total_support
    }

    return report_dict

def classification_report_to_markdown(report_dict):
    headers = ["Label", "Precision", "Recall", "F1-score", "Support"]
    rows = []

    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            row = [
                label,
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1-score']:.3f}",
                f"{metrics['support']:.0f}"
            ]
            rows.append(row)

    # Markdown table format
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows:
        table += "| " + " | ".join(row) + " |\n"

    return table


def make_hyperparameter_table(hyperparameters):
    headers = ["Hyperparameter", "Value"]
    table = f"| {headers[0]} | {headers[1]} |\n"
    table += f"|{'-' * (len(headers[0]) + 2)}|{'-' * (len(headers[1]) + 2)}|\n"
    
    for key, value in hyperparameters.items():
        table += f"| {key} | {value} |\n"
    
    return table

def make_example_output(ner_pipeline):
    # Apply it to some text
    result = ner_pipeline("EGFR T790M mutations affect treatment outcomes for NSCLC patients receiving erlotinib.")

    # Create a compact and commented version of the output
    for x in result:
        x['score'] = round(float(x['score']),5)
    jsoned = [ json.dumps(x) for x in result ]
    commented = [ f'# [ {line},' if i==0 else (f'#   {line} ]' if i==(len(jsoned)-1) else f'#   {line},') for i,line in enumerate(jsoned) ]

    return "\n".join(commented)

def prepare_model_repo(model_name, base_model, annotated_labels, n_trials, hyperparameters, train_collection, val_collection, test_collection, train_token_report, val_token_report, test_token_report, model_card_template_filename, dataset_info_filename, word_based):
    
    ner_pipeline = pipeline("token-classification", 
                            model=model_name,
                            aggregation_strategy="first" if word_based else "simple", 
                            device='cuda')

    with open(model_card_template_filename) as f:
        model_card_template = f.read()
        
    with open(dataset_info_filename) as f:
        dataset_info = f.read().strip()

    label_count = len(annotated_labels)

    if label_count == 1:
        label_explanation = f"It predicts spans with only 1 possible label ({annotated_labels[0]})."
    else:
        nice_labels = ", ".join(annotated_labels[:-1]) + f" and {annotated_labels[-1]}"
        label_explanation = f"It predicts spans with {label_count} possible labels. The labels are **{nice_labels}**."

    # Recover the number of epochs from training
    with open(f'{model_name}/epoch.txt') as f:
        epochs = int(f.read().strip())
    os.remove(f'{model_name}/epoch.txt')

    # Remove the training_args file as it may not contain args for the best run
    os.remove(f'{model_name}/training_args.bin')

    example_output = make_example_output(ner_pipeline)

    hyperparameters = { 'epochs':epochs, **hyperparameters }
    hyperparameter_table = make_hyperparameter_table(hyperparameters)
    
    train_span_report = evaluate_at_span_level(ner_pipeline, train_collection)
    val_span_report = evaluate_at_span_level(ner_pipeline, val_collection)
    test_span_report = evaluate_at_span_level(ner_pipeline, test_collection)

    readme = model_card_template.format(
        model_name=model_name,
        base_model=base_model,
        label_explanation=label_explanation,
        example_output=example_output,
        dataset_info=dataset_info,
        n_trials=n_trials,
        hyperparameter_table=hyperparameter_table,
        test_span_report=classification_report_to_markdown(test_span_report),
    )

    with open(f"{model_name}/best_hyperparameters.json", "w") as f:
        json.dump(hyperparameters, f, indent=2)

    with open(f"{model_name}/performance_report.md", "w") as f:
        f.write(f'# Performance on Training Set\n\n## Span Level\n\n{classification_report_to_markdown(train_span_report)}\n## Token Level\n\n{classification_report_to_markdown(train_token_report)}\n\n')
        f.write(f'# Performance on Validation Set\n\n## Span Level\n\n{classification_report_to_markdown(val_span_report)}\n## Token Level\n\n{classification_report_to_markdown(val_token_report)}\n\n')
        f.write(f'# Performance on Testing Set\n\n## Span Level\n\n{classification_report_to_markdown(test_span_report)}\n## Token Level\n\n{classification_report_to_markdown(test_token_report)}\n\n')
        
    with open(f"{model_name}/performance_report.json", "w") as f:
        combined_performance = {
            'train': {'token_level':train_token_report, 'span_level':train_span_report},
            'val': {'token_level':val_token_report, 'span_level':val_span_report},
            'test': {'token_level':test_token_report, 'span_level':test_span_report},
        }
        json.dump(combined_performance,f,indent=2)
        
    with open(f"{model_name}/README.md", "w") as f:
        f.write(readme)

    print("="*80)
    print("TEST SPAN REPORT:\n")
    test_span_report_df = pd.DataFrame(test_span_report)
    print(tabulate(test_span_report_df.T, headers='keys'))
    print("="*80)
