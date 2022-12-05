"""
TODO: Your code below!

This file should implement all steps described in Part 2, and can be structured however you want.

Rather than using normal BERT, you should use distilbert-base-uncased. This will train faster.

We recommend training on a GPU, either by using HPC or running the command line commands on Colab.

Hints:
    * It will probably be helpful to save intermediate outputs (preprocessed data).
    * To save your finetuned models, you can use torch.save().
"""

from src.dependency_parse import DependencyParse
from datasets import load_dataset
import csv     
import pickle5 as pickle

def get_parses(split):
    datasets = load_dataset('universal_dependencies', 'en_gum')
    dataset = datasets[split]

    dependency_parse_list = []

    for sentence in dataset:
        rel_pos_list = []
        heads = sentence['head']
        deprels = sentence['deprel']

        for idx in range(len(sentence['tokens'])): 
            if deprels[idx] == 'root': 
                head_i = idx
            else: 
                head_i = int(heads[idx]) - 1
            rel_pos_list.append(head_i - idx) 

        data_dict = {
            'text': sentence['text'], 
            'tokens': sentence['tokens'], 
            'rel_pos': rel_pos_list, 
            'dep_label': sentence['deprel']
        }

        dependency_parse_list.append(data_dict)

    return dependency_parse_list

def add_unk(pp_dataset, rel_pos_set, dep_label_set): 
    unk_dataset = []

    for sentence in pp_dataset: 
        rel_list = []
        deprel_list = []
        for rel_pos in sentence['rel_pos']: 
            if rel_pos not in rel_pos_set: 
                rel_list.append('unk')
            else:
                rel_list.append(rel_pos)

        for deprel in sentence['dep_label']: 
            if deprel not in dep_label_set: 
                deprel_list.append('unk')
            else:
                deprel_list.append(deprel)

        data_dict = {
            'text': sentence['text'], 
            'tokens': sentence['tokens'], 
            'rel_pos': rel_list, 
            'dep_label': deprel_list
        }
        unk_dataset.append(data_dict)

    return unk_dataset


if __name__ == "__main__":
    train_pp = get_parses('train')

    rel_pos_set = set()
    dep_label_set = set()

    for sentence in train_pp: 
        for rel_pos in sentence['rel_pos']: 
            rel_pos_set.add(rel_pos)

        for deprel in sentence['dep_label']:
            dep_label_set.add(deprel)

    vocab = {}
    vocab['rel_pos_set'] = rel_pos_set
    vocab['dep_label_set'] = dep_label_set

    with open('vocab.pickle', 'wb') as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    val_pp = get_parses('validation')
    test_pp = get_parses('test')

    val_pp = add_unk(val_pp, rel_pos_set, dep_label_set)
    test_pp = add_unk(test_pp, rel_pos_set, dep_label_set)   

    with open('val_pp.pickle', 'wb') as handle:
        pickle.dump(val_pp, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('test_pp.pickle', 'wb') as handle:
        pickle.dump(test_pp, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
    with open('en_gum_10.tsv', 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        for sentence in train_pp[:10]: 
            tsv_output.writerow([sentence['text'], sentence['rel_pos'], sentence['dep_label']])


