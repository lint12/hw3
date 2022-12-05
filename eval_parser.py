# pip install conllu
# pip install datasets
# pip install spacy
# python -m spacy download en_core_web_sm

from argparse import ArgumentParser
from collections import defaultdict

from src.dependency_parse import DependencyParse
from src.parsers.bert_parser import BertParser
from src.parsers.spacy_parser import SpacyParser
from src.metrics import get_metrics

from datasets import load_dataset
import numpy as np 

def get_parses(subset: str, test: bool = False):
    """Return a list of dependency parses in language specified by `subset` from the universal_dependencies dataset.py

    You should use HuggingFaceDatasets to load the dataset.
    
    Return the test set of test == True; validation set otherwise.
    """
    # TODO: Your code here!
    # print(f'subset: {subset}')
    datasets = load_dataset('universal_dependencies', subset)
    # print('datasets: ', datasets)

    dependency_parse_list = []
    
    if test: 
        print('using test') 
        dataset = datasets['test']
    else: 
        print('using val')
        dataset = datasets['validation']
    

    for sentence in dataset:
        dependency_parse_list.append(DependencyParse.from_huggingface_dict(sentence))

    # return type: List[DependencyParse]
    return dependency_parse_list


def parse_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("method", choices=["spacy", "bert"])
    arg_parser.add_argument("--data_subset", type=str, default="zh_gsdsimp")
    arg_parser.add_argument("--test", action="store_true")
    
    # SpaCy parser arguments.
    arg_parser.add_argument("--model_name", type=str, default="zh_core_web_sm")

    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.method == "spacy":
        parser = SpacyParser(args.model_name)
    elif args.method == "bert":
        parser = BertParser(model_path='./bert-parser-0.75.pt', mst=True)
    else:
        raise ValueError("Unknown parser")

    cum_metrics = defaultdict(list)
    sent_cnt = 1
    for gold in get_parses(args.data_subset, test=args.test):
        print(f'====On sentence {sent_cnt}====')
        pred = parser.parse(gold.text, gold.tokens)
        for metric, value in get_metrics(pred, gold).items():
            cum_metrics[metric].append(value)
        sent_cnt += 1

        # if sent_cnt > 2: 
        #     break
    
    print({metric: np.mean(data) for metric, data in cum_metrics.items()})

