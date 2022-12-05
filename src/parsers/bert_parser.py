from src.parsers.parser import Parser
from src.dependency_parse import DependencyParse
# from src.bert_parser_model import BertParserModel
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
from transformers import DistilBertTokenizer, DistilBertModel

import networkx as nx
from networkx.algorithms.tree.branchings import maximum_spanning_arborescence

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

class distilBERT_FT(nn.Module):

    def __init__(self, pre_model, hidden_dim, n_rel_classes, n_dep_classes):
      super().__init__()
      self.pre_model = pre_model
      self.rel_project = nn.Linear(hidden_dim, n_rel_classes)
      self.dep_project = nn.Linear(hidden_dim, n_dep_classes)

    def forward(self, input):
      hidden_states = self.pre_model(**input).last_hidden_state
      #print(f'hidden_states : {hidden_states.size()}')

      rel_out = self.rel_project(hidden_states)
      #print(f'rel_out size: {rel_out.size()}')
      #print(f'rel_out: {rel_out}')

      dep_out = self.dep_project(hidden_states)
      #print(f'dep_out size: {dep_out.size()}')
      #print(f'dep_out: {dep_out}')

      return rel_out, dep_out



class BertParser(Parser):

    """Represents a full parser that decodes parse trees from a finetuned BERT model."""

    def __init__(self, model_path: str, mst: bool = False):
        """Load your saved finetuned model using torch.load().

        Arguments:
            model_path: Path from which to load saved model.
            mst: Whether to use MST decoding or argmax decoding.
        """
        self.mst = mst
        # TODO: Load your neural net. 
        self.trained_model = torch.load(model_path)


    def filter_scores_and_preds(self, output, ud_token_len, bert_tokens, n_classes, idx_mapping): 
        # print('output: ', output)
        # print('ud_token_len: ', ud_token_len)
        # print('bert_tokens len: ', len(bert_tokens))
        # print('bert_tokens: ', bert_tokens)

        # tokens here are UD tokens
        filtered_scores = torch.zeros((1, ud_token_len, n_classes))

        hyphen_prev = False 
        filtered_idx = []

        for t_idx, token in enumerate(bert_tokens):
            if t_idx == 0 or t_idx == len(bert_tokens) - 1: 
                continue
            elif token.startswith('##'): 
                continue 
            elif token == '-': 
                hyphen_prev = True 
                continue
            elif hyphen_prev == True: 
                hyphen_prev = False 
                continue 
            else: 
                filtered_idx.append(t_idx)

        if len(filtered_idx) > ud_token_len: 
            filtered_idx = filtered_idx[:ud_token_len]

        #print('filtered_idx: ', filtered_idx)
        #print('num of idx in filtered_idx: ', len(filtered_idx))

        fs_idx = 0 
        for good_idx in filtered_idx:
            filtered_scores[0][fs_idx] = output[0][good_idx]
            fs_idx += 1
        
        #print('filtered_scores size: ', filtered_scores.size())
        #print('filtered_scores: ', filtered_scores)

        class_to_label = dict((v,k) for k,v in idx_mapping.items())

        preds = []
        for token_scores in filtered_scores[0]: 
            pred = torch.argmax(token_scores).item() # class idx
            preds.append(class_to_label[pred]) # label

        #print('preds: ', preds)
        return preds, filtered_scores

    def parse(self, sentence: str, tokens: list) -> DependencyParse:
        """Build a DependencyParse from the output of your loaded finetuned model.

        If self.mst == True, apply MST decoding. Otherwise use argmax decoding.        
        """
        # TODO: Your code here!
        with open('rel_pos_idxs.pickle', 'rb') as f:
            rel_pos_idxs = pickle.load(f)

        with open('dep_label_idxs.pickle', 'rb') as f:
            dep_label_idxs = pickle.load(f)

        n_rel_classes = len(rel_pos_idxs)
        n_dep_classes = len(dep_label_idxs)

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        classifier = distilBERT_FT(model, 768, n_rel_classes, n_dep_classes)

        classifier.load_state_dict(self.trained_model)

        #print(classifier)
        encoded_input = tokenizer(sentence, return_tensors='pt', padding=True)
        #print('encoded_input: ', encoded_input['input_ids'])
        print('encoded_input size: ', encoded_input['input_ids'].size())
        #print('UD Tokens: ', tokens)
        print('len of UD tokens: ', len(tokens))

        rel_output, dep_output = classifier(encoded_input)

        bert_tokens = ['cls'] + tokenizer.tokenize(sentence) + ['sep']
        print('bert tokens length: ', len(bert_tokens))
        #print('text of bert tokens: ', bert_tokens)

        print('rel_output size: ', rel_output.size())
        # print('n_rel_classes: ', n_rel_classes)
        print('dep_output size: ', dep_output.size())
        # print('n_dep_classes: ', n_dep_classes)

        pred_rels, filtered_rel_scores = self.filter_scores_and_preds(rel_output, len(tokens), bert_tokens, n_rel_classes, rel_pos_idxs)
        pred_deprel, filtered_dep_scores = self.filter_scores_and_preds(dep_output, len(tokens), bert_tokens, n_dep_classes, dep_label_idxs)


        if self.mst: 
            print('USING MST TO DECODE PREDICTIONS')
            G = nx.DiGraph()
            #print('filtered_rel_scores', filtered_rel_scores)

            num_filtered_tokens = filtered_rel_scores.size()[1]

            for idx, token_scores in enumerate(filtered_rel_scores[0]): 
                #print('token scores: ', token_scores)
                log_probs = F.log_softmax(token_scores)
                #print('log_probs: ', log_probs)

                G.add_node(idx)

                for t_idx in range(num_filtered_tokens): 
                    if t_idx != idx: 
                        rel_pos = t_idx - idx

                        if rel_pos in rel_pos_idxs: 
                            rel_class = rel_pos_idxs[rel_pos]
                        else: 
                            rel_class = rel_pos_idxs['unk']
                            
                        w = log_probs[rel_class]
                        #print('w: ', w.item())
                        G.add_edge(t_idx, idx, weight = w.item())

            print('G: ', G)
                
            MST = maximum_spanning_arborescence(G)
            print('MST edges: ', MST.edges())

            pred_head = ['0'] * len(MST.nodes()) 

            for edge in MST.edges(): 
                node = edge[1]
                head = edge[0] + 1
                pred_head[node] = str(head)
            
            print('pred_head: ', pred_head)

        else:
            print('USING ARGMAX TO DECODE PREDICTIONS')

            pred_head = []
            for idx, pred in enumerate(pred_rels): 
                if type(pred) != int: 
                    pred_head.append('-1')
                elif pred == 0: 
                    pred_head.append('0')
                else: 
                    pred_head.append(str(idx + pred + 1))

        data_dict = {'text': sentence, 'tokens': tokens, 'head': pred_head, 'deprel': pred_deprel}

        return DependencyParse.from_huggingface_dict(data_dict)







