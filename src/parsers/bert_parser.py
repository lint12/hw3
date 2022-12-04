from src.parsers.parser import Parser
from src.dependency_parse import DependencyParse
# from src.bert_parser_model import BertParserModel
import torch
import torch.nn as nn
import pickle
from transformers import DistilBertTokenizer, DistilBertModel

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
        # tokens here are UD tokens
        filtered_scores = torch.zeros((1, ud_token_len, n_classes))
    
        new_idx = 0 
        hyphen_prev = False
        for idx, token_scores in enumerate(output[0]): 
            #print('idx: ', idx)
            # if its the 0th index or last index then skip because they rep. bos and eos 
            if idx == 0 or idx == len(bert_tokens) - 1: 
                continue 
            elif idx >= ud_token_len: 
                continue
            elif bert_tokens[idx].startswith('##'): 
                continue
            elif bert_tokens[idx] == '-': 
                hyphen_prev = True 
                continue
            elif hyphen_prev == True: 
                hyphen_prev = False 
                continue 
            else: 
                #print('rel_new_idx: ', rel_new_idx)
                filtered_scores[0][new_idx] = token_scores
                new_idx += 1

        print('filtered_scores size: ', filtered_scores.size())
        #print('filtered_scores: ', filtered_scores)

        class_to_label = dict((v,k) for k,v in idx_mapping.items())

        preds = []
        for token_scores in filtered_scores[0]: 
            pred = torch.argmax(token_scores).item() # class idx
            preds.append(class_to_label[pred]) # label

        print('preds: ', preds)
        return preds 

    def parse(self, sentence: str, tokens: list) -> DependencyParse:
        """Build a DependencyParse from the output of your loaded finetuned model.

        If self.mst == True, apply MST decoding. Otherwise use argmax decoding.        
        """
        # TODO: Your code here!
        if self.mst: 
            print('USING MST TO DECODE PREDICTIONS')
        else:
            print('USING ARGMAX TO DECODE PREDICTIONS')

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

            bert_tokens = ['bos'] + tokenizer.tokenize(sentence) + ['eos']
            print('bert tokens length: ', len(bert_tokens))
            #print('text of bert tokens: ', bert_tokens)

            print('rel_output size: ', rel_output.size())
            print('n_rel_classes: ', n_rel_classes)
            print('dep_output size: ', dep_output.size())
            print('n_dep_classes: ', n_dep_classes)

            pred_rels = self.filter_scores_and_preds(rel_output, len(tokens), bert_tokens, n_rel_classes, rel_pos_idxs)
            pred_deprel = self.filter_scores_and_preds(dep_output, len(tokens), bert_tokens, n_dep_classes, dep_label_idxs)

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







