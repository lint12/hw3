from src.parsers.parser import Parser
from src.dependency_parse import DependencyParse

import spacy 

class SpacyParser(Parser):

    def __init__(self, model_name: str):
        # TODO: Your code here!
        self.model_name = model_name 

    def parse(self, sentence: str, tokens: list) -> DependencyParse:
        """Use the specified spaCy model to parse the sentence.py.

        The function should return the parse in the Dependency format.

        You should refer to the spaCy documentation for how to use the model for dependency parsing.
        """
        # Your code here!
        nlp = spacy.load(self.model_name)
        # doc = nlp(sentence)
        # spacy_tokens = [token.text for token in doc]
        # pred_head = [str(token.head.i) for token in doc]
        # pred_deprel = [(token.dep_).lower() for token in doc]

        spacy_tokens = []
        pred_head = []
        pred_deprel = []

        # form the sentence with the language model using the tokens passed in and 
        # then re-parse them with the language model 
        sent = spacy.tokens.doc.Doc(nlp.vocab, words=tokens)
        sent_tokens = nlp(sent)

        for token in sent_tokens: 
            if token.i == token.head.i: 
                head_i = '0'
            else: 
                head_i = str(token.head.i + 1)

            spacy_tokens.append(token.text)
            pred_head.append(head_i)
            pred_deprel.append(token.dep_)


        data_dict = {'text': sentence, 'tokens': spacy_tokens, 'head': pred_head, 'deprel': pred_deprel}

        return DependencyParse.from_huggingface_dict(data_dict)
