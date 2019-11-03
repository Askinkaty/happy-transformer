import torch
from pytorch_transformers import BertForMaskedLM, BertTokenizer

from happy_transformer.happy_transformer import HappyTransformer


class HappyBERT(HappyTransformer):

    def __init__(self, model='bert-large-uncased'):
        super().__init__()
        self.transformer = BertForMaskedLM.from_pretrained(model)
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.masked_token = self.tokenizer._mask_token
        self.sep_token = self.tokenizer._sep_token
        self.cls_token = self.tokenizer._cls_token

        self.model = 'BERT'

        self.transformer.eval()

    def predict_mask(self, text: str):
        """
        :param text: a string with a masked token within it
        :return: predicts the most likely word to fill the mask and its probability
        """

        # TODO: put in HappyBERT. Overwrite HappyTransformer.
        # TODO: easy: create a method to check if the sentence is valid
        # TODO: easy: if the sentence is not valid, provide the user with input requirements
        # TODO: easy: if sentence is not valid, indicate where the user messed up

        # TODO: medium: make it more modular

        # Generated formatted text so that it can be tokenized. Mainly, add the required tags
        formatted_text = self._HappyTransformer__get_formatted_text(text)
        tokenized_text = self.tokenizer.tokenize(formatted_text)

        masked_index = self._HappyTransformer__get_prediction_index(tokenized_text)
        segments_ids = self._HappyTransformer__get_segment_ids(tokenized_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        tokens_tensor = tokens_tensor.to(self.gpu_support)
        segments_tensors = segments_tensors.to(self.gpu_support)

        with torch.no_grad():
            outputs = self.transformer(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]

            softmax = self._HappyTransformer__softmax(predictions)

            top_prediction = torch.topk(softmax[0, masked_index], 1)
            prediction_softmax = top_prediction[0].tolist()
            prediction_index = top_prediction[1].tolist()

            prediction_token = self.tokenizer.convert_ids_to_tokens(prediction_index)

           # TODO: easy: del various variables

            if self.gpu_support == "cuda":
                torch.cuda.empty_cache()

            return prediction_token, prediction_softmax