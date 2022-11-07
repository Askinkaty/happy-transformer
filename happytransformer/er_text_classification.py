"""
Contains a class called HappyTextClassification that performs text classification
"""
from dataclasses import dataclass

from transformers import TextClassificationPipeline, AutoConfig, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelWithLMHead

from happytransformer.tc.trainer import TCTrainer, TCTrainArgs, TCEvalArgs, TCTestArgs
from happytransformer.cuda_detect import detect_cuda_device_number
from happytransformer.happy_transformer import HappyTransformer
from happytransformer.adaptors import get_adaptor
from happytransformer.tc import ARGS_TC_TRAIN, ARGS_TC_EVAL, ARGS_TC_TEST
from happytransformer.happy_trainer import EvalResult
from happytransformer.fine_tuning_util import create_args_dataclass

import torch
import torch.nn as nn

@dataclass
class TextClassificationResult:
    label: str
    score: float


class SimpleGPT3SequenceClassifier(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_classes: int,
            max_seq_len: int,
            gpt_model_name: str,
            tokenizer: None
    ):
        super(SimpleGPT3SequenceClassifier, self).__init__()
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.gpt3model = AutoModelWithLMHead.from_pretrained(
            gpt_model_name, output_hidden_states=True
        )
        if gpt_model_name != "":
            self.gpt3model.resize_token_embeddings(len(tokenizer))
        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path=gpt_model_name, num_labels=num_classes)
        self.pool1 = nn.MaxPool1d(3, stride=5)
        self.pool2 = nn.MaxPool1d(3, stride=5)
        self.fc1 = nn.Linear(402000, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)


    def forward(self, x_in):
        #         print(x_in.shape)
        gpt_out = self.gpt3model(x_in)
        #         embeds = self.gpt3model.get_input_embeddings()(x_in)
        #         print(embeds.shape)
        #         layer_output = self.gpt3model.transformer.h[-1](embeds)[0]
        #         print(layer_output.shape)
        gpt_out = gpt_out[0]  # returns tuple

        #         print(gpt_out.shape)
        #         print(gpt_out[1].shape)
        #         print(gpt_out[2].shape)
        # torch.Size([16, 200, 50258])
        # torch.Size([16, 10051600])
        batch_size = gpt_out.shape[0]
        #         batch_size = layer_output.shape[0]
        #         print(layer_output.view(batch_size, -1).shape)
        #         last_token = layer_output[:,0,:]
        #         print(last_token.shape)
        #         print(gpt_out.view(batch_size,-1).shape)
        pooled1 = self.pool1(gpt_out)
        pooled2 = self.pool2(pooled1)
        #         print(pooled2.shape)
        concat = pooled2.view(batch_size, -1)
        #         print(concat.shape)
        f1 = self.fc1(concat)
        #         concat = pooled.view(batch_size,-1)
        #         print(f1.shape)
        prediction_vector = self.fc2(f1)  # (batch_size , max_len, num_classes)
        #         prediction_vector = self.fc1(last_token) #(batch_size , max_len, num_classes)
        return prediction_vector


class ErrTextClassification(HappyTransformer):
    """
    A user facing class for Text Classification
    """

    def __init__(self, model_type="GPT",
                 model_name="", num_labels: int = 2,
                 load_path: str = "",
                 use_auth_token: str = None, max_len: int = 200,
                 hidden_size: int = 1536):
        self.adaptor = get_adaptor(model_type)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        model = SimpleGPT3SequenceClassifier(
            hidden_size=hidden_size,
            num_classes=num_labels,
            gpt_model_name=model_name,
            max_seq_len=max_len,
            tokenizer=self.tokenizer)

        super().__init__(model_type, model_name, model, use_auth_token=use_auth_token, load_path=load_path)

        device_number = detect_cuda_device_number()
        self._pipeline = TextClassificationPipeline(
            model=self.model, tokenizer=self.tokenizer,
            device=device_number
        )

        self._trainer = TCTrainer(
            self.model, self.model_type,
            self.tokenizer, self._device, self.logger
        )

    def classify_text(self, text: str) -> TextClassificationResult:
        """
        Classify text to a label based on model's training
        """
        # Blocking allowing a for a list of strings
        if not isinstance(text, str):
            raise ValueError("the \"text\" argument must be a single string")
        results = self._pipeline(text)
        # we do not support predicting a list of  texts, so only first prediction is relevant
        first_result = results[0]

        return TextClassificationResult(label=first_result["label"], score=first_result["score"])

    def train(self, input_filepath, eval_filepath, args=TCTrainArgs()):
        """
        Trains the question answering model
        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header values: text, label
        args: Either a TCTrainArgs() object or a dictionary that contains all of the same keys as ARGS_TC_TRAIN
        return: None
        """
        if type(args) == dict:
            method_dataclass_args = create_args_dataclass(default_dic_args=ARGS_TC_TRAIN,
                                                          input_dic_args=args,
                                                          method_dataclass_args=TCTrainArgs)
        elif type(args) == TCTrainArgs:
            method_dataclass_args = args
        else:
            raise ValueError("Invalid args type. Use a TCTrainArgs object or a dictionary")

        self._trainer.train(input_filepath=input_filepath, eval_filepath=eval_filepath,
                            dataclass_args=method_dataclass_args)

    def eval(self, input_filepath, args=TCEvalArgs()) -> EvalResult:
        """
        Evaluated the text classification answering model
        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header values:
        text, label

        return: an EvalResult() object
        """
        if type(args) == dict:
            method_dataclass_args = create_args_dataclass(default_dic_args=ARGS_TC_EVAL,
                                                          input_dic_args=args,
                                                          method_dataclass_args=TCEvalArgs)
        elif type(args) == TCEvalArgs:
            method_dataclass_args = args
        else:
            raise ValueError("Invalid args type. Use a TCEvalArgs object or a dictionary")

        return self._trainer.eval(input_filepath=input_filepath, dataclass_args=method_dataclass_args)

    def test(self, input_filepath, args=TCTestArgs()):
        """
        Tests the text classification  model. Used to obtain results
        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header value:
         text
        return: A list of TextClassificationResult() objects
        """

        if type(args) == dict:
            method_dataclass_args = create_args_dataclass(default_dic_args=ARGS_TC_TEST,
                                                          input_dic_args=args,
                                                          method_dataclass_args=TCTestArgs)
        elif type(args) == TCTestArgs:
            method_dataclass_args = args
        else:
            raise ValueError("Invalid args type. Use a TCTestArgs() object or a dictionary")

        return self._trainer.test(input_filepath=input_filepath, solve=self.classify_text,
                                  dataclass_args=method_dataclass_args)
