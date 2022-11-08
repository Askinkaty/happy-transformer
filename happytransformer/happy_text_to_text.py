"""
Contains a class called HappyTextToText which performs text to text generation
"""
from dataclasses import dataclass

from transformers import Text2TextGenerationPipeline, AutoModelForSeq2SeqLM, LogitsProcessorList, \
    MinLengthLogitsProcessor, BeamSearchScorer, StoppingCriteriaList, MaxLengthCriteria

from happytransformer.happy_transformer import HappyTransformer
from happytransformer.tt.trainer import TTTrainer
from happytransformer.cuda_detect import detect_cuda_device_number
from happytransformer.adaptors import get_adaptor
from happytransformer.tt.trainer import TTTrainArgs, TTEvalArgs, TTTestArgs
import torch


@dataclass
class TextToTextResult:
    """
    Returned when HappyTextToText.generate() is called
    """
    text: str


@dataclass
class TTSettings:
    """
    Used to adjust the text generation algorithm that's used when
    HappyTextToText.generate() is called

    """
    min_length: int = 3
    max_length: int = 100
    do_sample: bool = True
    early_stopping: bool = False
    num_beams: int = 5
    temperature: float = 1
    top_k: int = 50
    no_repeat_ngram_size: int = 0
    top_p: float = 0.95


class HappyTextToText(HappyTransformer):
    """
    A user facing class for text to text generation
    """

    def __init__(self, model_type: str = "T5", model_name: str = "t5-small", load_path: str = "",
                 use_auth_token: str = None):

        self.adaptor = get_adaptor(model_type)
        device = torch.device('cpu')

        if load_path != "":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(load_path)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=use_auth_token)

        super().__init__(model_type, model_name, self.model, use_auth_token=use_auth_token, load_path=load_path)

        device_number = detect_cuda_device_number()
        self._pipeline = Text2TextGenerationPipeline(model=self.model,
                                                     tokenizer=self.tokenizer, device=device_number)

        self._trainer = TTTrainer(self.model, model_type, self.tokenizer, self._device, self.logger)

    def __assert_default_text_is_val(self, text):
        """
        Ensures the input's text input is valid.
        Raises a Value Error if the text input is not valid.
        :param text: The value the user inputs for the "text" parameter
        """

        if not isinstance(text, str):
            raise ValueError("The text input must be a string")
        if not text:
            raise ValueError("The text input must have at least one character")

    def get_beam_seq(self, text: str, args: TTSettings = TTSettings()):
        
        torch.cuda.empty_cache()
        self.beam_scorer = BeamSearchScorer(batch_size=1, num_beams=args.num_beams, device=self.model.device,
                                           num_beam_hyps_to_keep=args.num_beams)
        self.logits_processor = LogitsProcessorList([MinLengthLogitsProcessor(5,
                                                                              eos_token_id=
                                                                              self.model.config.eos_token_id),])
        encoding = self.tokenizer(
            [text],
            padding="longest",
            max_length=args.max_length,
            truncation=True,
            return_tensors="pt",)
        encoder_input_ids, attention_mask = encoding.input_ids.to("cuda"), encoding.attention_mask.to("cuda")
        input_ids = torch.ones((args.num_beams, 1), device=self.model.device, dtype=torch.long)
        input_ids = input_ids * self.model.config.decoder_start_token_id
        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=args.max_length)])
        model_kwargs = {
            "encoder_outputs": self.model.get_encoder()(
            encoder_input_ids.repeat_interleave(args.num_beams, dim=0), return_dict=True)}
        outputs = self.model.beam_search(input_ids, self.beam_scorer,  output_scores=True,
                                         logits_processor=self.logits_processor,
                                         return_dict_in_generate=True,
                                         stopping_criteria=stopping_criteria, **model_kwargs)
        out = self.tokenizer.batch_decode(outputs.get('sequences'), skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
        return out, outputs.get('sequences_scores')

    def generate_text(self, text: str,
                      args: TTSettings = TTSettings()) -> TextToTextResult:
        """
        :param text: starting text that the model uses as a prompt to continue it.
        :param args: A TTSettings object
        :return: A TextToTextResult() object
        """
        self.__assert_default_text_is_val(text)

        output = self._pipeline(text, min_length=args.min_length,
                                max_length=args.max_length,
                                do_sample=args.do_sample,
                                early_stopping=args.early_stopping,
                                num_beams=args.num_beams,
                                temperature=args.temperature,
                                top_k=args.top_k,
                                no_repeat_ngram_size=args.no_repeat_ngram_size,
                                top_p=args.top_p,
                                )
        return TextToTextResult(text=output[0]['generated_text'])

    def train(self, input_filepath, eval_filepath, args=TTTrainArgs()):
        """
        Trains the text-to-text model
        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header values: text_1, text_2
        args: A TTTrainArgs() object
        return: None
        """
        self._trainer.train(input_filepath=input_filepath, eval_filepath=eval_filepath, dataclass_args=args)

    def tune_parameters(self, load_path, input_filepath, eval_filepath, args=TTTrainArgs()):
        def model_init():
            return AutoModelForSeq2SeqLM.from_pretrained(load_path)

        self._trainer.tune_parameters(model_init=model_init,
                                      input_filepath=input_filepath, eval_filepath=eval_filepath, dataclass_args=args)

    def eval(self, input_filepath, args=TTEvalArgs()):
        """
        Evaluated the text-to-text model

        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header values: text_1, text_2

        args: A TTEvalArgs() object
        return: an EvalResult() object
        """

        result = self._trainer.eval(input_filepath=input_filepath, dataclass_args=args)
        return result

    def test(self, input_filepath, args=TTTestArgs):
        raise NotImplementedError("test() is currently not available")
