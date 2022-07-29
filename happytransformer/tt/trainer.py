"""
Fine-tune Text-to-text models

Based on the following sources:
1. https://github.com/huggingface/notebooks/blob/master/examples/summarization.ipynb
2. https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization.py
3. https://github.com/huggingface/transformers/blob/master/examples/pytorch/translation/run_translation.py
4. https://huggingface.co/transformers/
5. https://huggingface.co/docs/datasets/
"""

from dataclasses import dataclass
from happytransformer.happy_trainer import HappyTrainer, EvalResult
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import tempfile
from ray.tune.schedulers import PopulationBasedTraining
from ray import tune
from ray.tune import CLIReporter


@dataclass
class TTTrainArgs:
    """
    Used to adjust the settings when calling HappyTextToText.train()
    """
    learning_rate: float = 5e-5
    num_train_epochs: int = 5
    weight_decay: float = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    save_preprocessed_data: bool = False
    save_preprocessed_data_path: str = ""
    load_preprocessed_data: bool = False
    load_preprocessed_data_path: str = ""
    preprocessing_processes: int = 1
    batch_size: int = 1
    max_input_length: int = 1024
    max_output_length: int = 1024
    fp16: bool = True
    output_dir: str = ""
    resume_from_checkpoint: str = ""
    generation_num_beams: int = 5
    logging_dir: str = ""
    logging_strategy: str = ""
    logging_steps: int = 20000
    num_samples: int = 5
    gpus_per_trial: int = 3


@dataclass
class TTEvalArgs:
    """
    Used to adjust the settings when calling HappyTextToText.eval()
    """
    batch_size: int = 1
    save_preprocessed_data: bool = False
    save_preprocessed_data_path: str = ""
    load_preprocessed_data: bool = ""
    load_preprocessed_data_path: str = ""
    preprocessing_processes: int = 1
    max_input_length: int = 1024
    max_output_length: int = 1024
    logging_dir: str = ""
    logging_strategy: str = ""
    logging_steps: int = 20000



@dataclass
class TTTestArgs:
    """
    Used to adjust the settings when calling HappyTextToText.test()
    """
    save_preprocessed_data: bool = False
    save_preprocessed_data_path: str = ""
    load_preprocessed_data: bool = False
    load_preprocessed_data_path: str = ""


class TTTrainer(HappyTrainer):
    """
    Trainer class for HappyTextToText
    """

    def __init__(self, model, model_type, tokenizer, device, logger):
        super().__init__(model, model_type, tokenizer, device, logger)
        self.__max_input_length = 1024
        self.__max_output_length = 1024

    def tune_parameters(self, model_init, input_filepath, eval_filepath, dataclass_args=TTTrainArgs):
        """
        :param input_filepath: A file path to a csv file that contains two columns: text_1 and text_2
        :param dataclass_args: A TTTrainArgs() object
        :return: None
        """

        if dataclass_args.save_preprocessed_data:
            self.logger.info("Saving preprocessed data is currently "
                             "not available for text-to-text models. "
                             )
        if dataclass_args.load_preprocessed_data:
            self.logger.info("Loading preprocessed data is currently "
                             "not available for text-to-text models. "
                             )

        self.logger.info("Preprocessing training data...")

        dataset = load_dataset("csv", data_files={"train": input_filepath}, delimiter=",")
        eval_dataset = load_dataset("csv", data_files={"eval": eval_filepath}, delimiter=",")

        self.__max_input_length = dataclass_args.max_input_length
        self.__max_output_length = dataclass_args.max_output_length

        tokenized_dataset = dataset.map(
            self.__preprocess_function,
            batched=True,
            num_proc=dataclass_args.preprocessing_processes,
            remove_columns=["input", "target"],
        )
        tokenized_eval_dataset = eval_dataset.map(
            self.__preprocess_function,
            batched=True,
            num_proc=dataclass_args.preprocessing_processes,
            remove_columns=["input", "target"],
        )

        self.logger.info("Training...")

        # Pads inputs and labels to max length
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        # A temp dir is used so any files that are generated are deleted after training
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            training_args = Seq2SeqTrainingArguments(
                # tmp_dir_name,
                output_dir=dataclass_args.output_dir,
                resume_from_checkpoint=dataclass_args.resume_from_checkpoint,
                generation_num_beams=dataclass_args.generation_num_beams,
                do_train=True,
                do_eval=True,
                learning_rate=dataclass_args.learning_rate,
                weight_decay=dataclass_args.weight_decay,
                adam_beta1=dataclass_args.adam_beta1,
                adam_beta2=dataclass_args.adam_beta2,
                adam_epsilon=dataclass_args.adam_epsilon,
                max_grad_norm=dataclass_args.max_grad_norm,
                num_train_epochs=dataclass_args.num_train_epochs,
                report_to=["none"],
                save_strategy="steps",
                evaluation_strategy="steps",
                eval_steps=20000,
                save_total_limit=5,
                per_device_train_batch_size=dataclass_args.batch_size,
                fp16=True,
                logging_dir=dataclass_args.logging_dir,
                logging_strategy="steps",
                logging_steps=20000,
            )

            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset['train'],
                eval_dataset=tokenized_eval_dataset['eval'],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                model_init=model_init,
            )

            tune_config = {
                "per_device_train_batch_size": 2,
                "per_device_eval_batch_size": 2,
                "num_train_epochs": tune.choice([2, 3, 4, 5]),
                "max_steps": -1,
            }

            scheduler = PopulationBasedTraining(
                time_attr="training_iteration",
                metric="eval_acc",
                mode="max",
                perturbation_interval=1,
                hyperparam_mutations={
                    "weight_decay": tune.uniform(0.0, 0.3),
                    "learning_rate": tune.uniform(1e-5, 5e-5),
                },
            )

            reporter = CLIReporter(
                parameter_columns={
                    "weight_decay": "w_decay",
                    "learning_rate": "lr",
                    "num_train_epochs": "num_epochs",
                },
                metric_columns=["eval_acc", "eval_loss", "epoch", "training_iteration"],
            )

            trainer.hyperparameter_search(
                hp_space=lambda _: tune_config,
                backend="ray",
                n_trials=dataclass_args.num_samples,
                resources_per_trial={"cpu": 1, "gpu": dataclass_args.gpus_per_trial},
                scheduler=scheduler,
                keep_checkpoints_num=1,
                checkpoint_score_attr="training_iteration",
                stop=None,
                progress_reporter=reporter,
                local_dir="/scratch/project_2002016/ray_results/",
                name="tune_transformer_pbt",
                log_to_file=True,
            )

    def train(self, input_filepath, eval_filepath, dataclass_args=TTTrainArgs):
        """
        :param input_filepath: A file path to a csv file that contains two columns: text_1 and text_2
        :param dataclass_args: A TTTrainArgs() object
        :return: None
        """

        if dataclass_args.save_preprocessed_data:
            self.logger.info("Saving preprocessed data is currently "
                             "not available for text-to-text models. "
                             )
        if dataclass_args.load_preprocessed_data:
            self.logger.info("Loading preprocessed data is currently "
                             "not available for text-to-text models. "
                             )

        self.logger.info("Preprocessing training data...")

        dataset = load_dataset("csv", data_files={"train": input_filepath}, delimiter=",")
        eval_dataset = load_dataset("csv", data_files={"eval": eval_filepath}, delimiter=",")

        self.__max_input_length = dataclass_args.max_input_length
        self.__max_output_length = dataclass_args.max_output_length

        tokenized_dataset = dataset.map(
            self.__preprocess_function,
            batched=True,
            num_proc=dataclass_args.preprocessing_processes,
            remove_columns=["input", "target"],
        )
        tokenized_eval_dataset = eval_dataset.map(
            self.__preprocess_function,
            batched=True,
            num_proc=dataclass_args.preprocessing_processes,
            remove_columns=["input", "target"],
        )

        self.logger.info("Training...")

        # Pads inputs and labels to max length
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        # A temp dir is used so any files that are generated are deleted after training
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            training_args = Seq2SeqTrainingArguments(
                # tmp_dir_name,
                output_dir=dataclass_args.output_dir,
                resume_from_checkpoint=dataclass_args.resume_from_checkpoint,
                generation_num_beams=dataclass_args.generation_num_beams,
                do_train=True,
                do_eval=True,
                learning_rate=dataclass_args.learning_rate,
                weight_decay=dataclass_args.weight_decay,
                adam_beta1=dataclass_args.adam_beta1,
                adam_beta2=dataclass_args.adam_beta2,
                adam_epsilon=dataclass_args.adam_epsilon,
                max_grad_norm=dataclass_args.max_grad_norm,
                num_train_epochs=dataclass_args.num_train_epochs,
                report_to=["none"],
                save_strategy="steps",
                evaluation_strategy="steps",
                eval_steps=20000,
                save_total_limit=5,
                per_device_train_batch_size=dataclass_args.batch_size,
                fp16=True,
                logging_dir=dataclass_args.logging_dir,
                logging_strategy="steps",
                logging_steps=20000,
            )

            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset['train'],
                eval_dataset=tokenized_eval_dataset['eval'],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )

            trainer.train(resume_from_checkpoint=dataclass_args.resume_from_checkpoint)

    def eval(self, input_filepath, dataclass_args=TTEvalArgs):
        """
        Evaluates the model by determining the loss.
        :param input_filepath: A file path to a csv file that contains two columns: text_1 and text_2
        :param dataclass_args: A TTEvalArgs() object
        :return: a EvalResult object that contains the loss
        """
        self.logger.info("Preprocessing evaluating data...")
        dataset = load_dataset("csv", data_files={"eval": input_filepath}, delimiter=",")

        self.__max_input_length = dataclass_args.max_input_length
        self.__max_output_length = dataclass_args.max_output_length

        tokenized_dataset = dataset.map(
            self.__preprocess_function,
            batched=True,
            remove_columns=["input", "target"],
            num_proc=dataclass_args.preprocessing_processes,
        )

        # Pads inputs and labels to max length
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        # A temp dir is used so any files that are generated are deleted after evaluating
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            eval_args = Seq2SeqTrainingArguments(
                tmp_dir_name,
                do_train=False,
                do_eval=True,
                seed=42,
                report_to=["none"],
                per_device_eval_batch_size=dataclass_args.batch_size,
                logging_dir=dataclass_args.logging_dir,
                logging_strategy="steps",
                logging_steps=20000,
                save_strategy="steps",
                evaluation_strategy="steps",
                eval_steps=20000,
            )

            trainer = Seq2SeqTrainer(
                model=self.model,
                args=eval_args,
                eval_dataset=tokenized_dataset['eval'],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )
            result = trainer.evaluate()
            return EvalResult(loss=result["eval_loss"])

    def __preprocess_function(self, examples):
        """
        :param examples:
        :return:
        """
        model_inputs = self.tokenizer(examples["input"], max_length=self.__max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples["target"], max_length=self.__max_output_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def test(self, input_filepath, solve, args=TTTestArgs):
        raise NotImplementedError()
