"""
Parent class for training classes, such as TCTrainer and QATrainer
"""
import torch
from dataclasses import dataclass
import tempfile
from transformers import TrainingArguments, Trainer
from ray.tune.schedulers import PopulationBasedTraining
from ray import tune
from ray.tune import CLIReporter
@dataclass
class EvalResult:
    loss: float

class HappyTrainer:
    def __init__(self, model, model_type, tokenizer, device, logger):
        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.device = device
        self.logger = logger

    def train(self, input_filepath,  eval_filepath, args):
        """

        :param input_filepath: A string to file location
        :param args: a dictionary that contains settings
        :return:
        """
        raise NotImplementedError()

    def test(self, input_filepath, solve, args):
        """

        :param input_filepath: A string to file location
        :param solve: a method for using the model for the given task
        :return: test results
        """
        raise NotImplementedError()

    def eval(self, input_filepath, args):
        """
        :param input_filepath: A string to file location
        :args a dictionary that contains settings
        :return: a dictionary that contains a key called "eval_loss" that holds the loss
         for the given eval dataset. May add more metrics later
        """
        raise NotImplementedError()

    @staticmethod
    def _get_data(filepath, test_data=False):
        """

        :param filepath:  A string to file location
        :param test_data: False for train and eval, True for test
        :return: varies for each task
        """
        raise NotImplementedError()

    def _get_training_args(self, dataclass_args, output_path):
        """
        :param args: a dataclass of arguments for training
        :param output_path: A string to a temporary directory
        :return: A TrainingArguments object
        """
        if self.device != "cuda":
            if dataclass_args.fp16:
                ValueError("fp16 is only available when CUDA/ a GPU is being used. ")

        return TrainingArguments(
            output_dir=dataclass_args.output_dir,
            resume_from_checkpoint=dataclass_args.resume_from_checkpoint,
            do_train=True,
            do_eval=True,
            learning_rate=dataclass_args.learning_rate,
            weight_decay=dataclass_args.weight_decay,
            adam_beta1=dataclass_args.adam_beta1,
            adam_beta2=dataclass_args.adam_beta2,
            adam_epsilon=dataclass_args.adam_epsilon,
            max_grad_norm=dataclass_args.max_grad_norm,
            num_train_epochs=dataclass_args.num_train_epochs,
            report_to=["wandb"],
            save_strategy="steps",
            evaluation_strategy="steps",
            eval_steps=2000,
            save_total_limit=5,
            fp16=dataclass_args.fp16,
            per_device_train_batch_size=dataclass_args.batch_size,
            logging_dir=dataclass_args.logging_dir,
            logging_strategy="steps",
            logging_steps=2000,

        )


    def _run_tune(self,  dataset, eval_dataset, dataclass_args, data_collator, model_init):
        """

                :param dataset: a child of torch.utils.data.Dataset
                :param dataclass_args: a dataclass that contains settings
                :return: None
                """
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            training_args = self._get_training_args(dataclass_args, tmp_dir_name)
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )
            tune_config = {
                "per_device_train_batch_size": 8,
                "per_device_eval_batch_size": 8,
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
                    # "max_grad_norm": tune.uniform(0.0, 1.0), # new
                    # "adam_epsilon": tune.uniform(0.0, 0.3) # new
                },
                require_attrs=False
            )

            reporter = CLIReporter(
                parameter_columns={
                    "weight_decay": "w_decay",
                    "learning_rate": "lr",
                    "num_train_epochs": "num_epochs",
                    # "max_grad_norm": "max_grad_norm",
                    # "adam_epsilon": "adam_eps"
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

    def _run_train(self, dataset, eval_dataset, dataclass_args, data_collator):
        """

        :param dataset: a child of torch.utils.data.Dataset
        :param dataclass_args: a dataclass that contains settings
        :return: None
        """
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            training_args = self._get_training_args(dataclass_args, tmp_dir_name)
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )
            trainer.train(resume_from_checkpoint=dataclass_args.resume_from_checkpoint)

    def _run_eval(self, dataset, data_collator, dataclass_args):
        """
        :param dataset: a child of torch.utils.data.Dataset
        :return: None
        """
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            eval_args = self._get_eval_args(tmp_dir_name, dataclass_args)
            trainer = Trainer(
                model=self.model,
                args=eval_args,
                eval_dataset=dataset,
                data_collator=data_collator
            )
            return trainer.evaluate()

    @staticmethod
    def _get_eval_args(output_path, dataclass_args):
        """
        :param output_path: A string to a temporary directory
        :return: A TrainingArguments object
        """
        return TrainingArguments(
            output_dir=output_path,
            seed=42,
            report_to=['none'],
            per_device_eval_batch_size=dataclass_args.batch_size,

        )
