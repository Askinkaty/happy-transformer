ARGS_TC_TRAIN = {
    'learning_rate': 5e-5,
    'num_train_epochs': 3.0,
    'batch_size': 8,
    'weight_decay': 0,
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'adam_epsilon': 1e-8,
    'max_grad_norm':  1.0,
    'save_preprocessed_data': False,
    'save_preprocessed_data_path': "",
    'load_preprocessed_data': False,
    'load_preprocessed_data_path': "",
    "resume_from_checkpoint": False,
    "output_dir": "",
    'fp16': False,
    'max_len': 500,
    'logging_dir': "",
    'logging_strategy': "",
    'logging_steps': 2000

}

ARGS_TC_EVAL = {
    'batch_size': 1,
    'save_preprocessed_data': False,
    'save_preprocessed_data_path': "",
    'load_preprocessed_data': False,
    'load_preprocessed_data_path': "",
    'max_len': 200
}

ARGS_TC_TEST = {
    'save_preprocessed_data': False,
    'save_preprocessed_data_path': "",
    'load_preprocessed_data': False,
    'load_preprocessed_data_path': "",
    'max_len': 200
}

