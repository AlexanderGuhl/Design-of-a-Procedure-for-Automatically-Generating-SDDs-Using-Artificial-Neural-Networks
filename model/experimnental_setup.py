import json
from eval_modul import launchTensorboard


def make_exp_grid():
    """
    creates a grid of experiments from the parameter lists data_dir, man_seed, batch_size, lr, max_epochs, sample_size,
    activation_funcs, models, alpha, mp_kernel

    Returns:
        experiments_dict: dictionary with sub dictionaries for each experiment
    """
    i = 0
    data_dir = ["I:/BA/Code/datasets/ur10_MTL_var_1.csv"]   # directory of data for training
    man_seed = [1, 2, 3, 4, 42]                             # manual seeds for reproducible training
    batch_size = [102]                                      # number of subdatasets per batch
    lr = [0.0001]                                           # learning rate
    max_epochs = [100]                                      # maximum number of epochs for training
    sample_size = [4000]                                    # size of sample in datasteps
    activation_funcs = ["ReLU"]                             # used activation functions
    models = ["OCT_MULTI"]                                  # used models
    alpha = [0.75]                                          # used alpha for octave convolution
    mp_kernel = [25]                                        # size of maxpool kernel
    experiments_dict = {}
    for k in man_seed:
        for l in batch_size:
            for m in lr:
                for n in sample_size:
                    for p in mp_kernel:
                        for o in max_epochs:
                            for q in models:
                                for r in alpha:
                                    for s in data_dir:
                                            experiments_dict[i] = {
                                                "EXPERIMENT_ID": i,
                                                "DATA_DIR": [s],
                                                "LOG_DIR": "I:/BA/Code/logs",
                                                "MAN_SEED": k,
                                                "BATCH_SIZE": l,
                                                "NUM_WORKERS": 1,
                                                "LEARNING_RATE": m,
                                                "MODEL": q,
                                                "SAMPLE_SIZE": n,
                                                "MAX_EPOCHS": o,
                                                "ACTIVATION_FUNCTION": "ReLU",
                                                "ALPHA": r,
                                                "MAXPOOL_KERNEL": p,
                                                "KERNEL_FACTOR": 0.1,
                                                }
                                            i += 1
    return experiments_dict


def experiments_to_json(experiment_dict: dict):
    """
    creates a json file from experiments dictionary

    Params:
        experiment_dict: dictionary with experiments
    """
    experiments = experiment_dict
    with open("experiments.json", "w") as json_file:
        json.dump(experiments, json_file, indent=4)

    print("experimental grid was created and saved in experiments.json\n\n")


def load_experiments():
    """
    loads experiments json file

    Returns:
        hparam: dictionary with all experiments as sub dictionaries
    """
    with open('experiments.json') as json_file:
        hparam = json.load(json_file)

    return hparam


def load_old_experiments(path):
    """
    loads experiments from json file in save folder of experiment

    Params:
        path: path to folder of experiment
    Returns:
        hparam: dictionary with all experiments as sub dictionaries
    """
    with open(path) as json_file:
        hparam = json.load(json_file)

    return hparam


def create_exp():
    """
    creates experiments and saves them as json file
    """
    experiments = make_exp_grid()
    experiments_to_json(experiments)


def save_results(hparam: dict, metrics: dict):
    """
    saving results and experimental setup as .json file in log-dir
    metrics must be passed as dict: [train_y_MSE, train_X_MSE, test_y_MSE, test_X_MSE]

    Params:
        hparam: dictionary of hparams for the current model
        metrics: dictionary of test metrics of current model
    """
    results = {"setup": hparam, "results": metrics}
    path = hparam["LOG_DIR"] + "/logs_exp/" + str(hparam["EXPERIMENT_ID"]) + "/results.json"
    with open(path, "w") as file:
        json.dump(results, file, indent=4)
        file.close()
    return print(f"Logged results of experiment ", hparam["EXPERIMENT_ID"])

# Quickrun
if __name__ == "__main__":
    #launchTensorboard()
    create_exp()
    #hparam = load_experiments()
    #hparam = hparam["0"]