import torch
from torch.utils.tensorboard import SummaryWriter
import steps as nw
import data_module as dm
from experimnental_setup import *
from eval_modul import plot_conf, get_metrics, list_mean
from operator import itemgetter
import numpy as np
from Baseline import Baseline
from MSC_CNN import OctMSC_CNN, MSC_CNN
from SGB.SGB import create_xml
import random


def find_indices(list_to_check, item_to_find):
    """
    function to find all indices of a given item in a given list

    Params:
        list_to_check: given list to search
        item_to_find: item to find in a given list

    Returns:
        list containing indices of searched item in searched list
    """
    return [idx for idx, value in enumerate(list_to_check) if value == item_to_find]


def map_sockets_to_labels(sockets: list, pred_labels: list, path_to_model_dict: str):
    """
    function for sgb creation
    assigns semantic neuron label to a socket of robot control box via the activated neuron of the network
    in socket_pred the value is the socket number and the key is the activated neuron
    in final_dict the key is the socket number and the value is the semantic label of the neuron
    
    Params:
        sockets: list containing the sockets where corresponding data was measured
        pred_labels: list containing prediction of network for corresponding data
        path_to_model_dict: path to a json file that contains the assignment of the output neurons the network to their semantic label
    
    Returns:
        final_dict: dictionary that assigns a semantic label to a socket number
    """
    uniques = np.unique(sockets)
    with open(path_to_model_dict, "r") as json_file:
        model_dict = json.load(json_file)

    socket_pred = {}
    final_dict = {}
    # map activated neuron to socket number
    for i in uniques:
        socket_idx = find_indices(sockets, i)
        filtered_list = list(itemgetter(*socket_idx)(pred_labels))
        pred = round(list_mean([int(item) for item in filtered_list]))
        socket_pred[pred] = str(i)

    # map semantic label to socket number via activated neuron
    for i in range(len(uniques)):
        final_dict[model_dict.get(str(i))] = socket_pred.get(i)

    return final_dict


def train_model(hparam: dict):
    """
    function to train model

    Params:
        hparam: dict with hyperparameters for one model
    """
    # set seed
    torch.cuda.manual_seed(hparam["MAN_SEED"])
    torch.manual_seed(hparam["MAN_SEED"])
    random.seed(hparam["MAN_SEED"])
    # load logger and device
    logger = SummaryWriter(log_dir=hparam["LOG_DIR"] + "/logs_exp/" + str(hparam["EXPERIMENT_ID"]) + "/")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    i = 0
    # load dataloader
    dl_train = dm.DataModule(hparam, flag="train").train_loader()
    dl_val = dm.DataModule(hparam, flag="train").val_loader()
    dl_test = dm.DataModule(hparam, flag="train").test_loader()

    # chose model
    if hparam["MODEL"] == "Baseline_CNN":
        model = Baseline(hparam)
    elif hparam["MODEL"] == "OctMSC_CNN":
        model = OctMSC_CNN(hparam)
    elif hparam["MODEL"] == "MSC_CNN":
        model = MSC_CNN(hparam)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparam["LEARNING_RATE"])

    # get networkshape with LazyModules
    x_dummy = torch.empty(hparam["BATCH_SIZE"], hparam["SAMPLE_SIZE"]).to(device)
    model.forward(x_dummy,1)
    print(f"**********************************************************************\nModel structure: {model}\n")
    print(f"Number of parameters: {sum(param.numel() for param in model.parameters())}")
    print("***********************************************************************\n")

    train_loss_l = []
    # iterate of epochs
    for e in range(hparam["MAX_EPOCHS"]):
        print("Epoch " + str(e + 1) + " of " + str(hparam["MAX_EPOCHS"]))
        loss_train, i = nw.train_step(model=model, dl=dl_train, device=device, logger=logger, optimizer=optimizer, i=i, e=e)
        logger.add_scalar("Training_loss_epoch", loss_train, e)
        train_loss_l.append(loss_train)
        if e % 5 == 0:
            loss_val, predictions, labels, logits = nw.val_step(model=model, dl=dl_val, device=device, e=e)
            # compute metrics
            val_acc, val_f1, val_mcc, val_ck = \
                get_metrics(torch.tensor(predictions), torch.tensor(labels), num_classes=55)

            if torch.isnan(loss_val):
                break

            print(val_acc.item())
            print(val_f1.item())
            print(val_mcc.item())
            print(val_ck.item())

            logger.add_scalar("Validation_Accuracy", val_acc.item(), e)
            logger.add_scalar("Validation_F1", val_f1.item(), e)
            logger.add_scalar("Validation_MCC", val_mcc.item(), e)
            logger.add_scalar("Validation_CohenK", val_ck.item(), e)
            logger.add_scalar("Validation_loss_epoch", loss_val.item(), e)

    # testing
    test_loss, predictions, labels, logits = nw.test_step(model=model, dl=dl_test, device=device, e=e)
    test_acc, test_f1, test_mcc, test_ck = get_metrics(torch.tensor(predictions), torch.tensor(labels), num_classes=55)

    logger.flush()
    logger.close()
    torch.save(model.state_dict(), hparam["LOG_DIR"] + "/logs_exp/" + str(hparam["EXPERIMENT_ID"]) + "/model.path")
    save_results(hparam=hparam, metrics={"loss_test": test_loss.item(),
        "acc_test":test_acc.item(), "f1_test": test_f1.item(), "mcc_test": test_mcc.item(),
        "CohenK_test": test_ck.item()})


def run(hparams: dict):
    """
    function for training multiple models with different hyperparameters

    Params:
        hparams: nested dictionary with multiple dicts with hyperparameters
    """
    errors = []

    for i in hparams:
        exp = hparams[i]
        try:
            print(exp)
            train_model(exp)
        except:
           errors.append(f"Experiment no. {exp['EXPERIMENT_ID']} with model {exp['MODEL']} failed.")
    for error in errors: print(error)
    return


def run_inf(model_folder: str, model_name: str, num_classes: int, path_inf_data: str, orig_ds: str, inf_ds: str, save_path: str):
    """
    function used for inference and evalutation.
    iterates over multiple models in given folder, computes metrics for each iteration and combines them.
    prints metrics from get_metrics and plot confusion matrix.

    Params:
        model_folder: folder with folders of models with hyperparameter json file and state dict
        model_name: name of model
        num_classes: number of classes of classification problem
        path_inf_data: path to data for inference
        orig_ds: name of dataset the model was originally trained on
        inf_ds: name of dataset for inference
        save_path: save path for confusion matrix
    """
    pred_list, label_list = [], []
    for i in range(5):
        path = model_folder + str(i) + "/results.json"
        save_path = model_folder + str(i) + "/model.path"
        hparam = load_old_experiments(path)["setup"]

        if model_name == "Baseline_CNN":
            model = Baseline(hparam)
        elif model_name == "OctMSC_CNN":
            model = OctMSC_CNN(hparam)
        elif model_name == "MSC_CNN":
            model = MSC_CNN(hparam)

        model.load_state_dict(torch.load(save_path))
        #for param_tensor in model.state_dict():
        #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        dl_test = dm.DataModule(hparam, flag="inference", path_inf_data=path_inf_data).inference_loader()
        _, predictions, labels, _ = nw.inference_step(model, dl_test, device, 1)
        pred_list = pred_list + predictions
        label_list = label_list + labels

    acc_pt = get_metrics(torch.tensor(pred_list), torch.tensor(label_list), num_classes=num_classes)
    print(acc_pt)
    plot_conf(pred_list, label_list, num_classes, model_name, orig_ds, inf_ds, save_path)


def run_sgb(model_path: str, model_path_state: str, model_name: str, path_inf_data: str, path_network_mapping: str):
    """
    loads a trained model with its hparams and applies it to measured data.
    maps sockets to semantic labels and creates sgb as xml from this mapping.

    Params:
        model_path: path to dictionary with model hyperparameters
        model_path_state: path to state dict of model
        model_name: name of model
        path_inf_data: path to data of robot for which sgb should be created
        path_network_mapping: path to json file with networkmapping (semantic to neuron)

    """

    robot_base_data = {"Vendor_name": "Universal Robots", "Model_name": "UR10 - eSeries", "ident_number": "0x02c6"}

    hparam = load_old_experiments(model_path)["setup"]
    dl_sgb = dm.DataModule(hparam, flag="sgb", path_inf_data=path_inf_data).sgb_loader()

    if model_name == "Baseline_CNN":
        model = Baseline(hparam)
    elif model_name == "OctMSC_CNN":
        model = OctMSC_CNN(hparam)
    elif model_name == "MSC_CNN":
        model = MSC_CNN(hparam)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(model_path_state))
    model.to(device)
    sockets, pred_labels = nw.sgb_step(model, dl_sgb, device)
    mapping_socket_name = map_sockets_to_labels(sockets=sockets, pred_labels=pred_labels, path_to_model_dict=path_network_mapping)

    create_xml("I:\BA\Code\SGB", mapping_socket_name, robot_base_data)


if __name__ == "__main__":
    #current_hparams = exp.load_experiments()
    #run(current_hparams)
    #run_inf(model_folder=r"I:\BA\Code\logs\Finales_Baseline_auf_UR3/",model_name="OctMSC_CNN", num_classes=55,
    #        path_inf_data=r"I:/BA/Code/datasets/ur10_MTL_var_1.csv", orig_ds="UR3", inf_ds="UR10e (w)", save_path="I:\BA\Plots/Confusion_matrix/")
    a = run_sgb("I:\BA\Code\logs/Finales_OctMULTI_auf_UR10/1/results.json", "I:\BA\Code\logs/Finales_OctMULTI_auf_UR10/1/model.path", "OctMSC_CNN",
           r"I:/BA/Code/datasets/ur10_MTL_var_1.csv", r"I:\BA\Code\SGB\model_neuron_semantic_label_assignment.json")