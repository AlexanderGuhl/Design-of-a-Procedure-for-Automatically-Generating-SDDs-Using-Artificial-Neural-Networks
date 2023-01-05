import os
from collections import Counter
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score,MulticlassCohenKappa, MulticlassMatthewsCorrCoef, MulticlassConfusionMatrix


def launchTensorboard(log_dir="logs/Finale_Dinge/Final5/"):
    os.system('tensorboard --logdir=../"' + log_dir + '"')
    return


def list_mean(list):
        return sum(list)/len(list)


def plot_conf(predicted_labels, true_labels, num_classes, model_name, orig_dataset, inference_dateset,save_path):
    """
    plots, shows and saves confusionmatrix with all classes for multiclass or binary problem

    params:
        predicted_labels: predicted labels from model with shape batchsize
        true_labels: label taken from dataset with shape batchsize
        num_classes: number of classes for classification problem
        model_name: name of model
        orig_dataset: dataset the model was trained on
        inference_dateset: dataset model is applied to
        save_path: path where plot should be saved
    """
    metric = MulticlassConfusionMatrix(num_classes=num_classes)
    conf_matrix = metric(torch.tensor(predicted_labels), torch.tensor(true_labels))
    fig_size = [7, 7]
    fig1 = plt.figure("test", fig_size)
    ax1 = fig1.gca()
    ax1.cla()

    counter_true = Counter(true_labels)
    counter_max_true = max(counter_true, key=counter_true.get)
    dupli_max_true = counter_true[counter_max_true]

    ax = sns.heatmap(conf_matrix, linewidths=0.5, cbar=True, cmap="gist_yarg",
        linecolor="w", fmt=".2g", vmin=0, vmax=dupli_max_true, square=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    ax.set_xlabel('Vorhergesagte Klasse', fontsize=20)
    ax.set_ylabel('Tatsächliche Klasse', fontsize=20)
    ax.set_title("Verwechselungsmatrix " + model_name + "\n" + " trainiert auf " + orig_dataset +  ", angewendet auf " + inference_dateset, fontsize=13)
    plt.savefig(save_path + "Confusion_Matrix_" + model_name + orig_dataset + "_" + inference_dateset +".svg", bbox_inches="tight")
    plt.show()


def get_metrics(preds, labels, num_classes):
    """
    function to calculate accuracy, f1-score, Matthews Correlation Coefficient and Cohen Kappa

    Params:
        preds: predictions of model
        labels: true labels
        num_classes: number of classes for classification problem

    Returns:
        accuracy_score: score for accuracy
        f1_score: score for f1
        mcc_score: score for Matthews Correlation Coefficient
        ck_score: score for Cohen Kappa
    """
    accuracy_macro = MulticlassAccuracy(num_classes=num_classes, average="macro")
    f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
    mcc = MulticlassMatthewsCorrCoef(num_classes=num_classes)
    ck = MulticlassCohenKappa(num_classes=num_classes)

    accuracy_score = accuracy_macro(preds, labels)
    f1_score = f1(preds, labels)
    mcc_score = mcc(preds, labels)
    ck_score = ck(preds, labels)

    return accuracy_score, f1_score, mcc_score, ck_score

if __name__ == "__main__":
    labels = []
    preds = []
    for i in range(55):
        labels.append(i)
        preds.append(i)


    plot_conf(preds,labels,55,"test",r"I:\BA\Code\model/")
