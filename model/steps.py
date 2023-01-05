import torch.nn as nn
from tqdm import tqdm
from eval_modul import *


def train_step(model, dl, device, logger, optimizer, i, e):
    losses = []
    model.train()
    with tqdm(dl, unit="batch") as tbatch:
        for x, y in tbatch:
            x, y = x.to(device), y.to(device)
            # Forward
            pred, _ = model(x, e)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(pred, y.to(torch.float))
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            logger.add_scalar("loss_training_step", loss, i)

            losses.append(loss.item())

            tbatch.set_postfix(loss=loss.item())
            i+=1

    return sum(losses)/len(losses), i


def val_step(model, dl, device, e):
    num_batches = len(dl)
    model.eval()
    val_loss = 0
    labels_list, predictions_list, logits_list = [], [], []
    with torch.no_grad():
        for x, y in dl:

            #print("is nan value in data tensor:" + str(torch.isnan(x).any()))
            #print("is nan value in label tensor:" + str(torch.isnan(y).any()))
            x, y = x.to(device), y.to(device)
            pred, _ = model(x, e)

            prediction = torch.argmax(pred, dim=1)
            label = torch.argmax(y, dim=1)

            # list for accuracy computation
            labels_list = labels_list + label.cpu().detach().numpy().tolist()
            predictions_list = predictions_list + prediction.cpu().detach().numpy().tolist()
            logits_list = logits_list + pred.cpu().detach().numpy().tolist()

            loss_fn = nn.CrossEntropyLoss()
            val_loss += loss_fn(pred, y.to(torch.float))


    val_loss /= num_batches
    print(f"Val Error CEL: ", val_loss)

    return val_loss, predictions_list, labels_list, logits_list


def test_step(model, dl, device, e):
    num_batches = len(dl)
    model.eval()
    test_loss, correct = 0, 0
    labels_list, predictions_list, logits_list = [], [], []
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            pred, _ = model(x, e)

            prediction = torch.argmax(pred, dim=1)
            label = torch.argmax(y, dim=1)
            # list for accuracy computation
            labels_list = labels_list + label.cpu().detach().numpy().tolist()
            predictions_list = predictions_list + prediction.cpu().detach().numpy().tolist()
            logits_list = logits_list + pred.cpu().detach().numpy().tolist()

            loss_fn = nn.CrossEntropyLoss()
            test_loss += loss_fn(pred, y.to(torch.float))

    test_loss /= num_batches
    print(f"Test Error CEL: ", test_loss)

    return test_loss, predictions_list, labels_list, logits_list


def inference_step(model, dl, device, e=None):
    model.eval()
    test_loss, correct = 0, 0
    labels_list, predictions_list, logits_list = [], [], []
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            pred, _ = model(x, e)

            prediction = torch.argmax(pred, dim=1)
            label = torch.argmax(y, dim=1)
            # list for accuracy computation
            labels_list = labels_list + label.cpu().detach().numpy().tolist()
            predictions_list = predictions_list + prediction.cpu().detach().numpy().tolist()
            logits_list = logits_list + pred.cpu().detach().numpy().tolist()

    return test_loss, predictions_list, labels_list, logits_list


def sgb_step(model, dl, device, e=None):
    """
    inference function to map sockets from where data was measured to neuron of network
    sockets in dataloader are the number of the socket from where data was measured
    x in dataloader is the measured data from corresponding socket

    Params:
        model: loaded model used for inference
        dl: used dataloader, should be inference dataloader
        device: device to run model on

    Returns:
        socket_labels: classification of data as output neuron activation e.g. neuron 1 or 0
        output_sockets: corresponding socket number to output neuron activation
    """
    output_sockets, socket_labels = [], []
    model.eval()
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for x, sockets in dl:
            x = x.to(device)
            preds, _ = model(x, e)
            pred_prob = softmax(preds)
            pred = torch.argmax(pred_prob, dim=1)
            socket_labels.append(sockets)
            output_sockets.append(pred)

    return torch.cat(socket_labels).cpu().detach().numpy().tolist(), torch.cat(output_sockets).cpu().detach().numpy().tolist()

if __name__ == "__main__":
   print("Hello")
