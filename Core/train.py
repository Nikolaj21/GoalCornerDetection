from tqdm import tqdm
import torch


def train(model, optimizer, loss_fun, trainloader, testloader, epochs=10,show_every=10):
    train_len = len(trainloader.dataset)
    test_len = len(testloader.dataset)
    # define function for training the model
    for epoch in tqdm(range(epochs),total=epochs):
        model.train()
        # initialize for calculating accuracy and loss
        train_correct = 0
        train_loss = 0
        # run through the data in batches
        for batch,labels in trainloader:
            # forward pass through network
            logits = model(*batch)
            # set gradients to zero so as not to accumulate gradients across batches
            optimizer.zero_grad()
            # calculate loss
            loss = loss_fun(logits,labels)
            train_loss += loss
            # backward pass through network
            loss.backward()
            # update the weights
            optimizer.step()
            # get probabilities for each class
            classprobs = model.softmax(logits)
            # calculate number of correct predictions in batch
            predicted = classprobs.detach().argmax(dim=1)
            train_correct += (labels == predicted).sum()#.cpu().item()
        if (epoch+1)%show_every==0 or (epoch+1)==epochs:
            # calculate accuracy of training epoch
            train_accuracy = train_correct / train_len
            avg_train_loss = train_loss / train_len
            tqdm.write(f'\nEpoch {epoch+1}')
            tqdm.write(f'Train accuracy: {train_accuracy:.2f}')
            tqdm.write(f'Train loss: {avg_train_loss:.2f}')

        ### evaluate model on test set ###
        model.eval()
        # initialize for computing accuracy and loss
        test_correct = 0
        test_loss = 0
        for batch,labels in testloader:
            with torch.no_grad():
                # forward pass
                logits = model(*batch)
                loss = loss_fun(logits,labels)
            test_loss += loss
            classprobs = model.softmax(logits)
            predicted = classprobs.detach().argmax(dim=1)
            test_correct += (labels == predicted).sum()
        if (epoch+1)%show_every==0 or (epoch+1)==epochs:
            # calculate accuracy and loss of epoch
            test_accuracy = test_correct / test_len
            avg_test_loss = test_loss / test_len
            tqdm.write(f'\nEpoch {epoch+1}')
            tqdm.write(f'Test accuracy: {test_accuracy:.2f}')
            tqdm.write(f'Test loss: {avg_test_loss:.2f}')

    print('finished training successfully')
    return (train_accuracy,test_accuracy)