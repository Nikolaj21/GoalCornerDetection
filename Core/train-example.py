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



#We define the training as a function so we can easily re-use it.
def train_thomas(model, optimizer, train_loader, val_loader, num_epochs=1, modelNr=1, dataAug=False):
    out_dict = {'test_loss': [],
              'train_loss': [],
              'train_mean_loss': [],
              'test_mean_loss': []}
    model.train()
    for epoch in range(num_epochs):        
        #For each epoch
        start = time.time()
        train_correct = 0
        train_loss = []
        for minibatch_no, (data, boxes, labels) in enumerate(train_loader): 
            # model.train()  
            # make the data fit faster-rcnn
            targets = []
            for i in range(len(data)):
                d = {}
                d['boxes'] = boxes[i]
                d['labels'] = torch.tensor([labels[i]])
                targets.append(d)

            # move to gpu
            data = data.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Forward pass your image through the network
            loss_dict = model(data, targets)
            #Compute the loss
            losses = sum(loss for loss in loss_dict.values())
            #Backward pass through the network
            losses.backward()
            #Update the weights
            optimizer.step()
            train_loss.append(losses.item())
            out_dict['train_loss'].append(losses.item())

            torch.cuda.empty_cache()

        #Comput the test accuracy
        test_loss = []
        test_correct = 0
        for data, boxes, labels in val_loader:
            # model.train()
            targets = []
            for i in range(len(data)):
                d = {}
                d['boxes'] = boxes[i]
                d['labels'] = torch.tensor([labels[i]])
                targets.append(d)

            data = data.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                loss_dict = model(data, targets)

            #Compute the loss
            losses = sum(loss for loss in loss_dict.values())
            test_loss.append(losses.item())
            out_dict['test_loss'].append(losses.item())
            torch.cuda.empty_cache()

        torch.save(model.state_dict(), f"../work/outputs/{DATASET}/model_v{modelNr}/Faster-RCNN_v{modelNr}epoch{epoch + 1}{'_dataaug' if dataAug else '' }.pt")
        
        out_dict['train_mean_loss'].append(np.mean(train_loss))
        out_dict['test_mean_loss'].append(np.mean(test_loss))
        print(f"Epoch: {epoch + 1} took {((time.time() - start) / 60):.3f} min \t",
            f"Loss train: {np.mean(train_loss):.3f}\t",
              f"Loss val: {np.mean(test_loss):.3f}\t")

    return out_dict