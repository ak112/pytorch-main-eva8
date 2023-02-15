import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Trainer:

    def __init__(self, model, device, criterion, optimizer, batch_size, scheduler=None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.train_losses = []
        self.train_acc = []
        self.test_losses = []
        self.test_acc = []
        self.train_loss_avg = []
        self.train_acc_avg = []
        self.misclassified_imgs = {}


    def get_train_stats(self):
        return list(map(lambda x: x.cpu().item(), self.train_loss_avg)), self.train_acc_avg
    
    def get_test_stats(self):
        return list(map(lambda x: x.cpu().item(), self.test_losses)), self.test_acc
    
    def train(self, train_loader):
        self.model.train()
        pbar = tqdm(train_loader)
        correct = 0
        processed = 0
        loss_epoch = 0
        acc_epoch = 0 

        for batch_idx, (data, target) in enumerate(pbar):

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            y_pred = self.model(data)

            loss = self.criterion(y_pred, target)
            self.train_losses.append(loss)
            loss_epoch += loss

            loss.backward()
            self.optimizer.step()


            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            pbar.set_description(desc= f' Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)
        
        loss_epoch /= len(train_loader.dataset)
        self.train_loss_avg.append(loss_epoch)
        self.train_acc_avg.append(100. * correct / len(train_loader.dataset))
        
        

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        idx = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
        self.test_acc.append(100. * correct / len(test_loader.dataset))

    
    def get_misclassified_images(self, test_loader):
        self.model.eval()
        idx = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output  = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)

                for sample in range(data.shape[0]):
                    if (target[sample] != pred[sample]):
                        self.misclassified_imgs[idx] = [data[sample].cpu(), target[sample].cpu(), pred[sample].cpu()]
                    idx += 1
        return self.misclassified_imgs


def get_criterion_for_classification():
    return nn.CrossEntropyLoss()

def get_sgd_optimizer(model, lr=0.001, momentum=0.9, scheduler=False):
    optimizer = optim.SGD(model.parameters(),
                     lr=lr,
                     momentum=momentum)
    if scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4,
                                                    gamma=0.1)
        return optimizer, scheduler
    else:
        return optimizer
    
def get_adam_optimizer(model, lr=0.001):
    return optim.Adam(model.parameters(), lr=lr)