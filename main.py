
# %% [markdown]
# # Classification on `emnist`
#
# ## 1. Create `Readme.md` to document your work
#
# Explain your choices, process, and outcomes.
#
# ## 2. Classify all symbols
#
# ### Choose a model
#
# Your choice of model! Choose wisely...
#
# ### Train away!
#
# Is do you need to tune any parameters? Is the model expecting data in a different format?
#
# ### Evaluate the model
#
# Evaluate the models on the test set, analyze the confusion matrix to see where the model performs well and where it struggles.
#
# ### Investigate subsets
#
# On which classes does the model perform well? Poorly? Evaluate again, excluding easily confused symbols (such as 'O' and '0').
#
# ### Improve performance
#
# Brainstorm for improving the performance. This could include trying different architectures, adding more layers, changing the loss function, or using data augmentation techniques.
#
# ## 2. Classify digits vs. letters model showdown
#
# Perform a full showdown classifying digits vs letters:
#
# 1. Create a column for whether each row is a digit or a letter
# 2. Choose an evaluation metric
# 3. Choose several candidate models to train
# 4. Divide data to reserve a validation set that will NOT be used in training/testing
# 5. K-fold train/test
#     1. Create train/test splits from the non-validation dataset
#     2. Train each candidate model (best practice: use the same split for all models)
#     3. Apply the model the the test split
#     4. (*Optional*) Perform hyper-parametric search
#     5. Record the model evaluation metrics
#     6. Repeat with a new train/test split
# 6. Promote winner, apply model to validation set
# 7. (*Optional*) Perform hyper-parametric search, if applicable
# 8. Report model performance

# %% [markdown]
#

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

num_classes = 62
image_size = 28

train_dataset = torchvision.datasets.MNIST(root= './data',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor(),download=True)

# %%
fig = plt.figure()
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.tight_layout()
    plt.imshow(train_dataset.train_data[i], cmap='gray', interpolation='none')
    plt.title("number/symbol: {}".format(train_dataset.train_labels[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

# %%
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
def train(epoch):
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim=1)
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()

        if batch_idx % 300 == 299:
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                  % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))
            running_loss = 0.0
            running_total = 0
            running_correct = 0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print('[%d / %d]: Accuracy on test set: %.1f %% ' % (epoch+1, EPOCH, 100 * acc))
    return acc

batch_size = 16 # 64
learning_rate = 0.001
momentum = 0.5
EPOCH = 10

if __name__ == '__main__':
    acc_list_test = []
    for epoch in range(EPOCH):
        train(epoch)
        # if epoch % 10 == 9:
        acc_test = test()
        acc_list_test.append(acc_test)

    plt.plot(acc_list_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy On TestSet')
    plt.show()
