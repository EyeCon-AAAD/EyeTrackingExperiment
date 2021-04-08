from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Module
from torch.nn import Softmax
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

class CNN(Module):
    # define model elements
    def __init__(self, depth):
        super(CNN, self).__init__()
        # to do 1.drop out 2. batch normalization 3. activation fucntion change

        # conv 16@64*64
        self.hidden1 = Conv2d(depth, 16, (3, 3), padding=1)  # in_channels = depth, out channels = 32,
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()

        # pooling 16@32*32
        self.hidden3 = MaxPool2d((2, 2), stride=(2, 2))

        # conv 32@32*32
        self.hidden4 = Conv2d(16, 32, (3, 3), padding=1)  # in_channels = depth, out channels = 32,
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        self.act4 = ReLU()

        # pooling 32@16*16
        self.hidden6 = MaxPool2d((2, 2), stride=(2, 2))

        # fully connected layer
        self.hidden10 = Linear(32 * 16 * 16, 1024)
        kaiming_uniform_(self.hidden10.weight, nonlinearity='relu')
        self.act10 = ReLU()

        # fully connected layer 2
        self.hidden11 = Linear(1024, 1024)
        kaiming_uniform_(self.hidden11.weight, nonlinearity='relu')
        self.act11 = ReLU()

        # output layer
        self.hidden12 = Linear(1024, 2)
        xavier_uniform_(self.hidden12.weight)

    # forward propagate input
    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)

        X = self.hidden3(X)

        X = self.hidden4(X)
        X = self.act4(X)

        X = self.hidden6(X)

        # flatten
        X = X.view(-1, 32 * 16 * 16)

        # fourth hidden layer
        X = self.hidden10(X)
        X = self.act10(X)
        # fifth hidden layer
        X = self.hidden11(X)
        X = self.act11(X)

        # output layer
        X = self.hidden12(X)

        return X


class CNN2(Module):
    # define model elements
    def __init__(self, depth):
        super(CNN2, self).__init__()
        # to do 1.drop out 2. batch normalization 3. activation fucntion change

        # conv 32@64*64
        self.hidden1 = Conv2d(depth, 32, (3, 3), padding=1)  # in_channels = depth, out channels = 32,
        #kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()

        # conv 32@64*64
        self.hidden2 = Conv2d(32, 32, (3, 3), padding=1)  # in_channels = depth, out channels = 32,
        #kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()

        # pooling 32@32*32
        self.hidden3 = MaxPool2d((2, 2), stride=(2, 2))

        # conv 64@32*32
        self.hidden4 = Conv2d(32, 64, (3, 3), padding=1)  # in_channels = depth, out channels = 32,
        #kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        self.act4 = ReLU()

        # conv 64@32*32
        self.hidden5 = Conv2d(64, 64, (3, 3), padding=1)  # in_channels = depth, out channels = 32,
        #kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        self.act5 = ReLU()

        # pooling 64@16*16
        self.hidden6 = MaxPool2d((2, 2), stride=(2, 2))

        # fully connected layer
        self.hidden10 = Linear(64 * 16 * 16, 1024)
        #kaiming_uniform_(self.hidden10.weight, nonlinearity='relu')
        self.act10 = ReLU()

        # fully connected layer 2
        self.hidden11 = Linear(1024, 1024)
        #kaiming_uniform_(self.hidden11.weight, nonlinearity='relu')
        self.act11 = ReLU()

        #output layer
        self.hidden12 = Linear(1024, 2)
        #xavier_uniform_(self.hidden12.weight)

    # forward propagate input
    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)

        X = self.hidden3(X)

        X = self.hidden4(X)
        X = self.act4(X)
        X = self.hidden5(X)
        X = self.act5(X)

        X = self.hidden6(X)

        # flatten
        X = X.view(-1, 64 * 16 * 16)

        # fourth hidden layer
        X = self.hidden10(X)
        X = self.act10(X)
        # fifth hidden layer
        X = self.hidden11(X)
        X = self.act11(X)

        # output layer
        X = self.hidden12(X)

        return X


class CNN3(Module):
    # define model elements
    def __init__(self, depth):
        super(CNN3, self).__init__()
        # to do 1.drop out 2. batch normalization 3. activation fucntion change

        #  32@32*32
        self.conv1 = Conv2d(depth, 32, (7, 7), padding=3)  # in_channels = depth, out channels = 32,
        self.act1 = ReLU()
        self.pool1 = MaxPool2d((2, 2), stride=(2, 2))

        #  32@16*16
        self.conv2 = Conv2d(32, 32, (5, 5), padding=2)  # in_channels = depth, out channels = 32,
        self.act2 = ReLU()
        self.pool2 = MaxPool2d((2, 2), stride=(2, 2))

        # conv 16@8*8
        self.conv3 = Conv2d(32, 16, (3, 3), padding=1)  # in_channels = depth, out channels = 32,
        self.act3 = ReLU()
        self.pool3 = MaxPool2d((2, 2), stride=(2, 2))


        # fully connected layer
        self.l1 = Linear(1024, 256)
        self.actl1 = ReLU()

        self.l2 = Linear(256, 128)
        self.actl2 = ReLU()

        self.l3 = Linear(128, 64)
        self.actl3 = ReLU()
        self.l4 = Linear(64, 32)
        self.actl4 = ReLU()
        self.l5 = Linear(32, 16)
        self.actl5 = ReLU()
        self.l6 = Linear(16, 8)
        self.actl6 = ReLU()
        self.l7 = Linear(8, 4)
        self.actl7 = ReLU()
        self.l8 = Linear(4, 2)


    # forward propagate input
    def forward(self, X):
        X = self.conv1(X)
        X = self.act1(X)
        X = self.pool1(X)

        X = self.conv2(X)
        X = self.act2(X)
        X = self.pool2(X)

        X = self.conv3(X)
        X = self.act3(X)
        X = self.pool3(X)

        # flatten
        X = X.view(-1, 1024)

        # connected layers
        X = self.l1(X)
        X = self.actl1(X)

        X = self.l2(X)
        X = self.actl2(X)

        X = self.l3(X)
        X = self.actl3(X)

        X = self.l4(X)
        X = self.actl4(X)

        X = self.l5(X)
        X = self.actl5(X)

        X = self.l6(X)
        X = self.actl6(X)

        X = self.l7(X)
        X = self.actl7(X)

        X = self.l8(X)

        return X


class CNN4(Module):
    # define model elements
    def __init__(self, depth):
        super(CNN4, self).__init__()
        # to do 1.drop out 2. batch normalization 3. activation fucntion change

        #  32@32*32
        self.conv1 = Conv2d(depth, 32, (7, 7), padding=3)  # in_channels = depth, out channels = 32,
        self.act1 = ReLU()
        self.pool1 = MaxPool2d((2, 2), stride=(2, 2))

        #  32@16*16
        self.conv2 = Conv2d(32, 32, (5, 5), padding=2)  # in_channels = depth, out channels = 32,
        self.act2 = ReLU()
        self.pool2 = MaxPool2d((2, 2), stride=(2, 2))

        # conv 16@8*8
        self.conv3 = Conv2d(32, 16, (3, 3), padding=1)  # in_channels = depth, out channels = 32,
        self.act3 = ReLU()
        self.pool3 = MaxPool2d((2, 2), stride=(2, 2))


        # fully connected layer
        self.l1 = Linear(1024, 256)
        self.actl1 = ReLU()

        self.l2 = Linear(256, 128)
        self.actl2 = ReLU()

        self.l3 = Linear(128, 64)
        self.actl3 = ReLU()
        self.l4 = Linear(64, 32)
        self.actl4 = ReLU()
        self.l5 = Linear(32, 16)
        self.actl5 = ReLU()
        self.l6 = Linear(16, 8)
        self.actl6 = ReLU()

        self.l7 = Linear(8, 4)


    # forward propagate input
    def forward(self, X):
        X = self.conv1(X)
        X = self.act1(X)
        X = self.pool1(X)

        X = self.conv2(X)
        X = self.act2(X)
        X = self.pool2(X)

        X = self.conv3(X)
        X = self.act3(X)
        X = self.pool3(X)

        # flatten
        X = X.view(-1, 1024)

        # connected layers
        X = self.l1(X)
        X = self.actl1(X)

        X = self.l2(X)
        X = self.actl2(X)

        X = self.l3(X)
        X = self.actl3(X)

        X = self.l4(X)
        X = self.actl4(X)

        X = self.l5(X)
        X = self.actl5(X)

        X = self.l6(X)
        X = self.actl6(X)

        X = self.l7(X)
        #X = Softmax(X)

        return X



class CNN5(Module):
    # define model elements
    def __init__(self, depth):
        super(CNN5, self).__init__()
        # to do 1.drop out 2. batch normalization 3. activation fucntion change

        #  32@32*32
        self.conv1 = Conv2d(depth, 32, (7, 7), padding=3)  # in_channels = depth, out channels = 32,
        self.act1 = ReLU()
        self.pool1 = MaxPool2d((2, 2), stride=(2, 2))

        #  32@16*16
        self.conv2 = Conv2d(32, 32, (5, 5), padding=2)  # in_channels = depth, out channels = 32,
        self.act2 = ReLU()
        self.pool2 = MaxPool2d((2, 2), stride=(2, 2))

        # conv 16@8*8
        self.conv3 = Conv2d(32, 16, (3, 3), padding=1)  # in_channels = depth, out channels = 32,
        self.act3 = ReLU()
        self.pool3 = MaxPool2d((2, 2), stride=(2, 2))

        # fully connected layer
        self.l1 = Linear(1024, 256)
        self.actl1 = ReLU()

        self.l2 = Linear(256, 128)
        self.actl2 = ReLU()

        self.l3 = Linear(128, 64)
        self.actl3 = ReLU()
        self.l4 = Linear(64, 32)
        self.actl4 = ReLU()
        self.l5 = Linear(32, 16)


    # forward propagate input
    def forward(self, X):
        X = self.conv1(X)
        X = self.act1(X)
        X = self.pool1(X)

        X = self.conv2(X)
        X = self.act2(X)
        X = self.pool2(X)

        X = self.conv3(X)
        X = self.act3(X)
        X = self.pool3(X)

        # flatten
        X = X.view(-1, 1024)

        # connected layers
        X = self.l1(X)
        X = self.actl1(X)

        X = self.l2(X)
        X = self.actl2(X)

        X = self.l3(X)
        X = self.actl3(X)

        X = self.l4(X)
        X = self.actl4(X)

        X = self.l5(X)
        return X

