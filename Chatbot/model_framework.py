import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) # Input to hidden
        self.linear2 = nn.Linear(hidden_size, hidden_size) # Hidden to hidden
        self.linear3 = nn.Linear(hidden_size, num_classes) # Hidden to output
        self.relu = nn.ReLU() 
    
    def forward(self, input):
        hidden1 = self.linear1(input)
        hidden1 = self.relu(hidden1)
        hidden2 = self.linear2(hidden1)
        hidden2 = self.relu(hidden2)
        output = self.linear3(hidden2)
        return output