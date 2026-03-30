import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        # 3 hidden layers αντί για 2
        self.l1 = nn.Linear(input_size, hidden_size) # input layer
        self.l2 = nn.Linear(hidden_size, hidden_size) # hidden layers
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, num_classes) # output layer

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        # Dropout: a regularization process to anoid overfitting
        # During training, the model randomly shuts off a percentage of nuerons (0,3 = 30% at each step)
        # It forces the network not to rely on specific neurons or key-words
        # In this way, the model learns general rules and will be able to understand the prompts that are phrased slightly different from those in intents.json

    def forward(self, x):

        # We apply dropout on the first layers, where the information is still raw
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.l2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # In the final layers, before output, we need the model to start stabilizing its decision
        # We do not apply dropout
        out = self.l3(out)
        out = self.relu(out)

        # The last layer must return unprocessing (raw) values (logits) 
        # ReLu converts all the negative numbers to 0. If we placed ReLu here, we would 'destroy' the propabilities of tags that the model considers unlikely
        # which is something the CrossEntropyLoss function needs to cprreclty calculate the error
        
        # We never apply dropout to the last layer cause we need all neurons (all potential tags) available to get the final answer
        # Dropout is a training tool, not a prediction tool
        out = self.l4(out)
        return out