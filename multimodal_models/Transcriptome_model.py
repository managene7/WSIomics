
import torch
import torch.nn as nn
import torch.nn.functional as F


class TranscriptomeModel(nn.Module): #  mlp layers
    def __init__(self, tdim=92, dropout=0.2, mul_node=1, n_classes=2):
        super().__init__()
        self.input=nn.Sequential(
            nn.Linear(tdim,512),
            nn.ELU(),
            nn.AlphaDropout(p=dropout), # add dropout to reduce overfitting
        )
        self.hidden=nn.Sequential(

            nn.Linear(512,512),
            nn.ELU(),
        )
        self.classifier=nn.Sequential(
            nn.Linear(512, n_classes),
        )
        
    def forward(self,x):
        first_layer_feature=self.input(x)
        hidden_feature=self.hidden(first_layer_feature)
        
        feature=first_layer_feature+hidden_feature
        feature=self.hidden(feature)
        logits=self.classifier(feature)
        Y_hat = torch.topk(logits, 1, dim = -1)[1]
        Y_prob=F.softmax(logits, dim=-1)

        return logits, Y_prob, Y_hat, feature