import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class MultimodalModel_MLP(nn.Module):
    def __init__(self, WSI_dim=512, transcriptome_dim=500, n_classes=2):
        super(MultimodalModel_MLP, self).__init__()

        self.common_encoder = nn.Sequential(
            nn.Linear(WSI_dim+transcriptome_dim, WSI_dim+transcriptome_dim),
            nn.ELU(),        
            # nn.AlphaDropout(p=0.25), # add dropout to reduce overfitting
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(WSI_dim+transcriptome_dim, n_classes),
        )
        
    def forward(self,WSI_vec, transcriptome_vec):
        o1=WSI_vec
        o2=transcriptome_vec
        o12=torch.cat((o1, o2), dim=1)
        feature = self.common_encoder(o12)
        logits=self.classifier(feature)
        Y_prob=F.softmax(logits, dim=-1)
        Y_hat = torch.topk(Y_prob, 1, dim = -1)[1]
        
        return logits, Y_prob, Y_hat, feature



