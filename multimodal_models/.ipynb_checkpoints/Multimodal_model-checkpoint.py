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


class AttentionBottleneck(nn.Module):
    """
    A simple attention bottleneck module.
    It first projects the input embeddings to a lower-dimensional bottleneck space,
    then applies multi-head self-attention (here we use a single head for simplicity),
    and finally returns the attended bottleneck features.
    
    Assumes input x is of shape: [sequence_length, batch_size, embed_dim]
    """
    def __init__(self, embed_dim, bottleneck_dim, num_heads=1):
        super(AttentionBottleneck, self).__init__()
        self.projection = nn.Linear(embed_dim, bottleneck_dim)
        # Use PyTorch's built-in multi-head attention.
        self.attn = nn.MultiheadAttention(embed_dim=bottleneck_dim, num_heads=num_heads)
    
    def forward(self, x):
        # Project the modality embeddings to bottleneck space.
        # x shape: [seq_len, batch, embed_dim] -> [seq_len, batch, bottleneck_dim]
        proj_x = self.projection(x)
        # Apply self-attention within the bottleneck space.
        attn_output, attn_weights = self.attn(proj_x, proj_x, proj_x)
        return attn_output  # shape: [seq_len, batch, bottleneck_dim]

class MultimodalModel_Attn(nn.Module):
    """
    A simple multimodal fusion model.
    The model takes two modalities (for example, WSI and omics features) with different input dimensions,
    embeds each into a common high-dimensional space, stacks them to form a sequence, fuses the sequence
    via an attention bottleneck, and finally performs classification.
    
    In this example, we assume:
      - WSI_input: [batch_size, WSI_dim]
      - omics_input: [batch_size, transcriptome_dim]
    """
    def __init__(self, WSI_dim, transcriptome_dim, n_classes=2, embed_dim=256, bottleneck_dim=128):
        super(MultimodalModel_Attn, self).__init__()
        # Separate embedding layers for each modality.
        self.WSI_embed = nn.Linear(WSI_dim, embed_dim)
        self.omics_embed = nn.Linear(transcriptome_dim, embed_dim)

        # Define the attention bottleneck module.
        self.attn_bottleneck = AttentionBottleneck(embed_dim, bottleneck_dim, num_heads=1)
        
        # Classification head after fusion.
        self.classifier = nn.Linear(bottleneck_dim, n_classes)
    
    def forward(self, WSI_input, omics_input):
        # Embed the WSI and omics inputs.
        # Resulting shapes: [batch_size, embed_dim]
        WSI_emb = F.relu(self.WSI_embed(WSI_input))
        omics_emb = F.relu(self.omics_embed(omics_input))
        
        # Stack the modality embeddings along the "sequence" dimension.
        # New shape: [num_modalities, batch_size, embed_dim]
        # Here, num_modalities = 2 (WSI and omics)
        fused = torch.stack([WSI_emb, omics_emb], dim=0)
        
        # Pass through the attention bottleneck to fuse the modalities.
        # The attention module will be applied across the modality (sequence) dimension.
        bottleneck_output = self.attn_bottleneck(fused)  # shape: [2, batch_size, bottleneck_dim]
        
        # Aggregate the outputs; for example, take the mean over the modality dimension.
        fused_features = bottleneck_output.mean(dim=0)  # shape: [batch_size, bottleneck_dim]
        
        # Apply the classifier
        logits = self.classifier(fused_features)  # shape: [batch_size, n_classes]
        
        Y_prob=F.softmax(logits, dim=-1)
        Y_hat = torch.topk(Y_prob, 1, dim = -1)[1]
        
        return logits, Y_prob, Y_hat, fused_features
        # return logits

# # Attention with ResNet

# class Attn_Net(nn.Module):

#     def __init__(self, L = 64, D = 16, dropout = False, n_classes = 1):
#         super(Attn_Net, self).__init__()
#         self.module = [
#             nn.Linear(L, D),
#             nn.Tanh()]
#         if dropout:
#             self.module.append(nn.Dropout(0.25))
#         self.module.append(nn.Linear(D, n_classes))
#         self.module = nn.Sequential(*self.module)

#     def forward(self, x):
#         return self.module(x), x # N x n_classes

# class Attn_Net_Gated(nn.Module):
#     def __init__(self, L = 2, D = 2, dropout = False, n_classes = 1):
#         super(Attn_Net_Gated, self).__init__()
#         self.attention_a = [
#             nn.Linear(L, D),
#             nn.Tanh()]
        
#         self.attention_b = [nn.Linear(L, D),
#                             nn.Sigmoid()]
#         if dropout:
#             self.attention_a.append(nn.Dropout(0.25))
#             self.attention_b.append(nn.Dropout(0.25))

#         self.attention_a = nn.Sequential(*self.attention_a)
#         self.attention_b = nn.Sequential(*self.attention_b)
#         self.attention_c = nn.Linear(D, n_classes)

#     def forward(self, x):
#         a = self.attention_a(x)
#         b = self.attention_b(x)
#         A = a.mul(b)
#         A = self.attention_c(A)  # N x n_classes
#         return A, x

# class MultimodalModel_Attn(nn.Module):
#     def __init__(self, gate = True, dropout = 0., WSI_dim=512, transcriptome_dim=92, n_classes=2, embed_dim=64):        
#         super().__init__()
#         # adjust embed_dim size
#         if embed_dim > transcriptome_dim:
#             embed_dim=transcriptome_dim

#         self.WSI_embedder=nn.Sequential(
#             nn.Linear(WSI_dim, embed_dim),
#             nn.ReLU(),
#         )
#         self.transcriptome_embedder=nn.Sequential(
#             nn.Linear(transcriptome_dim, embed_dim),
#             nn.ReLU(),
#         )
        
        
#         fc = [nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout)]
#         if gate:
#             attention_net = Attn_Net_Gated(L = embed_dim, D = embed_dim, dropout = dropout, n_classes = 1)
#         else:
#             attention_net = Attn_Net(L = embed_dim, D = embed_dim, dropout = dropout, n_classes = 1)
#         fc.append(attention_net)
#         self.attention_net = nn.Sequential(*fc)
        
#         self.common_encoder=nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.ReLU(),
#             nn.Linear(embed_dim, embed_dim),
#             nn.ReLU(),
            
            
#         )
#         self.classifiers = nn.Linear(embed_dim, n_classes)

    
#     @staticmethod
#     def create_positive_targets(length, device):
#         return torch.full((length, ), 1, device=device).long()
    
#     @staticmethod
#     def create_negative_targets(length, device):
#         return torch.full((length, ), 0, device=device).long()

#     def forward(self, wsi_feature, transcriptome_feature, instance_eval=False, return_features=False):

#         WSI_h=self.WSI_embedder(wsi_feature)
#         transcriptome_h=self.transcriptome_embedder(transcriptome_feature)
#         merged_h=WSI_h+transcriptome_h
        
#         merged_combination_feature=self.common_encoder(merged_h)
        
#         h=torch.stack((WSI_h, transcriptome_h),1)

#         A, h = self.attention_net(h)  # NxK      


#         A = torch.transpose(A, 2, 1)  # KxN
#         A = F.softmax(A, dim=1)  # softmax over N
        
#         M = torch.matmul(A, h)

#         merged_combination_feature=merged_combination_feature.unsqueeze(dim=1)
    
#         M=M+merged_combination_feature

#         logits = self.classifiers(M)
#         logits=logits.squeeze(dim=1)
        
#         Y_hat = torch.topk(logits, 1, dim = 1)[1]
#         Y_prob = F.softmax(logits, dim = 1)
#         if instance_eval:
#             results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
#             'inst_preds': np.array(all_preds)}
#         else:
#             results_dict = {}
#         if return_features:
#             results_dict.update({'features': M})
#         return logits, Y_prob, Y_hat, results_dict




