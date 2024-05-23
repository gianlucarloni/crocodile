"""
    Initially inspired to/adapted from: https://github.com/Alibaba-MIIL/ASL and https://github.com/zc2024/Causal_CXR 
    and then expanded with our newly introduced: CODI_ImageLevel_Loss, CODI_ImageLevel_Loss_embeddingSpace, CODI_MiniBatch_Loss, and ProjectionLayer.

    https://github.com/gianlucarloni/crocodile
"""
import torch
import torch.nn as nn
import numpy as np
import math

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps, max=1-self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps, max=1-self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w
        return -loss.sum()

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-5, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                # print("AsymmetricLossOptimized: Asymmetric Focusing - with torch.no_grad()")
                with torch.no_grad():
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(False)
                    self.xs_pos = self.xs_pos * self.targets
                    self.xs_neg = self.xs_neg * self.anti_targets
                    self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(True)
                self.loss =  self.loss * self.asymmetric_w
            else:
                # print("AsymmetricLossOptimized: Asymmetric Focusing - ELSE")
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                            self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)   
                self.loss = self.loss * self.asymmetric_w

        _loss = - self.loss.sum() / x.size(0)

        # _loss = _loss / y.size(1) * 1000
        #TODO commented out this line, why dividing for the number of classes and then scaling by 1k?
        _loss = _loss / y.size(1) * 10 #TODO 9 may, replaced the multiplication by 1000 with 10
        return _loss





#%% #TODO Define the custom loss class to compute  contrastive learning of representations where the latent representations of positive pairs and negative pairs sample within the minibatch are enforced similarity
#This is an example of our core idea, at the heart of our CROCODILE method:
# let the images be mapped by two feature extractors 'F_y' and 'F_d' to four latent representations: 'Z_y_c','Z_y_s','Z_d_c','Z_d_s'.
# Each feature extractor yields two representations ('_c','_s') and is trained on a different ground-truth label ('y', 'd').
# Within a training mini-batch of 8 samples, for instance, I would like to enforce two contraints:
#     at an "image level", I want 'Z_y_c' of an image to be aligned with its 'Z_d_s', and
#     at a "mini-batch level", I want
#         the 'Z_y_c' representations of images with same 'y' label to be aligned,
#         the 'Z_y_s' representations of images with different 'y' to be aligned, 
#         the 'Z_d_c' representations of images with same 'd' label to be aligned, and
#         the 'Z_d_s' representations of images with different 'd' label to be aligned.

class ProjectionLayer(nn.Module):
    '''
        Thanks to the attention-based feature disentanglement (Transformer: A and 1-A), we obtained separate features.
        Since the causal features of the disease branch (y_c) and the spurious features of the domain/dataset branch (d_s) have different shapes
        we need to project them to a shared embedding space with commond dimension, before computing a similarity loss such as MSELoss.
        Of course, this is true also for the other possible pairs, such as spurious features of disease branch with causal features of domain/dataset branch...

        If we were to implement that on a transformer's output level (Q and Q_bar objects), we would have had, e.g, Q=[6, 9, 2048] and Q_bar_crocodile=[6, 3, 2048], where 6 is the batch_size, 9 and 3 the classes, and 2048 the hiddem dimension of the transformer
        But we implement that on a logit level (z_ca and z_sp), so not to increase the model's complexity overhead: e.g. z_x=[6, 9] and z_c_crocodile=[6, 3]
    '''
    def __init__(self, in_features_higher, in_features_lower, common_dim, device='cuda'):
        super(ProjectionLayer, self).__init__()
        
        self.fc_higher_to_common = nn.Linear(in_features_higher, common_dim, device=device)
        self.fc_lower_to_common = nn.Linear(in_features_lower, common_dim, device=device)
    def forward(self, z_higherDimension, z_lowerDimension):
        z_higherToCommon = self.fc_higher_to_common(z_higherDimension)
        z_lowerToCommon = self.fc_lower_to_common(z_lowerDimension)
        return z_higherToCommon, z_lowerToCommon
    
class CODI_ImageLevel_Loss(nn.Module):
    '''
    This is part of the CO(ntrastive) DI(sentangled) Loss for our croCODIle setup.
    We will later leverage a combination of loss functions to enforce both image-level and mini-batch level alignments.

    Here we implement the Image-Level Loss:
    Define a similarity metric like cosine similarity between 'z_A' and 'z_B' for each image in the batch.
    You can use a loss function like Mean Squared Error (MSE) on the similarity score to encourage alignment.

    We will make use of a  projection layer to attain the same number of dimensions between the representations z_A and z_B

    Q and Q_bar objects have shape [batch_size, num_class, hiddem_dim], e.g.: [6, 9, 2048] and [6, 3, 2048]

    '''
    def __init__(self, in_features_A, in_features_B, common_dim=16):
        super(CODI_ImageLevel_Loss, self).__init__()

        self.mse = nn.MSELoss()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.projection_layer = None
        
        if in_features_A > in_features_B:
            in_features_higher=in_features_A
            in_features_lower=in_features_B
            self.projection_layer = ProjectionLayer(in_features_higher, in_features_lower, common_dim, device='cuda') # project them to the commond embedding dimension
        elif in_features_A < in_features_B:
            in_features_lower=in_features_A
            in_features_higher=in_features_B
            self.projection_layer = ProjectionLayer(in_features_higher, in_features_lower, common_dim, device='cuda') # project them to the commond embedding dimension
        else:
            print(f"CODI_ImageLevel_Loss - Skipping ProjectionLayer: there is no need to add complexity to our model, they already have the same shape: [batch_size, {in_features_A}].")                
   
    def __str__(self):
        return f"CODI_ImageLevel_Loss (with self.cosine_similarity and self.projection_layer)"

    def forward(self, z_A, z_B):    
        z_A = torch.nan_to_num(z_A)
        z_B = torch.nan_to_num(z_B)

        batch_size_A, dim_A = z_A.shape
        batch_size_B, dim_B = z_B.shape

        L_image=None

        assert batch_size_A==batch_size_B #TODO

        if dim_A != dim_B: ## This means we are computing the loss at an "Image" level: L_image
            assert self.projection_layer is not None
            # E.g., project z_x and z_c_crocodile to common space
            z_A, z_B = self.projection_layer(z_A, z_B)

            # Image-Level Loss (using cosine similarity on projected representations)
            if torch.all(torch.eq(z_A, 0)) or torch.all(torch.eq(z_B, 0)):
                # Handle case where either input is a zero vector (e.g., return a constant value)
                return 0.0  # Example: You can return a constant value in this case
            else:
                L_image = -self.cosine_similarity(z_A, z_B)
                L_image = self.mse(L_image + 1e-8, torch.zeros_like(L_image))  # Encourage positive correlation #TODO added a small epsilon value to avoid numerical instability  
        else:
            print("CODI_ImageLevel_Loss - you passed z_A and z_B with equal shape. Since their dimensions represent the radiological findings labels and the domain/dataset labels, it si very unlikely that they would be equal: this is a warning error.")
            raise ValueError
        
        L_image = torch.nan_to_num(L_image)
        return L_image  
        
            
## 
class CODI_ImageLevel_Loss_embeddingSpace(nn.Module):
    '''
    This is part of the CO(ntrastive) DI(sentangled) Loss for our croCODIle setup.
    We will later leverage a combination of loss functions to enforce both image-level and mini-batch level alignments.

    Here we implement the Image-Level Loss:
    Define a similarity metric like cosine similarity between 'z_A' and 'z_B' for each image in the batch.
    You can use a loss function like Mean Squared Error (MSE) on the similarity score to encourage alignment.

    In this version, we will NOT utilize a projection layer to attain the same number of dimensions between the representations; or, at least, not immediately.
    Indeed, our main idea here is to look at the produced Q, Q_bar, Q_crocodile, Q_bar_crocodile, and concatenate every possible combination of them into a score_matrix, inspired by https://openaccess.thecvf.com/content_cvpr_2018/papers/Sung_Learning_to_Compare_CVPR_2018_paper.pdf
    
    Q and Q_bar objects have shape [batch_size, num_class, hiddem_dim], e.g.: [6, 9, 2048] and [6, 3, 2048],
    so we CONCAT every embedding in the Q object by every embedding in the Q_bar object.
    For instance, if we have 9 radiological labels and 3 dataset labels, we will have a SCORE_MATRIX of shape (batch_size, num_class*num_class_crocodile*4, hidden_dim*2).

    We design a RelationalScorer module, in the form of a fully connected layer + Softmax, that maps each embedding of the SCORE_MATRIX to a relational_score (float value between 0.0 and 1.0)
    
    If we now imagine to assign to each of those new embedding a new column 'relation' valued to 1 if we want to impose a consistency between them, or 0 otherwise,
    we obtain a regression task for the MSELoss between the produced relational_score (float in range [0...1]) and the constructed ground-truth (0: do not relate, or 1: relate)
    
    E.g., the first element (along batch_size dimension) of the SCORE_MATRIX would represent: 
    SCORE_MATRIX[0,:,:]=
        EMBEDDINGS                                 | RELATION   |       ^
        cat(Q[0,0,:],  Q_crocodile [0,0,:])        |    0       |       |
        cat(Q[0,0,:],  Q_crocodile [0,1,:])        |    0       |       |
        cat(Q[0,0,:],  Q_crocodile [0,2,:])        |    0       |       |
        cat(Q[0,1,:],  Q_crocodile [0,0,:])        |    0       |       |
        cat(Q[0,1,:],  Q_crocodile [0,1,:])        |    0       |     these are num_class*num_class_crocodile combinations, such as 9*3=27 lines
        cat(Q[0,1,:],  Q_crocodile [0,2,:])        |    0       |       |
        ....                                                            |
        cat(Q[0,8,:],  Q_crocodile [0,0,:])        |    0       |       |
        cat(Q[0,8,:],  Q_crocodile [0,1,:])        |    0       |       |
        cat(Q[0,8,:],  Q_crocodile [0,2,:])        |    0       |       v
        cat(Q[0,0,:],  Q_bar_crocodile [0,0,:])    |    1       |       ^
        cat(Q[0,0,:],  Q_bar_crocodile [0,1,:])    |    1       |       |
        cat(Q[0,0,:],  Q_bar_crocodile [0,2,:])    |    1       |       |
        cat(Q[0,1,:],  Q_bar_crocodile [0,0,:])    |    1       |       |
        cat(Q[0,1,:],  Q_bar_crocodile [0,1,:])    |    1       |     these are, again, =num_class*num_class_crocodile combinations
        cat(Q[0,1,:],  Q_bar_crocodile [0,2,:])    |    1       |       |
        ....                                                            |
        cat(Q[0,8,:],  Q_crocodile [0,0,:])        |    1       |       |
        cat(Q[0,8,:],  Q_crocodile [0,1,:])        |    1       |       |
        cat(Q[0,8,:],  Q_crocodile [0,2,:])        |    1       |       v
        cat(Q_bar[0,0,:],  Q_crocodile [0,0,:])    |    1       |       ^
        cat(Q_bar[0,0,:],  Q_crocodile [0,1,:])    |    1       |       |
        cat(Q_bar[0,0,:],  Q_crocodile [0,2,:])    |    1       |       |
        cat(Q_bar[0,1,:],  Q_crocodile [0,0,:])    |    1       |       |
        cat(Q_bar[0,1,:],  Q_crocodile [0,1,:])    |    1       |     these are, again, =num_class*num_class_crocodile combinations
        cat(Q_bar[0,1,:],  Q_crocodile [0,2,:])    |    1       |       |
        ....                                                            |
        cat(Q_bar[0,8,:],  Q_crocodile [0,0,:])    |    1       |       |
        cat(Q_bar[0,8,:],  Q_crocodile [0,1,:])    |    1       |       |
        cat(Q_bar[0,8,:],  Q_crocodile [0,2,:])    |    1       |       v
        cat(Q_bar[0,0,:],  Q_bar_crocodile [0,0,:])|    0       |       ^
        cat(Q_bar[0,0,:],  Q_bar_crocodile [0,1,:])|    0       |       |
        cat(Q_bar[0,0,:],  Q_bar_crocodile [0,2,:])|    0       |       |
        cat(Q_bar[0,1,:],  Q_bar_crocodile [0,0,:])|    0       |       |
        cat(Q_bar[0,1,:],  Q_bar_crocodile [0,1,:])|    0       |     these are num_class*num_class_crocodile combinations, such as 9*3=27 lines
        cat(Q_bar[0,1,:],  Q_bar_crocodile [0,2,:])|    0       |       |
        ....                                                            |
        cat(Q_bar[0,8,:],  Q_bar_crocodile [0,0,:])|    0       |       |
        cat(Q_bar[0,8,:],  Q_bar_crocodile [0,1,:])|    0       |       |
        cat(Q_bar[0,8,:],  Q_bar_crocodile [0,2,:])|    0       |       v
        
    
    '''
    def __init__(self, batch_size=6, num_class=9, num_class_crocodile=3, hidden_dim=2048):
        super(CODI_ImageLevel_Loss_embeddingSpace, self).__init__()

        self.mse = nn.MSELoss()  

        self.num_class=num_class
        self.num_class_crocodile=num_class_crocodile        
        self.repetitions = self.num_class*self.num_class_crocodile*4

        self.hidden_dim=hidden_dim

        self.batch_size = batch_size

        self.RelationalScorer = nn.Sequential(
            nn.Linear(in_features = self.repetitions*self.hidden_dim*2,
                      out_features= self.repetitions,
                      device='cuda'),
            nn.Sigmoid()
        )

        self.relational_groundTruth = torch.zeros((self.batch_size,self.repetitions),device='cuda')
        self.relational_groundTruth[:,int(self.repetitions/4):int((3*self.repetitions/4-1))]=1.0 #TODO according to our policy above, we set to 1 only the pairing: Q_bar-Q_crocodile and Q-Q_bar_crocodile
        
    def __str__(self):
        return f"CODI_ImageLevel_Loss_embeddingSpace (with self.RelationalScorer: {self.repetitions*self.hidden_dim*2}-->{self.repetitions})"
        
    def forward(self, Q, Q_bar, Q_crocodile, Q_bar_crocodile):    
        Q = torch.nan_to_num(Q)
        Q_bar = torch.nan_to_num(Q_bar)
        Q_crocodile = torch.nan_to_num(Q_crocodile)
        Q_bar_crocodile = torch.nan_to_num(Q_bar_crocodile)

        batch_size = Q.shape[0]
        # , num_class_crocodile,  = Q_crocodile.shape

        # Step 1: Unsqueeze to add an extra dimension for concatenation
        Q_expanded = Q.unsqueeze(2)  # Shape: (batch_size, num_class, 1, hidden_dim)
        Q_bar_expanded = Q_bar.unsqueeze(2)  # Shape: (batch_size, num_class, 1, hidden_dim)
        Q_crocodile_expanded = Q_crocodile.unsqueeze(1)  # Shape: (batch_size, 1, num_class_crocodile, hidden_dim)
        Q_bar_crocodile_expanded = Q_bar_crocodile.unsqueeze(1)  # Shape: (batch_size, 1, num_class_crocodile, hidden_dim)

        # Step 2: Repeat tensors to match the concatenation pattern
        Q_repeated = Q_expanded.repeat(1, 1, self.num_class_crocodile, 1)  # Shape: (batch_size, num_class, num_class_crocodile, hidden_dim)
        Q_bar_repeated = Q_bar_expanded.repeat(1, 1, self.num_class_crocodile, 1)  # Same shape as Q_repeated
        Q_crocodile_repeated = Q_crocodile_expanded.repeat(1, self.num_class, 1, 1)  # Same shape as Q_repeated
        Q_bar_crocodile_repeated = Q_bar_crocodile_expanded.repeat(1, self.num_class, 1, 1)  # Same shape as Q_repeated

        # Step 3: Concatenate along the last dimension (hidden_dim)
        concat_1 = torch.cat((Q_repeated, Q_crocodile_repeated), dim=-1)  # Shape: (batch_size, num_class, num_class_crocodile, hidden_dim*2) # relational score ground truth 0
        concat_2 = torch.cat((Q_repeated, Q_bar_crocodile_repeated), dim=-1)  # Same shape as concat_1 #relational score ground truth 1
        concat_3 = torch.cat((Q_bar_repeated, Q_crocodile_repeated), dim=-1)  # Same shape as concat_1 #relational score ground truth 1
        concat_4 = torch.cat((Q_bar_repeated, Q_bar_crocodile_repeated), dim=-1)  # Same shape as concat_1 #relational score ground truth 0

        # Step 4: Concatenate all four along the second dimension (num_class*num_class_crocodile*4)
        score_matrix = torch.cat((concat_1, concat_2, concat_3, concat_4), dim=1)

        # # Step 5: Reshape to the desired shape (batch_size, num_class*num_class_crocodile*4, hidden_dim*2)
        # score_matrix = score_matrix.reshape(batch_size, num_class*num_class_crocodile*4, hidden_dim*2)
        ## Comment above, run below: actually, it is convenient to directly get the flattened version of that score_matrix: (batch_size, num_class*num_class_crocodile*4*hidden_dim*2)
        score_matrix = score_matrix.reshape(batch_size, self.num_class*self.num_class_crocodile*4*self.hidden_dim*2)
        relational_score = self.RelationalScorer(score_matrix) # (batch_size, self.num_class*self.num_class_crocodile*4)
        
        L_image = self.mse(relational_score + 1e-8, self.relational_groundTruth) #add small epsilon to combat possible instabilities
        L_image = torch.nan_to_num(L_image)
        
        return L_image         
        
class CODI_MiniBatch_Loss(nn.Module):
    '''
    This is part of the CO(ntrastive) DI(sentangled) Loss for our croCODIle setup.
    We leverage a combination of loss functions to enforce both image-level and mini-batch level alignments.

    Here, we implement the Mini-Batch Level loss term. For instance:
        Same Label Alignment: Calculate the average of 'Z_y_c' across images with the same 'y' label.
                              Use a loss function like MSE to minimize the distance between this average representation and each individual 'Z_y_c' within the same label group. Repeat for 'Z_y_s' and different labels.
        Different Label Alignment: Calculate the average of 'Z_y_c' across images with different 'y' labels.
                                Use a loss function like MSE to maximize the distance between this average representation and each individual 'Z_y_c'. Repeat for 'Z_d_c' and 'Z_d_s' with their respective labels.
    We are implementing this loss at either
    - 'activation space' (at the logit level) (z_x, z_c, z_c_cap, z_x_crocodile, z_c_crocodile, z_c_cap_crocodile), with shape, for instance, of (batch_size, num_class)
    or 'representation space' (embedding level)(Q, Q_bar, Q_crocodile, Q_bar_crocodile), with shape, for instance, of (batch_size, num_class, hidden_dim)
    '''
    def __init__(self, negative_or_positive_pair, contrastiveLossSpace = "representation"):
        super(CODI_MiniBatch_Loss, self).__init__()
        self.mse = nn.MSELoss()
        self.negative_or_positive_pair= negative_or_positive_pair #string with choices=[positive, negative]
        self.contrastiveLossSpace = contrastiveLossSpace # string with choices "activation" or "representation"       
    
    def __str__(self):
        return f"CODI_MiniBatch_Loss ({self.negative_or_positive_pair}, {self.contrastiveLossSpace})"

    def forward(self, z, labels):   
        z = torch.nan_to_num(z)
        L_mini_batch=0
        ## This means we are computing the loss at a "Mini-batch" level, considering other samples' classes: L_mini_batch
        # Mini-Batch Level Loss: we need to iterate through unique labels and calculate
        # the average representation for z within each label group.
        # Then, use MSE to enforce similarity between individual z and the group average (intra-class):            
        assert self.negative_or_positive_pair is not None
        
        for label in torch.unique(labels): # the set of possible labels seen in that batch
            
            # Get indices of images with label equal to/different from the current label
            if self.negative_or_positive_pair=="positive":
                # looking for samples with same class, being it same disease (Y branch) or domain/dataset (D branch)
                label_idx = (labels == label).nonzero(as_tuple=True)[0]
                if label_idx.size(0) == 0: return 0 # Prevent Potential Causes of NaN Values: division by zero (no items with that label are present in the batch)
                #TODO: implement prior knowledge on general-specific findings such as 'lung opacity' and 'consolidation': perfect (labels==label) and almost-perfect matches
            elif self.negative_or_positive_pair=="negative":
                # looking for samples with differing classes, being it disease (Y branch) or domain/dataset (D branch)
                label_idx = (labels != label).nonzero(as_tuple=True)[0]
                if label_idx.size(0) == 0: return 0 # Prevent Potential Causes of NaN Values: division by zero (no items with that label are present in the batch)
            
            # Average representation within the label group for z
            avg_z = torch.mean(z[label_idx], dim=0)
            avg_z = torch.nan_to_num(avg_z)

            # Intra-class similarity for z
            if self.contrastiveLossSpace == "activation":
                L_mini_batch += torch.nan_to_num(self.mse(z[label_idx] +1e-8, avg_z.unsqueeze(0).repeat(label_idx.size(0), 1)))  # TODO Add a small epsilon value before squaring the difference                 
            elif self.contrastiveLossSpace == "representation":
                L_mini_batch += torch.nan_to_num(self.mse(z[label_idx] +1e-8, avg_z.unsqueeze(0).repeat(label_idx.size(0), 1, 1))) # here, we have an additional axis for the embedding/representation (with size hidden_dim, such as 2048)                 

        return L_mini_batch
