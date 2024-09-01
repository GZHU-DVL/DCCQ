import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DCCQLoss:
    def __init__(self, T=1.0,
                 mode='simple',
                 pos_prior=0,
                 hp_lambda=0, 
                 hp_gamma=0,  
                 device='cpu'):
        self.device = device
        self.temperature = T
        self.mode = mode
        self.pos_prior = pos_prior

        self.hp_lambda = hp_lambda
        self.hp_gamma = hp_gamma

    def __call__(self, x1, x2, view1_feats, view2_feats, queue_feats, 
                 view1_soft_codes, view2_soft_codes, codebooks, 
                 global_step=0, writer=None):
        
        loss = self._simclr_loss(x1,x2,view1_feats, view2_feats, queue_feats=queue_feats)
        if writer is not None:
            writer.add_scalar('loss/simclr_loss', loss.item(), global_step)

        if self.hp_lambda != 0:
            entropy_reg_loss = None
            if view1_soft_codes is not None:
                entropy_reg_loss = self._entropy_regularization(view1_soft_codes)
            if view2_soft_codes is not None:
                if entropy_reg_loss is None:
                    entropy_reg_loss = self._entropy_regularization(view2_soft_codes)
                else:
                    entropy_reg_loss = (entropy_reg_loss + \
                        self._entropy_regularization(view2_soft_codes)) / 2
            if entropy_reg_loss is not None:
                loss += (self.hp_lambda * entropy_reg_loss)
                if writer is not None:
                    writer.add_scalar('loss/entropy_reg_loss', 
                                      entropy_reg_loss.item(), global_step)

        if self.hp_gamma != 0:
            codeword_reg_loss = self._codeword_regularization(codebooks)
            loss += (self.hp_gamma * codeword_reg_loss)
            if writer is not None:
                writer.add_scalar('loss/codeword_reg_loss',
                                  codeword_reg_loss.item(), global_step)

        if writer is not None:
            writer.add_scalar('loss/total_loss', loss.item(), global_step)
        return loss

    def _simclr_loss(self, x1,x2,view1_feats, view2_feats, queue_feats=None):
        cur_batch_size = view1_feats.shape[0]
        features1 = torch.cat([x1, view2_feats], dim=0)
        features2 = torch.cat([x2, view1_feats], dim=0)
        
        features1 = F.normalize(features1, dim=-1)
        features2 = F.normalize(features2, dim=-1)
        similarity_matrix1 = torch.matmul(features1, features1.T)
        similarity_matrix2 = torch.matmul(features2, features2.T)
        
        labels = torch.eye(cur_batch_size).repeat(2, 2).to(self.device)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        similarity_matrix1 = similarity_matrix1[~mask].view(similarity_matrix1.shape[0], -1)
        similarity_matrix2 = similarity_matrix2[~mask].view(similarity_matrix2.shape[0], -1)
        labels = labels[~mask].view(labels.shape[0], -1)

        pos_logits1 = similarity_matrix1[labels.bool()].view(labels.shape[0], -1)
        pos_logits2 = similarity_matrix2[labels.bool()].view(labels.shape[0], -1)
        neg_logits1 = similarity_matrix1[~labels.bool()].view(similarity_matrix1.shape[0], -1)
        neg_logits2 = similarity_matrix2[~labels.bool()].view(similarity_matrix2.shape[0], -1)
        if queue_feats is not None:
            queue_logits1 = torch.matmul(features1, queue_feats.T)
            queue_logits2 = torch.matmul(features2, queue_feats.T)
            neg_logits1 = torch.cat([neg_logits1, queue_logits1], dim=-1)
            neg_logits2 = torch.cat([neg_logits2, queue_logits2], dim=-1)
        pos_logits1 /= self.temperature
        pos_logits2 /= self.temperature
        neg_logits1 /= self.temperature
        neg_logits2 /= self.temperature
        pos_probs1 = pos_logits1.exp()
        pos_probs2 = pos_logits2.exp()
        neg_probs1 = neg_logits1.exp()
        neg_probs2 = neg_logits2.exp()

        if self.mode == 'debias':
            N = cur_batch_size * 2 - 2
            Ng1 = torch.clamp((-self.pos_prior * N * pos_probs1 + neg_probs1.sum(dim=-1)) / (1 - self.pos_prior),
                             min=math.exp(N * (-1 / self.temperature)))
            Ng2 = torch.clamp((-self.pos_prior * N * pos_probs2 + neg_probs2.sum(dim=-1)) / (1 - self.pos_prior),
                             min=math.exp(N * (-1 / self.temperature)))
        else:  
            Ng1 = neg_probs1.sum(dim=-1)
            Ng2 = neg_probs2.sum(dim=-1)
        loss1 = (- torch.log(pos_probs1 / (pos_probs1 + Ng1))).mean()
        loss2 = (- torch.log(pos_probs2 / (pos_probs2 + Ng2))).mean()
        return loss1+loss2

    def _entropy_regularization(self, soft_codes):
        return (- soft_codes * soft_codes.log()).sum(dim=-1).mean()

    def _codeword_regularization(self, codebooks):
        return torch.einsum('mkd,mjd->mkj', codebooks, codebooks).mean()

