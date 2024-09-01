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

    def __call__(self, x3, x4, x5, x6, view1_feats, view2_feats, queue_feats, 
                 view1_soft_codes, view2_soft_codes, codebooks, 
                 global_step=0, writer=None):
        loss = self._simclr_loss(x3,x4,x5,x6,view1_feats, view2_feats, queue_feats=queue_feats)
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

    def _simclr_loss(self, x3,x4,x5,x6,view1_feats, view2_feats, queue_feats=None):
        cur_batch_size = view1_feats.shape[0]
        features1 = torch.cat([x3, view1_feats], dim=0)
        features2 = torch.cat([x4, view1_feats], dim=0)
        features3 = torch.cat([x5, view1_feats], dim=0)
        features4 = torch.cat([x6, view1_feats], dim=0)
        features5 = torch.cat([x3, view2_feats], dim=0)
        features6 = torch.cat([x4, view2_feats], dim=0)
        features7 = torch.cat([x5, view2_feats], dim=0)
        features8 = torch.cat([x6, view2_feats], dim=0)
        features1 = F.normalize(features1, dim=-1)
        features2 = F.normalize(features2, dim=-1)
        features3 = F.normalize(features3, dim=-1)
        features4 = F.normalize(features4, dim=-1)
        features5 = F.normalize(features5, dim=-1)
        features6 = F.normalize(features6, dim=-1)
        features7 = F.normalize(features7, dim=-1)
        features8 = F.normalize(features8, dim=-1)
        similarity_matrix1 = torch.matmul(features1, features1.T)
        similarity_matrix2 = torch.matmul(features2, features2.T)
        similarity_matrix3 = torch.matmul(features3, features3.T)
        similarity_matrix4 = torch.matmul(features4, features4.T)
        similarity_matrix5 = torch.matmul(features5, features5.T)
        similarity_matrix6 = torch.matmul(features6, features6.T)
        similarity_matrix7 = torch.matmul(features7, features7.T)
        similarity_matrix8 = torch.matmul(features8, features8.T)
        labels = torch.eye(cur_batch_size).repeat(2, 2).to(self.device)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        similarity_matrix1 = similarity_matrix1[~mask].view(similarity_matrix1.shape[0], -1)
        similarity_matrix2 = similarity_matrix2[~mask].view(similarity_matrix2.shape[0], -1)
        similarity_matrix3 = similarity_matrix3[~mask].view(similarity_matrix3.shape[0], -1)
        similarity_matrix4 = similarity_matrix4[~mask].view(similarity_matrix4.shape[0], -1)
        similarity_matrix5 = similarity_matrix5[~mask].view(similarity_matrix5.shape[0], -1)
        similarity_matrix6 = similarity_matrix6[~mask].view(similarity_matrix6.shape[0], -1)
        similarity_matrix7 = similarity_matrix7[~mask].view(similarity_matrix7.shape[0], -1)
        similarity_matrix8 = similarity_matrix8[~mask].view(similarity_matrix8.shape[0], -1)
        labels = labels[~mask].view(labels.shape[0], -1)

        pos_logits1 = similarity_matrix1[labels.bool()].view(labels.shape[0], -1)
        pos_logits2 = similarity_matrix2[labels.bool()].view(labels.shape[0], -1)
        pos_logits3 = similarity_matrix3[labels.bool()].view(labels.shape[0], -1)
        pos_logits4 = similarity_matrix4[labels.bool()].view(labels.shape[0], -1)
        pos_logits5 = similarity_matrix5[labels.bool()].view(labels.shape[0], -1)
        pos_logits6 = similarity_matrix6[labels.bool()].view(labels.shape[0], -1)
        pos_logits7 = similarity_matrix7[labels.bool()].view(labels.shape[0], -1)
        pos_logits8 = similarity_matrix8[labels.bool()].view(labels.shape[0], -1)
        neg_logits1 = similarity_matrix1[~labels.bool()].view(similarity_matrix1.shape[0], -1)
        neg_logits2 = similarity_matrix2[~labels.bool()].view(similarity_matrix2.shape[0], -1)
        neg_logits3 = similarity_matrix3[~labels.bool()].view(similarity_matrix3.shape[0], -1)
        neg_logits4 = similarity_matrix4[~labels.bool()].view(similarity_matrix4.shape[0], -1)
        neg_logits5 = similarity_matrix5[~labels.bool()].view(similarity_matrix5.shape[0], -1)
        neg_logits6 = similarity_matrix6[~labels.bool()].view(similarity_matrix6.shape[0], -1)
        neg_logits7 = similarity_matrix7[~labels.bool()].view(similarity_matrix7.shape[0], -1)
        neg_logits8 = similarity_matrix8[~labels.bool()].view(similarity_matrix8.shape[0], -1)
        if queue_feats is not None:
            queue_logits1 = torch.matmul(features1, queue_feats.T)
            queue_logits2 = torch.matmul(features2, queue_feats.T)
            queue_logits3 = torch.matmul(features3, queue_feats.T)
            queue_logits4 = torch.matmul(features4, queue_feats.T)
            queue_logits5 = torch.matmul(features5, queue_feats.T)
            queue_logits6 = torch.matmul(features6, queue_feats.T)
            queue_logits7 = torch.matmul(features7, queue_feats.T)
            queue_logits8 = torch.matmul(features8, queue_feats.T)
            neg_logits1 = torch.cat([neg_logits1, queue_logits1], dim=-1)
            neg_logits2 = torch.cat([neg_logits2, queue_logits2], dim=-1)
            neg_logits3 = torch.cat([neg_logits3, queue_logits3], dim=-1)
            neg_logits4 = torch.cat([neg_logits4, queue_logits4], dim=-1)
            neg_logits5 = torch.cat([neg_logits5, queue_logits5], dim=-1)
            neg_logits6 = torch.cat([neg_logits6, queue_logits6], dim=-1)
            neg_logits7 = torch.cat([neg_logits7, queue_logits7], dim=-1)
            neg_logits8 = torch.cat([neg_logits8, queue_logits8], dim=-1)
        pos_logits1 /= self.temperature
        pos_logits2 /= self.temperature
        pos_logits3 /= self.temperature
        pos_logits4 /= self.temperature
        pos_logits5 /= self.temperature
        pos_logits6 /= self.temperature
        pos_logits7 /= self.temperature
        pos_logits8 /= self.temperature
        neg_logits1 /= self.temperature
        neg_logits2 /= self.temperature
        neg_logits3 /= self.temperature
        neg_logits4 /= self.temperature
        neg_logits5 /= self.temperature
        neg_logits6 /= self.temperature
        neg_logits7 /= self.temperature
        neg_logits8 /= self.temperature
        pos_probs1 = pos_logits1.exp()
        pos_probs2 = pos_logits2.exp()
        pos_probs3 = pos_logits3.exp()
        pos_probs4 = pos_logits4.exp()
        pos_probs5 = pos_logits5.exp()
        pos_probs6 = pos_logits6.exp()
        pos_probs7 = pos_logits7.exp()
        pos_probs8 = pos_logits8.exp()
        neg_probs1 = neg_logits1.exp()
        neg_probs2 = neg_logits2.exp()
        neg_probs3 = neg_logits3.exp()
        neg_probs4 = neg_logits4.exp()
        neg_probs5 = neg_logits5.exp()
        neg_probs6 = neg_logits6.exp()
        neg_probs7 = neg_logits7.exp()
        neg_probs8 = neg_logits8.exp()

        if self.mode == 'debias':
            N = cur_batch_size * 2 - 2
            Ng1 = torch.clamp((-self.pos_prior * N * pos_probs1 + neg_probs1.sum(dim=-1)) / (1 - self.pos_prior),
                             min=math.exp(N * (-1 / self.temperature)))
            Ng2 = torch.clamp((-self.pos_prior * N * pos_probs2 + neg_probs2.sum(dim=-1)) / (1 - self.pos_prior),
                             min=math.exp(N * (-1 / self.temperature)))
            Ng3 = torch.clamp((-self.pos_prior * N * pos_probs3 + neg_probs3.sum(dim=-1)) / (1 - self.pos_prior),
                             min=math.exp(N * (-1 / self.temperature)))
            Ng4 = torch.clamp((-self.pos_prior * N * pos_probs4 + neg_probs4.sum(dim=-1)) / (1 - self.pos_prior),
                             min=math.exp(N * (-1 / self.temperature)))
            Ng5 = torch.clamp((-self.pos_prior * N * pos_probs5 + neg_probs5.sum(dim=-1)) / (1 - self.pos_prior),
                             min=math.exp(N * (-1 / self.temperature)))
            Ng6 = torch.clamp((-self.pos_prior * N * pos_probs6 + neg_probs6.sum(dim=-1)) / (1 - self.pos_prior),
                             min=math.exp(N * (-1 / self.temperature)))
            Ng7 = torch.clamp((-self.pos_prior * N * pos_probs7 + neg_probs7.sum(dim=-1)) / (1 - self.pos_prior),
                             min=math.exp(N * (-1 / self.temperature)))
            Ng8 = torch.clamp((-self.pos_prior * N * pos_probs8 + neg_probs8.sum(dim=-1)) / (1 - self.pos_prior),
                             min=math.exp(N * (-1 / self.temperature)))
        else:  # 'simple'
            Ng1 = neg_probs1.sum(dim=-1)
            Ng2 = neg_probs2.sum(dim=-1)
            Ng3 = neg_probs3.sum(dim=-1)
            Ng4 = neg_probs4.sum(dim=-1)
            Ng5 = neg_probs5.sum(dim=-1)
            Ng6 = neg_probs6.sum(dim=-1)
            Ng7 = neg_probs7.sum(dim=-1)
            Ng8 = neg_probs8.sum(dim=-1)
        loss1 = (- torch.log(pos_probs1 / (pos_probs1 + Ng1))).mean()
        loss2 = (- torch.log(pos_probs2 / (pos_probs2 + Ng2))).mean()
        loss3 = (- torch.log(pos_probs3 / (pos_probs3 + Ng3))).mean()
        loss4 = (- torch.log(pos_probs4 / (pos_probs4 + Ng4))).mean()
        loss5 = (- torch.log(pos_probs5 / (pos_probs5 + Ng5))).mean()
        loss6 = (- torch.log(pos_probs6 / (pos_probs6 + Ng6))).mean()
        loss7 = (- torch.log(pos_probs7 / (pos_probs7 + Ng7))).mean()
        loss8 = (- torch.log(pos_probs8 / (pos_probs8 + Ng8))).mean()
        return loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8

    def _entropy_regularization(self, soft_codes):
        return (- soft_codes * soft_codes.log()).sum(dim=-1).mean()

    def _codeword_regularization(self, codebooks):
        return torch.einsum('mkd,mjd->mkj', codebooks, codebooks).mean()

