import torch
import torch.nn.functional as F
import torch.autograd as autograd
from torch import (tile,  
                   unsqueeze, 
                   mean, 
                   topk, 
                   sign, 
                   gather,
                   Tensor)

class LambdaRankLoss:
    def __init__(self, 
                 num_items : int = None, 
                 batch_size : int = None, 
                 sigma : float = 1.0, 
                 ndcg_at : int = 50, 
                 dtype : torch.dtype = torch.float32, 
                 lambda_normalization : bool = True,
                 pred_truncate_at : int = None, 
                 bce_grad_weight : float = 0.0, 
                 remove_batch_dim : bool = False):
        
        self.__name__ = 'lambdarank'
        self.num_items = num_items
        self.batch_size = batch_size
        self.sigma = sigma
        self.dtype = dtype
        self.bce_grad_weight = bce_grad_weight
        self.remove_batch_dim = remove_batch_dim
        self.params_truncate_at = pred_truncate_at

        self.params_ndcg_at = ndcg_at
        self.lambda_normalization = lambda_normalization
        self.less_is_better = False
        self.setup()
    
    def get_pairwise_diffs_for_vector(self, x):
        a, b = torch.meshgrid(x[:self.ndcg_at], torch.transpose(x))
        return b - a
    
    def get_pairwise_diff_batch(self, x):
        x_top_tile = tile(unsqueeze(x[:, :self.ndcg_at], 1), [1, self.pred_truncate_at, 1])
        x_tile = tile(unsqueeze(x, 2), [1, 1, self.ndcg_at])
        result = x_tile - x_top_tile
        return result

    def setup(self):
        if self.batch_size is None or self.num_items is None:
            return
        
        if self.params_truncate_at == None:
            self.pred_truncate_at = self.num_items
        else:
            self.pred_truncate_at = self.params_truncate_at

        self.ndcg_at = min(self.params_ndcg_at, self.num_items)
        self.dcg_position_discounts = 1. / torch.log2((torch.range(self.pred_truncate_at) + 2).type(self.dtype))
        self.top_position_discounts = self.dcg_position_discounts[:self.ndcg_at].view(self.ndcg_at, 1)
        self.swap_importance = torch.abs(self.get_pairwise_diffs_for_vector(self.dcg_position_discounts))
        self.batch_indices = tile(unsqueeze(torch.range(self.batch_size), 1), [1, self.pred_truncate_at]).view(self.pred_truncate_at * self.batch_size, 1)
        self.mask = (1 - F.pad(torch.ones(self.ndcg_at), (0, self.pred_truncate_at - self.ndcg_at)).view(1, self.pred_truncate_at)).type(self.dtype)
        
    def __call__(self, y_true, y_pred):
        if self.remove_batch_dim:
            y_true = y_true.view(self.batch_size, self.num_items)
            y_pred = y_pred.view(self.batch_size, self.num_items)

        result = mean(torch.abs(y_pred))

        def grad(dy):
            lambdarank_lambdas = self.get_lambdas(y_true, y_pred)
            bce_lambdas = self.get_bce_lambdas(y_true, y_pred)
            return 0 * dy, ((1 - self.bce_grad_weight) * lambdarank_lambdas + (bce_lambdas * self.bce_grad_weight)) * dy

        return result, grad

    def get_bce_lambdas(self, y_true, y_pred):

        bce_loss = F.binary_cross_entropy_with_logits(y_true, y_pred)
        logits_loss_lambdas = autograd.grad(bce_loss, (y_pred,)) / self.num_items

        return  logits_loss_lambdas

    def bce_lambdas_len(self, y_true, y_pred):
        bce_lambdas = self.get_bce_lambdas(y_true, y_pred)
        norms = torch.norm(bce_lambdas , axis=1)
        return self.bce_grad_weight * mean(norms)

    def get_lambdas(self, y_true, y_pred):
        sorted_by_score = topk(y_pred.type(self.dtype), self.pred_truncate_at)
        col_indices_reshaped = sorted_by_score.indices.view(self.pred_truncate_at * self.batch_size, 1)
        pred_ordered = sorted_by_score.values
        true_ordered = gather(y_true.type(self.dtype), sorted_by_score.indices)
        inverse_idcg = self.get_inverse_idcg(true_ordered)
        true_gains = 2 ** true_ordered - 1
        true_gains_diff = self.get_pairwise_diff_batch(true_gains)
        S = sign(true_gains_diff)
        delta_ndcg = true_gains_diff * self.swap_importance * inverse_idcg
        pairwise_diffs = self.get_pairwise_diff_batch(pred_ordered) * S

        #normalize dcg gaps - inspired by lightbm
        if self.lambda_normalization:
            best_score = pred_ordered[:, 0]
            worst_score = pred_ordered[:, -1]

            range_is_zero = torch.equal(best_score, worst_score).type(self.dtype).view(self.batch_size, 1, 1)
            norms = (1 - range_is_zero) * (torch.abs(pairwise_diffs) + 0.01) + (range_is_zero)
            delta_ndcg = torch.num_to_nan(torch.divide(delta_ndcg, norms))

        sigmoid = -self.sigma / (1 + torch.exp(self.sigma * (pairwise_diffs)))
        lambda_matrix =  delta_ndcg * sigmoid

        #calculate sum of lambdas by rows. For top items - calculate as sum by columns.
        lambda_sum_raw = torch.sum(lambda_matrix, axis=2)
        top_lambda_sum = torch.pad(-torch.sum(lambda_matrix, axis=1), (0, 0), (0, self.pred_truncate_at - self.ndcg_at))
        lambda_sum_raw_top_masked = lambda_sum_raw * self.mask
        lambda_sum_result = lambda_sum_raw_top_masked + top_lambda_sum

        if self.lambda_normalization:
            #normalize results - inspired by lightbm
            all_lambdas_sum = torch.reshape(torch.sum(torch.abs(lambda_sum_result), axis=(1)), (self.batch_size, 1))
            norm_factor = torch.num_to_nan(torch.divide(torch.log2(all_lambdas_sum + 1), all_lambdas_sum))

            lambda_sum = lambda_sum_result * norm_factor
        else:
            lambda_sum = lambda_sum_result

        indices = torch.concat([self.batch_indices, col_indices_reshaped], axis=1)
        result_lambdas = torch.zeros((self.batch_size, self.num_items)).scatter_(-1, indices, lambda_sum.view(self.pred_truncate_at * self.batch_size))
        return result_lambdas.type(torch.float32)

    def get_inverse_idcg(self, true_ordered):
        top_k_values = torch.nn.top_k(true_ordered, self.ndcg_at).values
        top_k_discounted = torch.linalg.matmul(top_k_values, self.top_position_discounts)


        return torch.num_to_nan(torch.divide(Tensor(1.0).type(self.dtype), top_k_discounted)).view(self.batch_size, 1, 1)
    
class LambdasSumWrapper:
    def __init__(self, lambdarank_loss : LambdaRankLoss):
        self.lambdarank_loss = lambdarank_loss
        self.name = "lambdarank_lambdas_len"

    def __call__(self, y_true, y_pred):
        lambdas = self.lambdarank_loss.get_lambdas(y_true, y_pred)
        return (1 - self.lambdarank_loss.bce_grad_weight) * torch.sum(torch.abs(lambdas))

class BCELambdasSumWrapper:
    def __init__(self, lambdarank_loss : LambdaRankLoss):
        self.lambdarank_loss = lambdarank_loss
        self.name = "bce_lambdas_len"

    def __call__(self, y_true, y_pred):
        lambdas = self.lambdarank_loss.get_bce_lambdas(y_true, y_pred)
        norms = torch.sum(lambdas, axis=1)
        return (self.lambdarank_loss.bce_grad_weight) * torch.sum(torch.abs(lambdas))