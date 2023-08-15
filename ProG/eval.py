import torch
import numpy as np
import torchmetrics
import warnings

class Evaluator:
    def __init__(self, eval_metric='hits@50'):
        self.eval_metric= eval_metric
        if 'hits@' in self.eval_metric:
            self.K = int(self.eval_metric.split('@')[1])

    def _parse_and_check_input(self, input_dict):
        if 'hits@' in self.eval_metric:
            if not 'y_pred_pos' in input_dict:
                raise RuntimeError('Missing key of y_pred_pos')
            if not 'y_pred_neg' in input_dict:
                raise RuntimeError('Missing key of y_pred_neg')
            y_pred_pos, y_pred_neg = input_dict['y_pred_pos'], input_dict['y_pred_neg']

            if not (isinstance(y_pred_pos, np.ndarray) or (torch is not None and isinstance(y_pred_pos, torch.Tensor))):
                raise ValueError('y_pred_pos needs to be either numpy ndarray or torch tensor')
            if not (isinstance(y_pred_neg, np.ndarray) or (torch is not None and isinstance(y_pred_neg, torch.Tensor))):
                raise ValueError('y_pred_neg needs to be either numpy ndarray or torch tensor')
            
            if torch is not None and (isinstance(y_pred_pos, torch.Tensor) or isinstance(y_pred_neg, torch.Tensor)):
                if isinstance(y_pred_pos, np.ndarray):
                    y_pred_pos = torch.from_numpy(y_pred_pos)
                if isinstance(y_pred_neg, np.ndarray):
                    y_pred_neg = torch.from_numpy(y_pred_neg)
                y_pred_pos = y_pred_pos.to(y_pred_neg.device)
                type_info = 'torch'
            else:
                type_info = 'numpy'

            if not y_pred_pos.ndim == 1:
                raise RuntimeError('y_pred_pos mnust to 1-dim array, {}-dim array given'.format(y_pred_pos.ndim))
            return y_pred_pos, y_pred_neg, type_info
        
        elif 'mrr' == self.eval_metric:
            if not 'y_pred_pos' in input_dict:
                raise RuntimeError('Missing key of y_pred_pos')
            if not 'y_pred_neg' in input_dict:
                raise RuntimeError('Missing key of y_pred_neg')
            y_pred_pos, y_pred_neg = input_dict['y_pred_pos'], input_dict['y_pred_neg']

            if not (isinstance(y_pred_pos, np.ndarray) or (torch is not None and isinstance(y_pred_pos, torch.Tensor))):
                raise ValueError('y_pred_pos needs to be either numpy ndarray or torch tensor')
            if not (isinstance(y_pred_neg, np.ndarray) or (torch is not None and isinstance(y_pred_neg, torch.Tensor))):
                raise ValueError('y_pred_neg needs to be either numpy ndarray or torch tensor')
            if torch is not None and (isinstance(y_pred_pos, torch.Tensor) or isinstance(y_pred_neg, torch.Tensor)):
                if isinstance(y_pred_pos, np.ndarray):
                    y_pred_pos = torch.from_numpy(y_pred_pos)
                if isinstance(y_pred_neg, np.ndarray):
                    y_pred_neg = torch.from_numpy(y_pred_neg)
                y_pred_pos = y_pred_pos.to(y_pred_neg.device)
                type_info = 'torch'
            else:
                type_info = 'numpy'
            if not y_pred_pos.ndim == 1:
                raise RuntimeError('y_pred_pos must to 1-dim array, {}-dim array given'.format(y_pred_pos.ndim))
            if not y_pred_neg.ndim == 2:
                raise RuntimeError('y_pred_neg must to 2-dim array, {}-dim array given'.format(y_pred_neg.ndim))
            return y_pred_pos, y_pred_neg, type_info
        else:
            raise ValueError('Undefined eval metric %s' % (self.eval_metric))
    def eval(self, input_dict):
        if 'hits@' in self.eval_metric:
            y_pred_pos, y_pred_neg, type_info = self._parse_and_check_input(input_dict)
            return self._eval_hits(y_pred_pos, y_pred_neg, type_info)
        elif self.eval_metric == 'mrr':
            y_pred_pos, y_pred_neg, type_info = self._parse_and_check_input(input_dict)
            return self.eval_mrr(y_pred_pos, y_pred_neg, type_info)
        else:
            raise ValueError('Undefined eval metric %s' % (self.eval_metric))
    def _eval_hits(self, y_pred_pos, y_pred_neg, type_info):
        if type_info == 'torch':
            res = torch.topk(y_pred_neg, self.K)
            kth_score_in_negative_edges = res[0][:, -1]
            hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)

        else:
            kth_score_in_negative_edges = np.sort(y_pred_neg)[-self.K]
            hitsK = float(np.sum(y_pred_pos > kth_score_in_negative_edges)) / len(y_pred_pos)

        return {'hits@{}'.format(self.K): hitsK}
    def _eval_mrr(self, y_pred_pos, y_pred_neg, type_info):
        if type_info == 'torch':
            y_pred = torch.cat([y_pred_pos.view(-1, 1), y_pred_neg], dim = 1)
            argsort = torch.argsort(y_pred, dim = 1, descending=True)
            ranking_list = torch.nonzero(argsort == 0, as_tuple = False)
            ranking_list = ranking_list[:, 1] + 1
            mrr_list = 1. / ranking_list.to(torch.float)
            return mrr_list.mean()
        else:
            y_pred = np.concatenate([y_pred_pos.reshape(-1, 1), y_pred_neg], axis = 1)
            argsort = np.argsort(-y_pred, axis=1)
            ranking_list = (argsort == 0).nonzero()
            ranking_list = ranking_list[1] + 1
            mrr_list = 1./ranking_list.astype(np.float32)
            return mrr_list.mean()


