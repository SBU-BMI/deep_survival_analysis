import torch
import torch.nn as nn


EPS = 1e-12


class CensoredCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CensoredCrossEntropyLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y, obs):
        # x: predicted
        # y: labels
        # obs: 1 observed event (uncensored), 0 unobserved event (right-censored)
        x = self.softmax(x).clamp(min=EPS)
        n = x.shape[0]
        loss_sum = 0
        for i in range(n):
            if obs[i] > 0.5:
                loss_sum += torch.log(x[i, y[i]])
            else:
                if y[i].item() == x.shape[1] - 1:
                    # loss_sum += x[i, y[i]] * 0.0
                    loss_sum += torch.log(x[i, y[i]].clamp(min=EPS))
                else:
                    loss_sum += torch.log(torch.sum(x[i, y[i]+1:]).clamp(min=EPS))
        loss_val = loss_sum / n * -1
        return loss_val

        
