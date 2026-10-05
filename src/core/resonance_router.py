import torch, torch.nn as nn

class ResonanceRouter(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps, self.means, self.inv_covs = eps, nn.ParameterDict(), nn.ParameterDict()
        
    @torch.no_grad()
    def calibrate(self, tid, acts):
        mu = acts.mean(0)
        cov = torch.cov(acts.T) + (self.eps * torch.eye(acts.size(1), device=acts.device))
        self.means[tid] = nn.Parameter(mu, requires_grad=False)
        self.inv_covs[tid] = nn.Parameter(torch.linalg.inv(cov), requires_grad=False)
        
    def forward(self, x):
        keys = list(self.means.keys())
        ds = [torch.sqrt(torch.sum((x-self.means[k])@self.inv_covs[k]*(x-self.means[k]), 1).clamp(min=1e-6)).unsqueeze(1) for k in keys]
        return [keys[i] for i in torch.argmin(torch.cat(ds, 1), 1)]
