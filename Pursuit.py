import torch
import numpy as np

class PCP:
    def __init__(self, data_: torch.tensor, device_):
        self.data_ = data_
        self.device_ = device_
        self.data_ = self.data_.to(self.device_)
        self.S = self.initializations()
        self.Y = self.initializations()
        self.L = self.initializations()

        
    def initializations(self):
        return torch.zeros(self.data_.shape).to(self.device_)

    def fit(self, iterations=1000, tolerance = 1e-4, lambda_= None, mu_= None, prints= True, iter_prints = 100 ):

        self.mu_ = mu_ if mu_ else torch.prod(torch.tensor(self.data_.shape)) / (4 * torch.linalg.norm(self.data_, ord=1))
        self.mu_inv = 1 / self.mu_; self.mu_inv = self.mu_inv.to(self.device_)
        self.lambda_ = lambda_ if lambda_ else 1 / torch.sqrt(torch.max(torch.tensor(self.data_.shape)))
        self.lambda_ = self.lambda_.to(self.device_)
        tolerance_ = tolerance if tolerance else 1E-7 * self.frobenius_norm(self.D)
        error = torch.inf
        errors =[]
        error_same = []

        for i in range(iterations):
            
            U, s, Vt = torch.linalg.svd(self.data_ - self.S + self.mu_inv * self.Y, full_matrices = False)
            U = U.to(self.device_); s= s.to(self.device_) ; Vt = Vt.to(self.device_)
            self.L = torch.matmul(U, torch.matmul(torch.diag_embed(self.shrinkage_operator(s, self.mu_inv)), Vt))
            self.S = self.shrinkage_operator( self.data_ - self.L  + (self.mu_inv * self.Y), self.mu_inv * self.lambda_)
            self.Y = self.Y + self.mu_ * (self.data_ - self.L  - self.S)
                                              
            error = torch.linalg.norm((self.data_ - self.L - self.S),'fro')
            error_2 = torch.linalg.norm((self.data_ - self.L),'fro')/ torch.linalg.norm(self.data_)

            if prints and i % iter_prints == 0:
                print(f" error rate: {error} at iteration {i}")

            errors.append(error); error_same.append(error_2)
            if error < tolerance_: 
                print("**** Optimal Found, Recovery Complete")
                break
        return self.L, self.S, errors, error_same
    
    @staticmethod
    def shrinkage_operator(matrix, arguement):
      
      return torch.sign(matrix) * torch.maximum((torch.abs(matrix) - arguement), torch.zeros(matrix.shape, device=matrix.device))