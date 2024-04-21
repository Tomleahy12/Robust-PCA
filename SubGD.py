import torch

class SubGD:
    def __init__(self, data_ : torch.tensor, device_, initializations = None, rank_ = 1):
        """
        Creates Sub gradient model for Robust PCA

        :params data_: data matrice
        :params rank_: rank of the target. determines number of columns u,v have 
        :params initializations: method for constructing targets u, v. see initialization for more
        :params w: target vector
        """
        self.data_ = data_.double()
        self.rank_ = rank_
        self.device_ = device_
        self.data_ = self.data_.to(self.device_)
        self.initializations = initializations
        self.w = self.initialization(initializations)
        

    def initialization(self, method): 
        """
        Initializes the vector u, v then concats into target vector w.
        
        :params methd: if random, u is initilized as Normal(0,1) else vector of ones. 

        returns: Torch.Tensor w vector
        """
        if method == "random":
            return torch.cat((torch.rand((self.data_.size()[0],self.rank_)).T,
                           torch.ones((self.data_.size()[1],self.rank_)).T), dim=1).to(self.device_).double()
        else:
            return torch.cat((torch.ones((self.data_.size()[0],self.rank_)).T,
                           torch.ones((self.data_.size()[1],self.rank_)).T), dim=1).to(self.device_).double()
        
    def SubDD_Obj(self):
        """
        Computes the objective function as defined in the model

        returns: Torch.tensor
        """
        n_rows, n_cols = self.data_.shape
        return torch.sum(
            torch.abs(
                torch.matmul(self.w[:, :n_rows].T, self.w[:, n_rows:]) - self.data_)) +\
                      self.alpha * torch.abs( 
                          torch.sum(self.w[:, :n_rows] ** 2) -\
                              torch.sum(self.w[:, n_rows:] ** 2))     
     
    def fit(self, iterations = 1000 ,  tolerance = 1e-4, mu = None, beta_ = .95, gamma_ = 1/2, alpha = 1, prints = True, iter_prints = 100):
        """
        Solver
        :params iterations: max iterations to run for
        :params tolerance: termination term. breaks loop is < tolerance
        :params mu: starting value for line search. default is 1
        :params beta_: regularizer hyper paramater for indicator function. 
        :params gamma: step solver hyper parameter. must follow a geometrically dimishing step size. 
        :params alpha: hyper paramater, defined only to be strictly positive. 
        :params prints: whether user wants progress printed.
        :params iter_prints: frequency of prints per iter. default 100
        """

        n_rows, n_cols = self.data_.shape
        self.alpha = alpha
        errors = []
        for i in range(1,iterations):
            subgradient = self.subgrad(beta=beta_)

            if mu is None:
                mu = 1
                step_size = self.step_solve_(mu, i, subgradient, gamma = gamma_)
                mu = None
            else:
                mu_store = mu
                step_size = self.step_solve_(mu, i, subgradient, gamma = gamma_)
                mu = mu_store
            
            self.w = self.w - torch.mul(step_size, subgradient)
            truth, value  = self.termination_condition(tolerance)     
            if truth:
                if prints:
                    print("**** Optimal Found, Recovery Complete")
                break
            errors.append(value)
            if prints and i % iter_prints == 0:
                print(f" error rate: {value} at iteration {i}")
               
        self.sparse = self.data_ - torch.matmul(self.w[:,:n_rows].T , self.w[:,n_rows:]) 
        
        return torch.matmul(self.w[:, :n_rows], self.w[:, :n_rows].T), self.sparse
    
    def subgrad(self, beta):
        """
        Calls sub functions, concats returns into subgradient of objective

        :param beta: indicator function tolerance for regularizer. 

        returns: Torch.tensor 
        """
        n_rows, n_cols = self.data_.shape

        return torch.cat(( self.sub_mat_signs(self.w[:, n_rows:], case = 1) + self.squared_signs(self.w[:,:n_rows]),
                           self.sub_mat_signs(self.w[:, :n_rows], case = 2) + self.squared_signs(self.w[:,n_rows:] )), dim=1)\
                        + self.grad_regularizer(beta=beta)
                    
    def sub_mat_signs(self, argument, case: int):
        """
        :param argument: pass self.w_i or self.w_j 
        :param case 1: takes the sign matrix untransposed when subgrad with respect to wi
        :param case 2: takes the sign matrix transposed when subgrad with respect to wj

        returns: Torch.tensor
        """
        n_rows, n_cols = self.data_.shape
        if case == 1:
            return torch.matmul(torch.sign( torch.matmul(self.w[:, :n_rows].T,
                                                                self.w[:, n_rows:]) - self.data_), argument.T).T
        else:
            return torch.matmul(torch.sign( torch.matmul(self.w[:, :n_rows].T,
                                                                self.w[:, n_rows:]) - self.data_).T, argument.T).T
        
    def squared_signs(self, argument):
        """
        computes sign{wi**2 - wj**2} @ wi or wj

        :param argument: pass self.w_i or self.w_j 

        returns Torch.Tensor 
        """
        n_rows, n_cols = self.data_.shape
        return 2 * self.alpha *  torch.matmul(self.custom_sign(torch.matmul(self.w[:, :n_rows], self.w[:, :n_rows].T)
                                                                - torch.matmul(self.w[:, n_rows:],self.w[:, n_rows:].T)), argument)
    
    def step_solve_(self, mu_0, k, sub_gradient, gamma):
        """
        Step Size solver subroutine, solves s.t. self.w > 0 

        :param  mu_0: starting point for the search
        :param k: current iteration the solver is on. 
        :param subgradient: sub gradient of the objective function
        :param gamma: scaling hyperparameter.
        
        returns Scalar step_size
        """
        mu_k = mu_0
        while True:
            if (self.w - mu_k * sub_gradient > 0).all():
                break
            else:
                mu_k *= torch.tensor(mu_0 * gamma, dtype=torch.float64)
                    
        return mu_k
  
    def regularizer(self, beta = .95):
        """
        Computes the regularization term of the objective function
        :param beta: cut off param for indicator function. default = .95

        returns: Torch.tensor regularizer
        """
        n_rows, n_cols = self.data_.shape
        lambda_asym = (n_rows + n_cols) / 2
        reg_func = lambda x, coef, beta:  coef * (x - beta)**4  
        regularizer = torch.sum(torch.where(self.w >= beta, reg_func(self.w, lambda_asym, beta), 0))
        return regularizer

    def grad_regularizer(self, beta = .95):
        """
        Computes the regularization term of the subgradient
        :param beta: cut off param for indicator function. default = .95

        returns: Torch.tensor gradient_regularizer
        """
        n_rows, n_cols = self.data_.shape
        lambda_asym = (n_rows + n_cols) / 2
        reg_func = lambda x, coef, beta: 4 * coef * (x - beta)**3 * x 
       
        return torch.where(torch.abs(self.w) >= beta, reg_func(self.w, lambda_asym, beta), 0)
    
    def custom_sign(self, x, epsilon=1e-5):
        """
        No longer employed in algorithm. 
        """
        return torch.where( torch.abs(x) < epsilon, 0, torch.sign(x))
    
    def termination_condition(self, tolerance): 
        """
        Checks whether to break the solver for loop

        :params tolerance: Termination tolerance for solver. 

        returns: Bool, Torch.tensor
        """
        n_rows, n_cols = self.data_.shape
        object_ = torch.matmul(self.w[:, :n_rows].T, self.w[:, n_rows:]) - self.data_
        if torch.linalg.matrix_norm(object_, "fro") / torch.linalg.matrix_norm(self.data_, "fro") < tolerance:
            value = torch.linalg.matrix_norm(object_, "fro") / torch.linalg.matrix_norm(self.data_, "fro")
            return True,value 
        else:
            value = torch.linalg.matrix_norm(object_, "fro") / torch.linalg.matrix_norm(self.data_, "fro")
            return False, value

