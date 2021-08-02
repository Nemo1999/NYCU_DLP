from dataclasses import dataclass
import numpy as np

@dataclass(init=True, eq=False, order=False, unsafe_hash=False, frozen=False)
class Var:
    # static variable, defining the gradient tape we are currently in 
    currentTape = None
    """
    This class is a thin wrapper around numpy array,
    which implement automatic differentiation of 
    1. add  (with broadcasting along single axis)
    2. minus (with broadcasting along single axis)
    3. matmul (and matapp)
    4. element-wise activation (need to provide derivative)
    5. transpose

    Attributes:
    -----------
    value: numpy.ndarray
        the cached value of current node
        The value is  valid only if the `compute` method of self or ancester node is invoked
        constant Var node has valid value by default
    parents: list[ tupple[ Func [numpy.ndarray , numpy.ndarray ] , Var]]
        The computation graph is stored using the `parents` attribute of each node
    shape: 
        The shape of current value
    grad:
        gradient of current node
    gradValid: 
        `True` if current grad is valid
    tape:
        The gradient tape that Var belongs to

    Methods:
    --------
    compute():
        When the value of input Node changes, the value of parent nodes are updated lazily.
        The computation beneath a particular Var node is only executed if the `compute` method is invoked,
         the `value` of current node is updated.
    findGrad():
        Find gradient of current node with respect to a particular **output node**, whose `gradValid` is set to True.
        And whose self.grad is set to corresponding mask with ones selecting the target values.
    

    """


    value : np.ndarray
    def __post_init__(self):
        assert isinstance(self.value, np.ndarray), "value of Var should be numpy ndarray"
        self.shape=self.value.shape
        assert len(self.shape) == 2, "we only accept arrays with exactly two dimensions"
        #initialize gradient to zero
        self.grad=np.zeros(self.shape)
        self.gradValid = False
        #parents in the computational graph
        self.parents=[]
        self.compute = lambda : self.value
        if Var.currentTape:
            Var.currentTape.addVar(self)

    def findGrad(self):
        #traverse the paths 
        if self.gradValid: 
            return  self.grad  
        else:
            self.grad = np.zeros(self.shape)
            for local_grad, var in self.parents:
                self.grad += local_grad(var.findGrad())
            self.gradValid = True
            return self.grad

    def __add__(self, other):
        assert isinstance(other,self.__class__), f"operands must have type {self.__class__}"
        # implement broadcasting behaviour
        ans = Var(self.value + other.value)
        def c(): 
            ans.value = self.compute() + other.compute()
            return ans.value
        ans.compute = c
        
        if self.shape == other.shape:
            self.parents.append((lambda x:x,ans))
            other.parents.append((lambda x:x,ans))
        elif other.shape[0] == 1 and self.shape[1] == other.shape[1]:
            self.parents.append((lambda da:da,ans))
            other.parents.append((lambda da: np.sum(da,axis=0, keepdims=True), ans))
        elif other.shape[1] == 1 and self.shape[0] == other.shape[0]:
            self.parents.append((lambda da:da,ans))
            other.parents.append((lambda da: np.sum(da,axis=1, keepdims=True), ans))
        elif self.shape[0] == 1 and self.shape[1] == other.shape[1]:
            self.parents.append((lambda da: np.sum(da,axis=0, keepdims=True), ans))
            other.parents.append((lambda da:da,ans))
        elif self.shape[1] == 1 and self.shape[0] == other.shape[0]:
            self.parents.append((lambda da: np.sum(da,axis=1, keepdims=True), ans))
            other.parents.append((lambda da:da,ans))
        elif self.shape == (1,1):
            self.parents.append((lambda da: np.sum(da,keepdims=True),ans))
            other.parents.append((lambda da:da,ans))
        elif other.shape == (1,1):
            other.parents.append((lambda da: np.sum(da,keepdims=True),ans))
            self.parents.append((lambda da:da,ans))
        else:
            assert Fasle, f"Cannot broadcast for input shape {self.shape}, {other.shape}"
        return ans

    def __mul__(self, other):
        assert isinstance(other,self.__class__), f"operands must have type {self.__class__}"
        # implement broadcasting behaviour
        ans = Var(self.value * other.value)
        def c(): 
            ans.value = self.compute() * other.compute()
            return ans.value
        ans.compute = c
        
        if self.shape == other.shape:
            self.parents.append((lambda da:da*other.value,ans))
            other.parents.append((lambda da:da*self.value,ans))
        elif other.shape[0] == 1 and self.shape[1] == other.shape[1]:
            self.parents.append((lambda da:da*other.value,ans))
            other.parents.append((lambda da: np.sum(da*self.value,axis=0, keepdims=True), ans))
        elif other.shape[1] == 1 and self.shape[0] == other.shape[0]:
            self.parents.append((lambda da:da*other.value,ans))
            other.parents.append((lambda da: np.sum(da*self.value,axis=1, keepdims=True), ans))
        elif self.shape[0] == 1 and self.shape[1] == other.shape[1]:
            self.parents.append((lambda da: np.sum(da*other.value,axis=0, keepdims=True), ans))
            other.parents.append((lambda da:da*self.value,ans))
        elif self.shape[1] == 1 and self.shape[0] == other.shape[0]:
            self.parents.append((lambda da: np.sum(da*other.value,axis=1, keepdims=True), ans))
            other.parents.append((lambda da:da*self.value,ans))
        elif self.shape == (1,1):
            self.parents.append((lambda da: np.sum(da*other.value,keepdims=True),ans))
            other.parents.append((lambda da:da*self.value,ans))
        elif other.shape == (1,1):
            other.parents.append((lambda da: np.sum(da*self.value,keepdims=True),ans))
            self.parents.append((lambda da:da*other.value,ans))
        else:
            assert Fasle, f"Cannot broadcast for input shape {self.shape}, {other.shape}"
        return ans
    
    def __sub__(self, other):
        assert isinstance(other,self.__class__), f"operands must have type {self.__class__}"
        ans = self + (- other)
        return ans

    def __neg__(self):
        ans = Var(-self.value)
        def c():
            ans.value = -self.compute()
            return ans.value
        ans.compute = c

        self.parents.append((lambda x: -x, ans))
        return ans

    def __pos__(self):
        return self
    
    def __matmul__(self,other):
        assert isinstance(other,self.__class__), f"operands must have type {self.__class__}"
        assert self.shape[1] == other.shape[0], f"operands shape {self.shape}, {other.shape} not suitable for matmul"
        ans = Var(self.value @ other.value)
        def c():
            ans.value = self.compute() @ other.compute()
            return ans.value
        ans.compute = c
        
        # assume self.shape = (m,n), other.shape = (n,k), da.shape = (m,k)
        self.parents.append((lambda da: da @ other.value.T, ans)) 
        other.parents.append((lambda da: self.value.T @ da, ans))   
        return ans
    
    def activate(self,func):
        f = func['forward']
        df = func['backward']
        ans = Var(f(self.value))
        def c():
            ans.value = f(self.compute())
            return ans.value
        ans.compute = c 

        self.parents.append((lambda da:da * df(self.value),ans))
        return ans
    @property
    def T(self):
        ans = Var(self.value.T)
        def c():
            ans.value = self.compute().T
            return ans.value
        ans.compute = c
        self.parents.append((lambda da: da.T, ans))
        return ans

# define activation funcitons

class Activatoins:
    Relu = {
        'forward':lambda x: x * (x>0),
        'backward': lambda x: (x>0).astype(np.float64) 
    }

    Sigmoid = {
        'forward': lambda x: 1.0/(1.0 + np.exp(-x)),
        'backward': lambda x: np.exp(-x) / ((1.0 + np.exp(-x))**2)
    }

    Tanh = {
        'forward': lambda x: np.tanh(x),
        'backward': lambda x: 1 - (np.tanh(x) ** 2)
    }


class GradTape: 
    """
        manage a computation graph
        In side the with statement, the new Variables are added to the GradTape

        Attributes:
        ----------- 


    """
    def __init__(self):
        self.Vars = set()
        self.current_top = None
    def __enter__(self):
        self.Vars = set()
        Var.currentTape=self
        return self
    def addVar(self,v):
        self.Vars.add(v)
        v.tape = self
    def forward(self, top_var):
        top_var.compute()
        self.current_top = None
        return top_var.value
    def backward(self, top_var):
        for v in self.Vars:
            v.gradValid = False
        top_var.gradValid = True
        top_var.grad = np.ones(shape=top_var.shape)
        for v in self.Vars:
            v.findGrad()
    def findGrad(self,var):
        return var.findGrad()
    def __exit__(self,a,b,c):
        Var.currentTape=None
