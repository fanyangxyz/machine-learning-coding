import math


class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


def example_basic_expression():
    # z = x * y + x
    x = Value(2.0)
    y = Value(3.0)
    z = x * y + x
    z.backward()
    print("Example 1: z = x * y + x")
    print(f"  z.data = {z.data}")   # 8.0
    print(f"  dz/dx = {x.grad}")    # y + 1 = 4.0
    print(f"  dz/dy = {y.grad}")    # x = 2.0
    print()


def example_shared_subgraph():
    # q = (x + y) * (x + 1), so x is used by two branches.
    x = Value(2.0)
    y = Value(-4.0)
    q = (x + y) * (x + 1)
    q.backward()
    print("Example 2: q = (x + y) * (x + 1)")
    print(f"  q.data = {q.data}")   # -6.0
    print(f"  dq/dx = {x.grad}")    # (x + y) + (x + 1) = 1.0
    print(f"  dq/dy = {y.grad}")    # (x + 1) = 3.0
    print()


def example_relu_gate():
    # ReLU blocks gradient when input <= 0.
    x = Value(-1.5)
    out = x.relu() * 3
    out.backward()
    print("Example 3: out = relu(x) * 3 with x = -1.5")
    print(f"  out.data = {out.data}")  # 0
    print(f"  dout/dx = {x.grad}")     # 0.0
    print()


if __name__ == "__main__":
    example_basic_expression()
    example_shared_subgraph()
    example_relu_gate()
