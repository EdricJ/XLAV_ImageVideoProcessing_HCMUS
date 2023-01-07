# abstract layer class
class BaseLayer:
    def __index__(self):
        pass

    def forward(self, in_data):
        pass

    def backward(self, out_error, rate):
        pass
