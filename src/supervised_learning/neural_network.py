class Dense():
    def __init__(self) -> None:
        pass

class Sequential():
    def __init__(self, layers) -> None:
        self.layers = layers

    def compile(self, loss, learning_rate):
        self.loss = loss
        self.learning_rate = learning_rate
        
    def fit(self, X, y, epoch, batch_size):
        self.epoch = epoch
        self.batch_size = batch_size

    def predict(self, X):
        pass