class DummyAmp(object):
    def __init__(self, enabled=False):
        self.loss = None

    @staticmethod
    def init(enabled=False):
        return DummyAmp(enabled=enabled)

    def scale_loss(self, loss, optimizer):
        self.loss = loss
        return self

    def __enter__(self):
        return self.loss

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
