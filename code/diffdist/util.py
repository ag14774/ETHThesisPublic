class AsyncOpList(object):
    def __init__(self, ops):
        self.ops = ops

    def wait(self):
        for op in self.ops:
            op.wait()

    def is_completed(self):
        for op in self.ops:
            if not op.is_completed():
                return False
        return True
