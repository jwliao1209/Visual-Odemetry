
class CameraPose:
    def __init__(self, R, t):
        self.R = R
        self.t = t
    
    def update(self, R, t):
        self.t += self.R @ t
        self.R = self.R @ R
        return

    def get(self):
        return self.R, self.t
