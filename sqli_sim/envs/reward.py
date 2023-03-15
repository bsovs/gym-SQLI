class Reward:
    def __init__(self, capture=10, escape=-1, rows=-1, error=1, wrong=-1):
        self.capture = capture
        self.error = error
        self.escape = escape
        self.rows = rows
        self.wrong = wrong
