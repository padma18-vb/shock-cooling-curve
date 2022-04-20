import pandas as pd

class preprocess:
    def __init__(self, file):
        self.file = file
        self.df = pd.read_csv(self.file)
        self.reduce()

    def reduce(self):
        return
