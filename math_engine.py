from pandas import Series
from ta.momentum import RSIIndicator

class MathEngine:
    def __init__(self):
        pass

    def rsi_predict(self, data):
        rsi = RSIIndicator(Series(data), fillna=False).rsi()
        return list(rsi[-1])/100
