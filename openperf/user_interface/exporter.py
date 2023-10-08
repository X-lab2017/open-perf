import pandas as pd

class Exporter:
    def to_csv(self, data: pd.DataFrame, path: str):
        data.to_csv(path)
