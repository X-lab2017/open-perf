import pandas as pd

class DataImporter:

    def from_csv(self, path: str) -> pd.DataFrame:
        """
        Import data from a CSV file.

        Parameters:
        - path: String path to the CSV file.

        Returns:
        - A pandas DataFrame containing the imported data.
        """
        try:
            data = pd.read_csv(path)
            return data
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return None

    def from_json(self, path: str) -> pd.DataFrame:
        """
        Import data from a JSON file.

        Parameters:
        - path: String path to the JSON file.

        Returns:
        - A pandas DataFrame containing the imported data.
        """
        try:
            data = pd.read_json(path)
            return data
        except Exception as e:
            print(f"Error reading JSON: {e}")
            return None
