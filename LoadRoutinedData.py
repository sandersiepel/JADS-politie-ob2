import pandas as pd
import sys


class RDData:
    def __init__(self):
        self.main()

    def main(self):
        # Load pkl data file
        try:
            self.df = pd.read_pickle("data/Routined.pkl")
        except FileNotFoundError as fnf:
            print(
                "Error loading data file: make sure that you have a 'data' folder in the same directory as this script"
            )
            sys.exit(1)

        # Select only the column that are relevant.
        self.df = self.df[["ZLATITUDE", "ZLONGITUDE", "ZTIMESTAMP", "Z_PK"]]

        # Rename the columns to better names.
        self.df = self.df.rename(
            columns={
                "ZLATITUDE": "latitude",
                "ZLONGITUDE": "longitude",
                "ZTIMESTAMP": "timestamp",
                "Z_PK": "id",
            }
        )
