import json
import pandas as pd
import sys


class GMData:
    def __init__(self):
        self.main()

    def main(self):
        self.data = self.load_data()
        self.process_data()

    def load_data(self):
        try:
            with open("data/Records.json") as json_file:
                data = json.load(json_file)
        except FileNotFoundError:
            print(
                "Error loading data file: make sure that you have a 'data' folder in the same directory as this script"
            )
            sys.exit(1)

        return data

    @staticmethod
    def convertE7(latitudeE7, longitudeE7) -> tuple:
        if latitudeE7 > 900000000:
            latitudeE7 = latitudeE7 - 4294967296
        if longitudeE7 > 1800000000:
            longitudeE7 = longitudeE7 - 4294967296
        latitudeE7 /= 1e7
        longitudeE7 /= 1e7

        return (latitudeE7, longitudeE7)

    def process_data(self) -> None:
        data_processed = []
        for d in self.data["locations"]:
            try:
                converted = self.convertE7(d["latitudeE7"], d["longitudeE7"])
                data_processed.append(converted + (d["timestamp"],) + (d["source"],))
            except KeyError:
                # Sometimes the "source" attribute is missing; reason unknown.
                data_processed.append(converted + (d["timestamp"],) + ("UNKNOWN",))

        # Convert to DataFrame and change timestamp to datetime format
        self.df = pd.DataFrame(
            data_processed, columns=["latitude", "longitude", "timestamp", "source"]
        )

        # Four options:
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], format="mixed")

        self.df["timestamp"] = self.df["timestamp"].dt.tz_localize(None)
