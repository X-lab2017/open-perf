
import requests
import json
import pandas as pd

def getInfluenceData():
    # URL of the JSON file
    url = "https://xlab-open-source.oss-cn-beijing.aliyuncs.com/open_leaderboard/open_rank/company/global/202310.json"

    # Sending a GET request to fetch the data from the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON data
        data = response.json()
        # You can now work with `data`, which is a Python dictionary
    else:
        print("Error: Unable to fetch data from the URL")

    return data

def run():
    data = getInfluenceData()
    # Extracting the 'data' part of the JSON
    data = data['data']
    # Flattening the nested structure in 'item' and creating a DataFrame
    df = pd.json_normalize(data, sep='_')
    return df

if __name__ == "__main__":
    print(run())
