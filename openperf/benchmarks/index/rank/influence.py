import pandas as pd
from io import StringIO


def run():
    data = """
    repo,Degree centrality,PageRank,OpenRank
    home-assistant/core,0.015660,0.0035,2393.86
    NixOS/nixpkgs,0.008743,0.0008,2207.5
    microsoft/vscode,0.015247,0.003,1960.39
    flutter/flutter,0.012138,0.002,1460.34
    pytorch/pytorch,0.009624,0.0012,1421.18
    MicrosoftDocs/azure-docs,0.239616,0.08,1216.01
    dotnet/runtime,0.004141,0.0006,1181.12
    microsoft/winget-pkgs,0.061954,0.0075,1106.3
    godotengine/godot,0.203330,0.045,1105.51
    odoo/odoo,0.175534,0.043,907.97
    """

    # Use StringIO to simulate a file-like object
    df = pd.read_csv(StringIO(data))

    return df
