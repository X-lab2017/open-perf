import pandas as pd
from io import StringIO


def run():
    
    data = """
    repo,participants,events,OpenActivity
    NixOS/nixpkgs,1908,33956,5163.91
    home-assistant/core,3384,18402,4380.23
    microsoft/vscode,3557,16907,3643.1
    flutter/flutter,2649,15300,2938.33
    MicrosoftDocs/azure-docs,1774,9944,2884.33
    pytorch/pytorch,2102,52636,2839.17
    odoo/odoo,1285,34123,2470.15
    dotnet/runtime,862,16641,2298.13
    godotengine/godot,2247,11204,2114.51
    microsoft/winget-pkgs,539,26600,1703.35
    """

    # Use StringIO to simulate a file-like object
    df = pd.read_csv(StringIO(data))
    
    return df
