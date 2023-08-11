
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np



def get_h_price():

    symbols = [
    "ANA.MC", "ACX.MC", "ACS.MC", "AENA.MC", "AMS.MC", "MTS.MC", "SAB.MC", "BKIA.MC", "BKT.MC", "BBVA.MC", "CABK.MC",
    "CLNX.MC", "CIE.MC", "ENG.MC", "ELE.MC", "FER.MC", "GRF.MC", "IAG.MC", "IBE.MC", "ITX.MC", "IDR.MC", "COL.MC",
    "MEL.MC", "MRL.MC", "NTGY.MC", "PHM.MC", "REE.MC", "REP.MC", "SGRE.MC", "SLR.MC", "TRE.MC", "TEF.MC", "VIS.MC",
    "VWS.CO"
    ]


    # Descargar datos
    start_date = "2016-01-01"
    end_date = "2023-01-01"


    data = yf.download(symbols, start=start_date, end=end_date)


    adj_close = data['Adj Close']


    adj_close = adj_close.drop(["BKIA.MC", "REE.MC", "SGRE.MC"  ], axis = 1)


    adj_close = adj_close.fillna(method="ffill")

    return adj_close