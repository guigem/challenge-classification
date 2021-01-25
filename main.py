
from script.choose_model import choose_model
import os

import pandas as pd

basedir = os.path.abspath(os.path.dirname(__file__))


if __name__ == "__main__":
    
   data_path = os.path.join(basedir, "CSV/UCI_Credit_Card.csv")
   
   data = pd.read_csv(data_path)
   
   scores = choose_model(data, True, True, False, False, False)
   scores.bool_model()
