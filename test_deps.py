import autogen          # pyautogen
import requests
import sklearn          # scikit-learn
from sklearn.metrics import cohen_kappa_score
import pingouin as pg
import pandas as pd
import matplotlib.pyplot as plt
#import openai           # comment out if not using, want local LLMs instead of paid APIs 

print("All core libraries imported successfully!")
print("AutoGen version:", autogen.__version__)

# Quick test of Kappa
y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 1, 0, 0]
kappa = cohen_kappa_score(y_true, y_pred)
print("Sample Cohen's Kappa:", round(kappa, 2))
