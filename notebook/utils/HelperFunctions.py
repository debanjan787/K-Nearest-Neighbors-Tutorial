import pandas as pd
import numpy as np
import math

class HelperFunctions:
    @staticmethod
    def calculateL2Distance(point1: list, point2: list) -> float:
        return math.dist(point1, point2)

    @staticmethod
    def getSortedFrequency(col: list) -> list:
        return sorted(set(col), key=col.count, reverse=True)
    
    def predictionClassification(df: pd.DataFrame, feature: str) -> str:
        col = df[feature].tolist()
        frequency = HelperFunctions.getSortedFrequency(col)
        return frequency[0]
    
    @staticmethod
    def predictionRegression(df: pd.DataFrame, target: str, distance: str, weight: str = False) -> float:
        if not weight:
            return df[target].mean()
        
        df_copy = df.copy()
        df_copy['weight'] = np.where(df_copy[distance] == 0, np.inf, 1 / df_copy[distance])
        
        numerator = (df_copy[target] * df_copy['weight']).sum()
        denominator = df_copy['weight'].sum()

        return numerator / denominator