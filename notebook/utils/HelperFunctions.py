import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

class HelperFunctions:
    @staticmethod
    def drawScatterPlot(**kwargs) -> None:
        """
        Draw a scatter plot.

        Required:
            - x: list or array-like (x-axis values)
            - y: list or array-like (y-axis values)

        Optional:
            - title: plot title (default: "Scatter Plot")
            - xlabel: x-axis label (default: "X-axis")
            - ylabel: y-axis label (default: "Y-axis")
        """
        x = kwargs.get('x')
        y = kwargs.get('y')

        if x is None or y is None:
            print("Invalid data for scatter plot.")
            return

        title = kwargs.get('title', 'Scatter Plot')
        xlabel = kwargs.get('xlabel', 'X-axis')
        ylabel = kwargs.get('ylabel', 'Y-axis')

        plt.scatter(x, y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    @staticmethod
    def calculateL2Distance(point1: list, point2: list) -> float:
        return math.dist(point1, point2)

    @staticmethod
    def getSortedFrequency(col: list) -> list:
        return sorted(set(col), key=col.count, reverse=True)
    
    @staticmethod
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