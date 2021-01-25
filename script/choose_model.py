
from script.model import model
import pandas as pd

class choose_model:
    
    def __init__(self, data: pd.DataFrme, bool_rf: bool, bool_dt: bool, bool_log: bool, bool_grad: bool, bool_knn: bool):
        '''
        

        Parameters
        ----------
        data : pd.DataFrme
            Data pre-processed, ready to aply our models on.
        bool_rf : bool
            Boolean to ask if we want to do a random forest classifier or not.
        bool_dt : bool
            Boolean to ask if we want to do a decision tree classifier or not..
        bool_log : bool
            Boolean to ask if we want to do a logistic regression or not..
        bool_grad : bool
            Boolean to ask if we want to do a gradient boosting classifier or not..
        bool_knn : bool
            Boolean to ask if we want to do a knn or not..


        '''
        self.data = data
        self.bool_rf = bool_rf
        self.bool_dt = bool_dt
        self.bool_log = bool_log
        self.bool_grad = bool_grad
        self.bool_knn = bool_knn
        
        self.scores = model(self.data)
        
    def bool_model(self):
        '''
        
        This function runs the appropriate models based on the boolean values entered by the user.

        '''
        if self.bool_rf:
            
            self.scores.random_forest()
            
        if self.bool_dt:
            
            self.scores.decision_tree()
            
        if self.bool_log:
            
            self.scores.logistic_reg()
            
        if self.bool_grad:
            
            self.scores.gradient_boost()
            
        if self.bool_knn:
            
            self.scores.knn()
            
            
            
        
            
        
        
        