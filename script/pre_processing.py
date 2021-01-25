
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np

class pre:
    
    def __init__(self,df: pd.DataFrame):
        '''
        

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe ready for pre-processing.


        '''
        
        self.df = df
    
    
    def upsampling(self) -> pd.DataFrame:
        '''
        This function removes firstly features that are not very relevant
        (using correlation matrix and decision tree's best features).
         
         Secondly, it doubles the number of cases for the minority class (target=1) 
         Finally, it downsamples the majority class (target=0) to reach the same 
         level as the other class

        Returns
        -------
        data_upsample : Pandas dataframe
            Returns a dataframe with equal numbers of rows for each class.

        '''
        
        #Removing features with low importance in our model
        data = self.df.drop(["ID","SEX","EDUCATION","MARRIAGE","AGE","PAY_4","PAY_5","PAY_6"], axis=1)
        
        #Select all data of the minority class 
        data_default = data.loc[data["default.payment.next.month"]==1]
        
        #Doucle this data to upsample and reach a level of 13272 rows with the target class=1
        data_default = pd.concat([data_default, data_default])
        
        #Randomly pick the same number of rows of the majority class      
        data_not_default = data.loc[data["default.payment.next.month"]==0].sample(n=len(data_default))
        
        #Reaching a upsampled dataset of 26.544 data
        data_upsample = pd.concat([data_default, data_not_default])
        
        #Sorting randomly and reindexing our dataset
        
        data_upsample = data_upsample.sample(frac=1).reset_index(drop=True)
        
        return data_upsample
        
    
    def standardize(self) -> np.ndarray: 
        '''
        

        Returns
        -------
        X_train : numpy array
            Training features.
        X_test : numpy array
            Testing features.
        y_train : numpy array
            Training targets.
        y_test : numpy array
            Testing targets.
        X_data : numpy array
            Entire features dataset.
        Y_data : numpy array
            Entire targets dataset.

        '''
        
        #Generate upsampled dataset
        data_upsample = self.upsampling()
        
        #Seperate features and targets
        X_data = data_upsample.drop("default.payment.next.month", axis=1)
        Y_data = data_upsample["default.payment.next.month"]
        
        #Dividing in train-test sets
        X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
        
        #Standardize each set
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        X= sc.fit_transform(X_data)
        
        return X_train, X_test, y_train, y_test, X_data, Y_data
        
        
        
        
        
        
    
