from script.pre_processing import pre
import pandas as pd



from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier


from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score


class model:
    
    def __init__(self, data: pd.DataFrame):
        '''
        

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe ready for pre-processing.


        '''
        
        self.data = data 
        
        #Call class pre in order to preprocess our dataset
        pre_process = pre(self.data)
        self.dataset = pre_process.standardize()
        
        
    
    def random_forest(self):
        '''
        
        Create a random forest with the following criterion:
            - 300 trees in the forest
            - maximum depth of each tree is 100
            - No random state
            
        Results are evaluated by a holdout set (accuracy, precision, etc.)
        and using cross-validation with 10 folds (only accuracy).                                       


        '''
        clf_forest=RandomForestClassifier(n_estimators=300, max_depth = 100, random_state=42)

        clf_forest.fit(self.dataset[0], self.dataset[2])
        
        y_pred = clf_forest.predict(self.dataset[1])
        print("Classification matrix for random forest")
        print(classification_report(self.dataset[3],y_pred))
        
        print("Accuracy score with cross-validation (10 folds)")
        scores_randfor = cross_val_score(clf_forest, self.dataset[4], self.dataset[5], cv=10)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores_randfor.mean(), scores_randfor.std()))
        
    def decision_tree(self):
        '''
        
        Create a decision tree with the following criterion:
            - gini method to compute best feature
            - Start split at best feature
            - No random state
            
        Results are evaluated by a holdout set (accuracy, precision, etc.)
        and using cross-validation with 10 folds (only accuracy). 

        '''
        
        
        clf = DecisionTreeClassifier(criterion="gini", random_state=42, splitter="best")
        
        clf.fit(self.dataset[0], self.dataset[2])
       
        predictions = clf.predict(self.dataset[1])
        
        print("Classification matrix for decision tree")
        print(classification_report(self.dataset[3],predictions))
        
        print("Accuracy score with cross-validation (10 folds)")
        scores_tree = cross_val_score(clf, self.dataset[4], self.dataset[5], cv=10)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores_tree.mean(), scores_tree.std()))
        
        
    def logistic_reg(self):
        '''
        
        Create a logistic regression
        
        Results are evaluated by a holdout set (accuracy, precision, etc.)
        and using cross-validation with 10 folds (only accuracy). 

        '''
        model = LogisticRegression()

        model.fit(self.dataset[0], self.dataset[2])
        
        predictions_logistic = model.predict(self.dataset[1])
        
        print("Classification matrix for logistic regression")
        print(classification_report(self.dataset[3],predictions_logistic))
            
        print("Accuracy score with cross-validation (10 folds)")
        scores_log = cross_val_score(model, self.dataset[4], self.dataset[5], cv=10)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores_log.mean(), scores_log.std())) 
        
        
    def gradient_boost(self):
        '''
        
        Create a Gradient Boosting Classifier
        
        Results are evaluated by a holdout set (accuracy, precision, etc.)
        and using cross-validation with 10 folds (only accuracy). 

        '''     
        
        gbk = GradientBoostingClassifier()

        gbk.fit(self.dataset[0], self.dataset[2])
        
        gbk_pred = gbk.predict(self.dataset[1])
        
        print("Classification matrix for gradient boosting")
        print(classification_report(self.dataset[3],gbk_pred))
           
        print("Accuracy score with cross-validation (10 folds)")
        scores_gbk = cross_val_score(gbk, self.dataset[4], self.dataset[5], cv=10)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores_gbk.mean(), scores_gbk.std())) 
        
        
    def knn(self):
        '''
        
        Create a knn classifier with 10 neighbors
            
        Results are evaluated by a holdout set (accuracy, precision, etc.)
        and using cross-validation with 10 folds (only accuracy).                                       


        '''
        model_knn = KNeighborsClassifier(n_neighbors=10)

        model_knn.fit(self.dataset[0], self.dataset[2])

        knn_pred = model_knn.predict(self.dataset[1])
        
        print("Classification matrix for knn)
        print(classification_report(self.dataset[3],knn_pred))
           
        print("Accuracy score with cross-validation (10 folds)")
        scores_knn = cross_val_score(knn_pred, self.dataset[4], self.dataset[5], cv=10)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores_knn.mean(), scores_knn.std())) 
        
        
        
        
        



