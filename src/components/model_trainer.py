import os,sys
import pandas as ps
import numpy as pd 
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object

from    xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from src.utils import eval_models

@dataclass
class ModelTrainerConfig:
    train_model_file_path= os.path.join("artifacts/pkls","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
                            
            logging.info("Spitting dataset into Dependent and independent features")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("Done Splitting")
            logging.info("Model Training Started !! ")
            model = {
                    "Random_Forest": RandomForestClassifier(),
                    "XGB_Classifier": XGBClassifier(),
                    "Logistic_Regression": LogisticRegression()
                    }
            
            params= {
                    "Random_Forest" : {"max_depth": [5, 8,  None, 10],
                                    "max_features": [5, 7, "auto", 8],
                                     "min_samples_split": [2, 8, 15],
                                     "n_estimators": [100, 200, 500 ],
                                    'criterion':["gini"]
                                     },

                    'XGB_Classifier' : {"learning_rate": [0.1, 0.01],
                                     "max_depth": [5, 8, 12, 20],
                                     "n_estimators": [100, 300],
                                     "colsample_bytree": [0.5, 0.8, 1, 0.3, 0.4],
                                     'n_jobs':[-1],
                                    },
            

                    "Logistic_Regression": { "class_weight":['balanced'],
                                     'penalty':['l1','l2'],
                                     'C':[0.001,0.01,0.1,1],
                                    'solver':['liblinear','saga']
                                 }
                    }
                
            #model_report:dict= eval_models(X_train,y_train,X_test,y_test,model,params)
            model_report:dict = eval_models(X_train =X_train ,X_test=X_test,y_train=y_train,y_test=y_test,
                                           models=model,params=params)
            logging.info("Returned from eval function ")
            #to get best model from our report dictionary
            best_model_score= max(sorted(model_report.values()))
            logging.info("Best model Score = {best_model_score} ")
            best_model_name= list(model.keys())[list(model_report.values()).index(best_model_score)]
            best_model= model[best_model_name]
            
            logging.info(f"Best model found, ModelName : {best_model_name}, with  Accuracy Score: {best_model_score}")
                        
            logging.info("Saving the best model in pickle file")
            save_object(file_path=self.model_trainer_config.train_model_file_path,
                          obj=best_model)
            
        except Exception as e:
            raise CustomException (e,sys)
        
