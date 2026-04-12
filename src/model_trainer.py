

from src.utils import setup_logging
import os
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from src.utils import save_object, evaluate_models


logger = setup_logging()


class ModelTrainer:

    def __init__(self):
        pass

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "XGBoost": XGBClassifier(eval_metric='mlogloss', tree_method='hist'),

        }

        params = {
            "Logistic Regression": {
                "C": [1, 5, 10]
            },
            "Random Forest": {
                "n_estimators": [50, 100]
            },
            "Decision Tree": {
                "max_depth": [5, 10, None]
            },

            "KNN": {
                "n_neighbors": [3, 5, 7]
            },
            "XGBoost": {
                "n_estimators": [50, 100],
                "learning_rate": [0.1]
            }
        }

        model_report = {}

        #  Train & evaluate each model
        for name, model in models.items():
            logger.info(f"Training {name}")

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            model_report[name] = acc

            logger.info(f"{name} Accuracy: {acc}")

        #  Select best model
        best_model_name = max(model_report, key=model_report.get)
        best_model = models[best_model_name]
        best_score = model_report[best_model_name]

        logger.info(
            f"Best Model: {best_model_name} with accuracy {best_score}")

        # Save best model
        # with open("best_model.pkl", "wb") as f:
        #     pickle.dump(best_model, f)

        return best_model, best_score, model_report
