import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf
import random

class ModelTrainer:
    def __init__(self, data_path, sheet_name):
        self.df = pd.read_excel(data_path, sheet_name)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}

    def preprocess(self):
        def find_outliers(col):
            from scipy import stats
            z = np.abs(stats.zscore(col))
            return np.where(z > 3, True, False)

        df_outliers = pd.DataFrame(self.df)
        for col in self.df.describe().columns:
            df_outliers[col] = find_outliers(self.df[col])

        outs = df_outliers.apply(lambda x: np.any(x), axis=1)
        df_clean = self.df.loc[outs == False]

        scaler = StandardScaler()
        X = df_clean.iloc[:, 0:10]
        y = df_clean.iloc[:, 10]-1
        df_scaled = scaler.fit_transform(X)

        pca = PCA(0.9)
        X_pca = pca.fit_transform(df_scaled)

        random.seed(42)
        oversample = SMOTE(sampling_strategy=0.65, random_state=42)
        X_sm, y_sm = oversample.fit_resample(X_pca, y)

        return X_sm, y_sm

    def split_dataset(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_sm, self.y_sm, test_size=0.2, random_state=42, stratify=self.y_sm
        )

    def setup(self):
        self.X_sm, self.y_sm = self.preprocess()
        self.split_dataset()
        self.y_train = self.y_train.to_numpy()

    def train_model(self, model, model_name, proba_method=None, **kwargs):
        model.fit(self.X_train, self.y_train, **kwargs)
        y_pred = model.predict(self.X_test)
        y_pred_proba = (
            getattr(model, proba_method)(self.X_test)[:, 1]
            if proba_method
            else model.decision_function(self.X_test)
        )
        self.models[model_name] = (model, y_pred, y_pred_proba)
        self.results(y_pred, y_pred_proba, model_name)

    def results(self, y_pred, y_pred_proba, model_name="Model"):
        y_pred=y_pred+1
        y_test_original = self.y_test + 1

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        self.ROC(y_test_original, y_pred_proba, model_name, ax=axes[0])
        self.conf_matrix_and_metrics(y_test_original, y_pred, model_name, ax=axes[1])
        plt.tight_layout()
        plt.show()

    @staticmethod
    def ROC(y_test, y_pred_proba, model_name, ax=None):
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba, pos_label=2)
        roc_auc = metrics.auc(fpr, tpr)
        ax.plot(fpr, tpr, color="blue", lw=2, label=f"{model_name} (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=2, label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(True)

    @staticmethod
    def conf_matrix_and_metrics(y_test, y_pred, model_name, ax=None):
        cm = metrics.confusion_matrix(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred, average="weighted")
        precision = metrics.precision_score(y_test, y_pred, average=None)
        recall = metrics.recall_score(y_test, y_pred, average=None)

        display_labels = np.unique(y_test)

        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        cm_display.plot(ax=ax, colorbar=False)
        ax.set_title(f'{model_name}')

        metric_text = (
            f"F1 Score (weighted): {f1:.2f}\n\n"
            f"Class {display_labels[0]}:\n  Precision: {precision[0]:.2f}\n  Recall: {recall[0]:.2f}\n\n"
            f"Class {display_labels[1]}:\n  Precision: {precision[1]:.2f}\n  Recall: {recall[1]:.2f}"
        )
        ax.text(1.1, 0.5, metric_text, transform=ax.transAxes, fontsize=12, va="center", ha="left")

    def train_svm(self):
        self.train_model(svm.SVC(kernel="linear", probability=True), "Support Vector Machine", proba_method="predict_proba")

    def train_log_reg(self):
        self.train_model(LogisticRegression(), "Logistic Regression", proba_method="predict_proba")

    def train_knn(self):
        parameters = {"n_neighbors": range(1, 50), "weights": ["uniform", "distance"]}
        gridsearch = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parameters, cv=5)
        gridsearch.fit(self.X_train, self.y_train)
        self.train_model(gridsearch.best_estimator_, "KNN", proba_method="predict_proba")

    def train_decision_tree(self):
        self.train_model(DecisionTreeClassifier(criterion="gini", max_depth=3,random_state=42, min_samples_leaf=5), "Decision Tree", proba_method="predict_proba")

    def train_naive_bayes(self):
        self.train_model(GaussianNB(), "Naive Bayes", proba_method="predict_proba")

    def train_neural_network(self):
        tf.random.set_seed(42)
        model = Sequential([Input(shape=(6,)), Dense(10, activation="relu"), Dense(1, activation="sigmoid")])
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        unique_classes = np.unique(self.y_train)
        class_weights = compute_class_weight("balanced", classes=unique_classes, y=self.y_train)

        model.fit(self.X_train, self.y_train, epochs=20, batch_size=10, verbose=0, class_weight=dict(enumerate(class_weights)))
        y_pred_proba = model.predict(self.X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        self.results(y_pred, y_pred_proba, "Neural Network")

    def train_boosting(self):
        self.train_model(XGBClassifier(), "Boosting Ensemble", proba_method="predict_proba")

    def train_bagging(self):
        self.train_model(BalancedBaggingClassifier(DecisionTreeClassifier(), n_estimators=10, random_state=42), "Bagging Ensemble", proba_method="predict_proba")

    def run_models(self):
        print("Running all models...")
        self.train_svm()
        self.train_log_reg()
        self.train_knn()
        self.train_decision_tree()
        self.train_naive_bayes()
        self.train_neural_network()
        self.train_boosting()
        self.train_bagging()


trainer = ModelTrainer(data_path="C:\\Users\\spano\\Desktop\\ΔΙΠΛΩΜΑΤΙΚΗ\\indian_liver_cleaned.xlsx", sheet_name="Sheet2")
trainer.setup()
trainer.run_models()
