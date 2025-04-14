from metaflow import FlowSpec, step, Parameter
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import mlflow
import mlflow.sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os  

class SklearnPipelineFlow(FlowSpec):
    train_data_path = Parameter(
        'train_data_path', 
        help="Path to train.csv", 
        default='../dataset/train.csv'
    )
    test_data_path = Parameter(
        'test_data_path',
        help="Path to test.csv", 
        default='../dataset/test.csv'
    )
    gender_submission_path = Parameter(
        'gender_submission_path', 
        help="Path to gender_submission.csv", 
        default='../dataset/gender_submission.csv'
    )

    @step
    def start(self):
        print("🚀 Khởi động pipeline ML với sklearn + Metaflow + MLflow!")
        self.next(self.load_data)

    @step
    def load_data(self):
        self.train_df = pd.read_csv(self.train_data_path)
        self.test_df = pd.read_csv(self.test_data_path)
        self.gender_submission_df = pd.read_csv(self.gender_submission_path)

        self.train_df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)
        self.X_train = self.train_df.drop('Survived', axis=1)
        self.y_train = self.train_df['Survived']

        self.test_df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)
        self.X_test = self.test_df

        self.num_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        self.cat_features = ['Sex', 'Embarked']

        self.next(self.build_preprocessor)

    @step
    def build_preprocessor(self):
        # Xử lý numeric
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Xử lý categorical
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(transformers=[
            ('num', num_transformer, self.num_features),
            ('cat', cat_transformer, self.cat_features)
        ])

        self.next(self.train_model)

    @step
    def train_model(self):
        print("🛠️ Đang huấn luyện mô hình...")

        mlflow.start_run()

        mlflow.log_param("train_data_path", self.train_data_path)
        mlflow.log_param("test_data_path", self.test_data_path)

        rf_model = RandomForestClassifier(random_state=42)

        rf_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', rf_model)
        ])

        param_grid_rf = {
            'preprocessor__num__imputer__strategy': ['mean', 'median'],
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 5, 10],
        }

        grid_search = GridSearchCV(estimator=rf_pipeline, param_grid=param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
        grid_search.fit(self.X_train, self.y_train)

        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_params(self.best_params)
        mlflow.log_metric("cv_best_accuracy", self.best_score)

        joblib.dump(self.best_model, 'best_model.pkl')
        mlflow.log_artifact('best_model.pkl')
        mlflow.sklearn.log_model(self.best_model, artifact_path="sklearn_model")

        print(f"✅ Best params: {self.best_params}")
        print("✅ Cross-validated Accuracy: {:.4f}".format(self.best_score))
        self.next(self.evaluate)  

    @step
    def evaluate(self):
        preds = self.best_model.predict(self.X_test)
        true_labels = self.gender_submission_df['Survived']
        accuracy = accuracy_score(true_labels, preds)
        cm = confusion_matrix(true_labels, preds)

        mlflow.log_metric("test_accuracy", accuracy)

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        print(f"📊 Test Accuracy: {accuracy:.4f}")
        self.next(self.end)

    @step
    def end(self):
        print("🎉 Pipeline hoàn tất.")

if __name__ == '__main__':
    SklearnRandomForestFlow()

    @step
    def end(self):
        print("🎉 Pipeline hoàn tất.")
        mlflow.end_run()  # Kết thúc MLflow run

if __name__ == '__main__':
    SklearnPipelineFlow()
