from metaflow import FlowSpec, step, Parameter
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib  

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
        self.next(self.check_versions)

    @step
    def check_versions(self):
        import sys
        import pandas as pd
        import sklearn
        import numpy as np
        import metaflow
        import mlflow
        import joblib  

        print("📦 Phiên bản thư viện đang sử dụng:")
        print(f"Python: {sys.version}")
        print(f"pandas: {pd.__version__}")
        print(f"scikit-learn: {sklearn.__version__}")
        print(f"numpy: {np.__version__}")
        print(f"metaflow: {metaflow.__version__}")
        print(f"mlflow: {mlflow.__version__}")  # In version của mlflow
        print(f"joblib: {joblib.__version__}")  # In version của joblib

        self.next(self.load_data)

    @step
    def load_data(self):
        print("Begin step 1")
        self.train_df = pd.read_csv(self.train_data_path)
        self.test_df = pd.read_csv(self.test_data_path)
        self.gender_submission_df = pd.read_csv(self.gender_submission_path)

        self.train_df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)
        self.X_train = self.train_df.drop('Survived', axis=1)
        self.y_train = self.train_df['Survived']

        self.test_df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)
        self.X_test = self.test_df
        print(f"Dữ liệu train: {self.X_train.shape[0]} mẫu, {self.X_train.shape[1]} đặc trưng")
        print("End step 1")
        self.next(self.build_pipeline)

    @step
    def build_pipeline(self):
        print("Begin step 2")
        self.num_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        self.cat_features = ['Sex', 'Embarked']

        num_pipeline = Pipeline([('imputer', SimpleImputer()), ('scaler', StandardScaler())])
        cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])

        self.preprocessor = ColumnTransformer([('num', num_pipeline, self.num_features), ('cat', cat_pipeline, self.cat_features)])
        print("End step 2")
        self.next(self.train_model)

    @step
    def train_model(self):
        print("Begin step 3")

        # Đảm bảo khởi tạo MLflow
        mlflow.start_run()

        # Log dataset thông tin
        mlflow.log_param("train_data_path", self.train_data_path)
        mlflow.log_param("test_data_path", self.test_data_path)
        mlflow.log_param("gender_submission_path", self.gender_submission_path)

        # Định nghĩa mô hình RandomForest và KNN
        rf_model = RandomForestClassifier(random_state=22521391)
        knn_model = KNeighborsClassifier()

        # Tạo pipeline đầy đủ với preprocessor và classifier cho RandomForest
        rf_pipeline = Pipeline([('preprocessor', self.preprocessor), ('classifier', rf_model)])

        # Tạo pipeline đầy đủ với preprocessor và classifier cho KNN
        knn_pipeline = Pipeline([('preprocessor', self.preprocessor), ('classifier', knn_model)])

        # Grid search cho Random Forest
        param_grid_rf = {
            'preprocessor__num__imputer__strategy': ['mean', 'median'],
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 5, 10],
        }

        # Grid search cho KNN
        param_grid_knn = {
            'preprocessor__num__imputer__strategy': ['mean', 'median'],
            'classifier__n_neighbors': [3, 5, 7],
            'classifier__weights': ['uniform', 'distance'],
        }

        # Log tên mô hình
        mlflow.log_param("model", "RandomForest and KNN")

        # Thực hiện grid search cho Random Forest
        mlflow.log_param("model_type", "RandomForest")
        self.grid_search_rf = GridSearchCV(estimator=rf_pipeline, param_grid=param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
        self.grid_search_rf.fit(self.X_train, self.y_train)
        print(f"✅ Random Forest - Tốt nhất: {self.grid_search_rf.best_params_}")

        # Thực hiện grid search cho KNN
        mlflow.log_param("model_type", "KNN")
        self.grid_search_knn = GridSearchCV(estimator=knn_pipeline, param_grid=param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
        self.grid_search_knn.fit(self.X_train, self.y_train)
        print(f"✅ KNN - Tốt nhất: {self.grid_search_knn.best_params_}")

        # Log kết quả
        self.best_model_rf = self.grid_search_rf.best_estimator_
        self.best_model_knn = self.grid_search_knn.best_estimator_

        # Lưu mô hình và log lại đường dẫn
        joblib.dump(self.best_model_rf, 'best_rf_model.pkl')
        joblib.dump(self.best_model_knn, 'best_knn_model.pkl')

        # Lưu mô hình vào MLflow
        mlflow.log_artifact('best_rf_model.pkl')
        mlflow.log_artifact('best_knn_model.pkl')

        mlflow.log_param("RandomForest_best_params", self.grid_search_rf.best_params_)
        mlflow.log_param("KNN_best_params", self.grid_search_knn.best_params_)
        mlflow.log_param("RandomForest_best_score", self.grid_search_rf.best_score_)
        mlflow.log_param("KNN_best_score", self.grid_search_knn.best_score_)

        print("End step 3")
        self.next(self.evaluate)

    @step
    def evaluate(self):
        print("Begin step 4")
        # Đánh giá mô hình Random Forest và KNN
        rf_preds = self.best_model_rf.predict(self.X_test)
        knn_preds = self.best_model_knn.predict(self.X_test)

        rf_accuracy = accuracy_score(self.gender_submission_df['Survived'], rf_preds)
        knn_accuracy = accuracy_score(self.gender_submission_df['Survived'], knn_preds)

        print(f"🎯 Accuracy trên tập test với RandomForest: {rf_accuracy:.4f}")
        print(f"🎯 Accuracy trên tập test với KNN: {knn_accuracy:.4f}")

        # Log accuracy vào MLflow
        mlflow.log_metric("RandomForest_Accuracy", rf_accuracy)
        mlflow.log_metric("KNN_Accuracy", knn_accuracy)

        print("End step 4")
        self.next(self.end)

    @step
    def end(self):
        print("🎉 Pipeline hoàn tất.")
        mlflow.end_run()  # Kết thúc MLflow run

if __name__ == '__main__':
    SklearnPipelineFlow()
