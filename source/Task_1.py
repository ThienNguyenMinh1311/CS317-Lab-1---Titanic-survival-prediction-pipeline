from metaflow import FlowSpec, step, Parameter
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

class SklearnPipelineFlow(FlowSpec):
    # Thêm các tham số để nhận đường dẫn của ba file
    train_data_path = Parameter('train_data_path', 
                            help="Path to train.csv", 
                            default="/home/workspace/workspaces/tutorials/MLops/MLops_task1/train.csv")
    test_data_path = Parameter('test_data_path', 
                            help="Path to test.csv", 
                            default="/home/workspace/workspaces/tutorials/MLops/MLops_task1/test.csv")
    gender_submission_path = Parameter('gender_submission_path', 
                            help="Path to gender_submission.csv", 
                            default="/home/workspace/workspaces/tutorials/MLops/MLops_task1/gender_submission.csv")

    @step
    def start(self):
        print("🚀 Khởi động pipeline ML với sklearn + Metaflow!")
        self.next(self.load_data)

    @step
    def load_data(self):
        print("Begin step 1")
        # Load dữ liệu từ ba file
        self.train_df = pd.read_csv(self.train_data_path)
        self.test_df = pd.read_csv(self.test_data_path)
        self.gender_submission_df = pd.read_csv(self.gender_submission_path)

        # Làm sạch dữ liệu trong train.csv
        self.train_df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)
        self.X_train = self.train_df.drop('Survived', axis=1)
        self.y_train = self.train_df['Survived']

        # Làm sạch dữ liệu trong test.csv
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

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer()),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer([
            ('num', num_pipeline, self.num_features),
            ('cat', cat_pipeline, self.cat_features)
        ])

        print("End step 2")
        self.next(self.train_model)

    @step
    def train_model(self):
        print("Begin step 3")
        model = RandomForestClassifier(random_state=22521391)

        full_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', model)
        ])

        param_grid = {
            'preprocessor__num__imputer__strategy': ['mean', 'median'],
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 5, 10],
        }

        self.grid_search = GridSearchCV(
            estimator=full_pipeline,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )

        print("🔍 Đang thực hiện GridSearchCV...")
        self.grid_search.fit(self.X_train, self.y_train)
        print(f"✅ Tốt nhất: {self.grid_search.best_params_}")
        self.best_model = self.grid_search.best_estimator_
        print("End step 3")
        self.next(self.evaluate)

    @step
    def evaluate(self):
        print("Begin step 4")
        preds = self.best_model.predict(self.X_test)
        # Lấy kết quả dự đoán từ model
        self.accuracy = accuracy_score(self.gender_submission_df['Survived'], preds)
        print(f"🎯 Accuracy trên tập test: {self.accuracy:.4f}")
        print("End step 4")
        self.next(self.end)

    @step
    def end(self):
        print("🎉 Pipeline hoàn tất.")
        print(f"Độ chính xác cuối cùng: {self.accuracy:.4f}")
