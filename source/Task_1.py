pip install metaflow

# from metaflow import FlowSpec, step, Parameter
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.metrics import accuracy_score
# import pandas as pd

# class SklearnPipelineFlow(FlowSpec):

#     data_path = Parameter('data_path', help="Path đến file CSV", default='titanic.csv')

#     @step
#     def start(self):
#         print("🚀 Khởi động pipeline ML với sklearn + Metaflow!")
#         self.next(self.load_data)

#     @step
#     def load_data(self):
#         df = pd.read_csv(self.data_path)
#         df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)

#         # Giả sử bài toán phân loại sống/chết
#         self.X = df.drop('Survived', axis=1)
#         self.y = df['Survived']
#         print(f"Dữ liệu: {self.X.shape[0]} mẫu, {self.X.shape[1]} đặc trưng")
#         self.next(self.split_data)

#     @step
#     def split_data(self):
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#             self.X, self.y, test_size=0.2, random_state=22521391
#         )
#         self.next(self.build_pipeline)

#     @step
#     def build_pipeline(self):
#         self.num_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
#         self.cat_features = ['Sex', 'Embarked']

#         num_pipeline = Pipeline([
#             ('imputer', SimpleImputer()),
#             ('scaler', StandardScaler())
#         ])

#         cat_pipeline = Pipeline([
#             ('imputer', SimpleImputer(strategy='most_frequent')),
#             ('encoder', OneHotEncoder(handle_unknown='ignore'))
#         ])

#         self.preprocessor = ColumnTransformer([
#             ('num', num_pipeline, self.num_features),
#             ('cat', cat_pipeline, self.cat_features)
#         ])

#         self.next(self.train_model)

#     @step
#     def train_model(self):
#         model = RandomForestClassifier(random_state=22521391)

#         full_pipeline = Pipeline([
#             ('preprocessor', self.preprocessor),
#             ('classifier', model)
#         ])

#         param_grid = {
#             'preprocessor__num__imputer__strategy': ['mean', 'median'],
#             'classifier__n_estimators': [100, 200],
#             'classifier__max_depth': [None, 5, 10],
#         }

#         self.grid_search = GridSearchCV(
#             estimator=full_pipeline,
#             param_grid=param_grid,
#             cv=5,
#             scoring='accuracy',
#             n_jobs=-1,
#             verbose=2
#         )

#         print("🔍 Đang thực hiện GridSearchCV...")
#         self.grid_search.fit(self.X_train, self.y_train)
#         print(f"✅ Tốt nhất: {self.grid_search.best_params_}")
#         self.best_model = self.grid_search.best_estimator_
#         self.next(self.evaluate)

#     @step
#     def evaluate(self):
#         preds = self.best_model.predict(self.X_test)
#         self.accuracy = accuracy_score(self.y_test, preds)
#         print(f"🎯 Accuracy trên tập test: {self.accuracy:.4f}")
#         self.next(self.end)

#     @step
#     def end(self):
#         print("🎉 Pipeline hoàn tất.")
#         print(f"Độ chính xác cuối cùng: {self.accuracy:.4f}")

# if __name__ == "__main__":
#     SklearnPipelineFlow()
