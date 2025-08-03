# Model Documentation: RandomForestClassifier

## Specifications
- **n_estimators**: `100`
- **max_depth**: `6`
- **min_samples_leaf**: `5`
- **max_features**: `sqrt`
- **class_weight**: `balanced`
- **random_state**: `42`

## Expected Features
- onehotencoder__MultipleLines_Yes
- onehotencoder__InternetService_DSL
- onehotencoder__InternetService_Fibra Ótica
- onehotencoder__InternetService_No
- onehotencoder__OnlineSecurity_Yes
- onehotencoder__OnlineBackup_Yes
- onehotencoder__TechSupport_Yes
- onehotencoder__StreamingTV_Yes
- onehotencoder__StreamingMovies_Yes
- onehotencoder__Contract_Anual
- onehotencoder__Contract_Bianual
- onehotencoder__Contract_Mensal
- onehotencoder__PaymentMethod_Electronic check
- remainder__SeniorCitizen
- remainder__Partner
- remainder__Dependents
- remainder__Tenure
- remainder__PaperlessBilling
- remainder__Monthly

## Sample Input (X.head())
```
column_name | 1st row value | dtype
----------------------------------------
onehotencoder__MultipleLines_Yes              | 0.0                  | float64
onehotencoder__InternetService_DSL            | 0.0                  | float64
onehotencoder__InternetService_Fibra Ótica    | 1.0                  | float64
onehotencoder__InternetService_No             | 0.0                  | float64
onehotencoder__OnlineSecurity_Yes             | 0.0                  | float64
onehotencoder__OnlineBackup_Yes               | 1.0                  | float64
onehotencoder__TechSupport_Yes                | 0.0                  | float64
onehotencoder__StreamingTV_Yes                | 1.0                  | float64
onehotencoder__StreamingMovies_Yes            | 1.0                  | float64
onehotencoder__Contract_Anual                 | 0.0                  | float64
onehotencoder__Contract_Bianual               | 0.0                  | float64
onehotencoder__Contract_Mensal                | 1.0                  | float64
onehotencoder__PaymentMethod_Electronic check | 1.0                  | float64
remainder__SeniorCitizen                      | 0.0                  | float64
remainder__Partner                            | 0.0                  | float64
remainder__Dependents                         | 0.0                  | float64
remainder__Tenure                             | 0.4305555555555555   | float64
remainder__PaperlessBilling                   | 1.0                  | float64
remainder__Monthly                            | 0.7587064676616915   | float64
```

## Sample Output (y.head())
```
 y_i | model.predict() | model.predict_proba() | dtype
----------------------------------------------------------------------
 y0  |       1.0       |      [0.28,0.72]      | float64
 y1  |       0.0       |      [0.80,0.20]      | float64
 y2  |       1.0       |      [0.37,0.63]      | float64
 y3  |       1.0       |      [0.41,0.59]      | float64
 y4  |       1.0       |      [0.20,0.80]      | float64
```
