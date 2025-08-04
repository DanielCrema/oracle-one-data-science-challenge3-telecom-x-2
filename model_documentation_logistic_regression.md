# Model Documentation: LogisticRegression
## Specifications
- **C**: `0.01`
- **penalty**: `l2`
- **solver**: `liblinear`
- **max_iter**: `1000`
- **random_state**: `42`

## Expected Features
- onehotencoder__MultipleLines_Yes
- onehotencoder__InternetService_DSL
- onehotencoder__InternetService_Fibra Ótica
- onehotencoder__InternetService_No
- onehotencoder__OnlineSecurity_Yes
- onehotencoder__OnlineBackup_Yes
- onehotencoder__DeviceProtection_Yes
- onehotencoder__TechSupport_Yes
- onehotencoder__StreamingTV_Yes
- onehotencoder__StreamingMovies_Yes
- onehotencoder__Contract_Anual
- onehotencoder__Contract_Bianual
- onehotencoder__Contract_Mensal
- onehotencoder__PaymentMethod_Credit card (automatic)
- onehotencoder__PaymentMethod_Electronic check
- onehotencoder__PaymentMethod_Mailed check
- remainder__SeniorCitizen
- remainder__Dependents
- remainder__PhoneService
- remainder__PaperlessBilling
- remainder__log_Monthly
- remainder__log_Tenure

## Sample Input - X.head(1)

| Column Name                                          | First row value    | Dtype   |
| ---------------------------------------------------- | ------------------ | ------- |
| onehotencoder__MultipleLines_Yes                     | 0.0                | float64 |
| onehotencoder__InternetService_DSL                   | 0.0                | float64 |
| onehotencoder__InternetService_Fibra Ótica           | 1.0                | float64 |
| onehotencoder__InternetService_No                    | 0.0                | float64 |
| onehotencoder__OnlineSecurity_Yes                    | 0.0                | float64 |
| onehotencoder__OnlineBackup_Yes                      | 1.0                | float64 |
| onehotencoder__DeviceProtection_Yes                  | 0.0                | float64 |
| onehotencoder__TechSupport_Yes                       | 0.0                | float64 |
| onehotencoder__StreamingTV_Yes                       | 1.0                | float64 |
| onehotencoder__StreamingMovies_Yes                   | 1.0                | float64 |
| onehotencoder__Contract_Anual                        | 0.0                | float64 |
| onehotencoder__Contract_Bianual                      | 0.0                | float64 |
| onehotencoder__Contract_Mensal                       | 1.0                | float64 |
| onehotencoder__PaymentMethod_Credit card (automatic) | 0.0                | float64 |
| onehotencoder__PaymentMethod_Electronic check        | 1.0                | float64 |
| onehotencoder__PaymentMethod_Mailed check            | 0.0                | float64 |
| remainder__SeniorCitizen                             | 0.0                | float64 |
| remainder__Dependents                                | 0.0                | float64 |
| remainder__PhoneService                              | 1.0                | float64 |
| remainder__PaperlessBilling                          | 1.0                | float64 |
| remainder__log_Monthly                               | 4.5591262474866845 | float64 |
| remainder__log_Tenure                                | 3.4657359027997265 | float64 |


## Sample Output - y.head()

| y_i | model.predict() | model.predict_proba() | dtype   |
| --- | --------------- | --------------------- | ------- |
| y0  | 1.0             | [0.30,0.70]           | float64 |
| y1  | 0.0             | [0.73,0.27]           | float64 |
| y2  | 1.0             | [0.40,0.60]           | float64 |
| y3  | 0.0             | [0.53,0.47]           | float64 |
| y4  | 1.0             | [0.10,0.90]           | float64 |

