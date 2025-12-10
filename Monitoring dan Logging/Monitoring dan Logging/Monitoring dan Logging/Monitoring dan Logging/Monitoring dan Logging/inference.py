import mlflow.pyfunc
import pandas as pd

model = mlflow.pyfunc.load_model("model")

sample = pd.DataFrame({
    "age": [30],
    "income": [5000],
    "gender_F": [0],
    "gender_M": [1],
    "gender_None": [0]
})

pred = model.predict(sample)
print("Prediction:", pred)
