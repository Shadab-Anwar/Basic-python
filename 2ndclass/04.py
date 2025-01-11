import pandas as pd
data={
    "Name":["Alice","Bob","Charlie","David"],
    "Age":[24,30,29,35],
    "City":["New York","Chicago","San Frascisco","New York"]
}
Data=pd.DataFrame(data)
filtered_Data=Data[Data["Age"]>28]
print("filtered dataframe: ")
print(filtered_Data)