import pandas as pd
data={
    "Name":["Alice","Bob","Charlie","David"],
    "Age":[24,30,29,35],
    "City":["New York","Chicago","San Frascisco","New York"]
}
df=pd.DataFrame(data)
filtered_df=df[df["Age"]>28]
print("filtered dataframe: ")
print(filtered_df)