import pandas as pd
import json

csv_path = "data/Electric_Vehicle_Population_Data_Preprocessed.csv"

data = pd.read_csv(csv_path)

makes = sorted(data["Make"].dropna().unique().tolist())
models = sorted(data["Model"].dropna().unique().tolist())

options = {
    "makes": makes,
    "models": models
}

with open("data/vehicle_options.json", "w", encoding="utf-8") as file:
    json.dump(options, file, indent=2, ensure_ascii=False)

print("vehicle_options.json created successfully!")
print("Makes:", len(makes))
print("Models:", len(models))