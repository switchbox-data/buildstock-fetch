# mypy: disable-error-code="import-untyped"  # scikit-learn types
# type: ignore[reportAssignmentType]
import os

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from utils import ev_utils
from utils.ev_demand import EVDemandCalculator, EVDemandConfig

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# Load data first
config = EVDemandConfig(state="NY", release="resstock_tmy3_release_1")
metadata_df, nhts_df, pums_df, weather_df = ev_utils.load_all_input_data(config)
np.random.seed(42)
# Create calculator
calculator = EVDemandCalculator(metadata_df=metadata_df, nhts_df=nhts_df, pums_df=pums_df, weather_df=weather_df)

# Cap vehicles BEFORE splitting
pums_df = pums_df.with_columns(pl.when(pl.col("vehicles") > 2).then(2).otherwise(pl.col("vehicles")).alias("vehicles"))

# Split PUMS data for evaluation
pums_train, pums_test = train_test_split(pums_df, test_size=0.2, random_state=42)

# Fit model on training data
calculator.fit_vehicle_ownership_model(pums_train)

# Print actual coefficients to update expected values
print("\nActual coefficients:")
print(calculator.vehicle_ownership_model.coef_.tolist())


# Prepare test data
test_features = pums_test.select(["occupants", "income", "metro"])
test_target = pums_test.select("vehicles")

# Encode test features
test_encoded = test_features.clone()
metro_encoded = calculator.label_encoders["metro"].transform(test_features.get_column("metro").to_numpy())
test_encoded = test_encoded.with_columns(pl.Series(metro_encoded).alias("metro"))

# Scale and predict
test_scaled = calculator.scaler.transform(test_encoded.to_numpy())
test_pred_encoded = calculator.vehicle_ownership_model.predict(test_scaled)
test_pred = calculator.target_encoder.inverse_transform(test_pred_encoded)

# Get actual values
test_actual = test_target.get_column("vehicles").to_numpy()

# Performance metrics
accuracy = accuracy_score(test_actual, test_pred)
conf_matrix = confusion_matrix(test_actual, test_pred)
class_report = classification_report(test_actual, test_pred)

print(f"Model Accuracy: {accuracy:.3f}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
