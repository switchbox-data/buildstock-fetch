import logging
from dataclasses import dataclass, field
from typing import Any

import polars as pl
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.preprocessing import LabelEncoder, StandardScaler  # type: ignore[import-untyped]


@dataclass
class VehicleOwnershipModel:
    """Multinomial logistic model for household vehicle counts from PUMS features."""

    max_vehicles: int = 2
    random_state: int = 42
    feature_columns: list[str] = field(default_factory=lambda: ["occupants", "income", "metro"])
    label_encoders: dict[str, LabelEncoder] = field(default_factory=dict, init=False, repr=False)
    target_encoder: LabelEncoder | None = field(default=None, init=False, repr=False)
    scaler: StandardScaler | None = field(default=None, init=False, repr=False)
    vehicle_ownership_model: Any | None = field(default=None, init=False, repr=False)

    def fit(self, pums_df: pl.DataFrame) -> Any:
        """
        Fit a multinomial logistic regression model to predict number of vehicles per household using PUMS data.
        Results limited to 0, 1, or 2 vehicles.

        Args:
            pums_df: DataFrame with PUMS household data, including 'occupants', 'income', 'metro', 'vehicles'

        Returns:
            Trained model object
        """
        # Preprocess data: replace vehicles > 2 with 2, drop nulls, encode categorical
        pums_df = pums_df.with_columns(
            pl.when(pl.col("vehicles") > self.max_vehicles)
            .then(self.max_vehicles)
            .otherwise(pl.col("vehicles"))
            .alias("vehicles")
        )

        # Drop rows with missing values in required features
        feature_columns = self.feature_columns

        # Drop rows with missing values in required features and target variable
        initial_count = len(pums_df)
        pums_df = pums_df.drop_nulls(subset=[*feature_columns, "vehicles"])
        dropped_count = initial_count - len(pums_df)

        if dropped_count > 0:
            logging.warning(f"Dropped {dropped_count} records with missing values in vehicle ownership model features")

        # Prepare features and encode categorical
        X = pums_df.select(feature_columns)
        y = pums_df.select("vehicles")

        # Encode metro (only categorical feature)
        self.label_encoders = {}
        le = LabelEncoder()
        metro_encoded = le.fit_transform(X.get_column("metro").to_numpy())
        X_encoded = X.with_columns(pl.Series(metro_encoded).alias("metro"))
        self.label_encoders["metro"] = le

        # Encode target and scale features
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y.get_column("vehicles").to_numpy())

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_encoded.to_numpy())

        # Fit model with sample weights
        self.vehicle_ownership_model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=self.random_state)
        self.vehicle_ownership_model.fit(X_scaled, y_encoded, sample_weight=pums_df.get_column("hh_weight").to_numpy())

        return self.vehicle_ownership_model

    def predict(self, metadata_df: pl.DataFrame) -> pl.DataFrame:
        """
        Predict number of vehicles for each household in the metadata using the fitted model.
        If the model hasn't been fitted yet, it will be fitted automatically using the PUMS data.

        Args:
            metadata_df: DataFrame with ResStock metadata.

        Returns:
            DataFrame with an added 'vehicles' column
        """
        if self.scaler is None or self.target_encoder is None or self.vehicle_ownership_model is None:
            raise ValueError("Vehicle ownership model has not been fitted yet. Call fit() first.")

        # Step 1: Prepare features from metadata
        feature_columns = self.feature_columns
        X = metadata_df.select(["bldg_id", *feature_columns])

        # Validate no missing values in required features
        if X.select(feature_columns).null_count().sum_horizontal().item() > 0:
            # Find building IDs with missing values
            missing_data = X.filter(
                pl.col("occupants").is_null() | pl.col("income").is_null() | pl.col("metro").is_null()
            )
            missing_bldg_ids = missing_data.get_column("bldg_id").to_list()
            raise ValueError("Missing vehicle ownership model input data for building IDs: " + str(missing_bldg_ids))

        # Separate features and building IDs
        features = X.select(feature_columns)

        # Step 2: Encode categorical variables
        features_encoded = features.clone()
        metro_encoded = self.label_encoders["metro"].transform(features.get_column("metro").to_numpy())
        features_encoded = features_encoded.with_columns(pl.Series(metro_encoded).alias("metro"))

        # Step 3: Scale features
        features_scaled = self.scaler.transform(features_encoded.to_numpy())

        # Step 4: Make predictions
        predictions_encoded = self.vehicle_ownership_model.predict(features_scaled)

        # Step 5: Decode predictions and add to DataFrame
        predictions_decoded = self.target_encoder.inverse_transform(predictions_encoded)

        # Add predictions to original DataFrame
        return metadata_df.with_columns(pl.Series(predictions_decoded).alias("vehicles"))
