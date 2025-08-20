from datetime import datetime

import numpy as np
import polars as pl
import pytest

from utils.ev_demand import (
    EVDemandCalculator,
    MetadataDataFrameError,
)


@pytest.fixture
def sample_pums_data():
    """Create sample PUMS data for testing."""
    return pl.DataFrame({
        "occupants": [1, 2, 3, 4, 1, 2, 3, 4],
        "income": [30000, 50000, 75000, 100000, 25000, 45000, 80000, 120000],
        "metro": ["urban", "urban", "suburban", "suburban", "rural", "rural", "urban", "urban"],
        "vehicles": [0, 1, 2, 2, 1, 2, 1, 2],
        "hh_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    })


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return pl.DataFrame({
        "bldg_id": ["00001", "00002", "00003", "00004", "00005"],
        "occupants": [1, 2, 3, 4, 2],
        "income": [30000, 50000, 75000, 100000, 45000],
        "metro": ["urban", "urban", "suburban", "suburban", "rural"],
        "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
    })


@pytest.fixture
def mock_nhts_data():
    """Create mock NHTS data."""
    return pl.DataFrame({
        "vehicle_id": [1, 2, 3],
        "start_time": ["08:00", "09:00", "10:00"],
        "end_time": ["18:00", "19:00", "20:00"],
        "miles_driven": [30.0, 25.0, 35.0],
    })


@pytest.fixture
def calculator(sample_metadata, mock_nhts_data):
    """Create EVDemandCalculator instance for testing."""
    return EVDemandCalculator(
        metadata_df=sample_metadata,
        nhts_df=mock_nhts_data,
        pums_df=pl.DataFrame(),  # Will be overridden in tests
        start_date=datetime(2020, 1, 1),  # One week starting from Monday
        end_date=datetime(2020, 1, 7),  # to Sunday
        max_vehicles=2,
        random_state=42,
    )


def test_fit_vehicle_ownership_model_success(calculator, sample_pums_data):
    """Test successful model fitting."""
    model = calculator.fit_vehicle_ownership_model(sample_pums_data)

    # Check that model was fitted
    assert model is not None
    assert calculator.vehicle_ownership_model is not None

    # Check that encoders and scaler were created
    assert hasattr(calculator, "label_encoders")
    assert hasattr(calculator, "target_encoder")
    assert hasattr(calculator, "scaler")

    # Check that metro encoder was created
    assert "metro" in calculator.label_encoders


def test_fit_vehicle_ownership_model_caps_vehicles_at_2(calculator):
    """Test that vehicles > 2 are capped at 2."""
    pums_data = pl.DataFrame({
        "occupants": [1, 2, 3, 4],
        "income": [30000, 50000, 75000, 100000],
        "metro": ["urban", "urban", "suburban", "suburban"],
        "vehicles": [0, 1, 3, 5],  # Values including 0, 1, and > 2
        "hh_weight": [1.0, 1.0, 1.0, 1.0],
    })

    model = calculator.fit_vehicle_ownership_model(pums_data)

    # Check that the model was fitted successfully
    assert model is not None

    # The internal data should have vehicles capped at 2
    # We can verify this by checking the target encoder classes
    expected_classes = [0, 1, 2]  # Should only have 0, 1, 2 vehicles
    assert set(calculator.target_encoder.classes_) == set(expected_classes)


def test_fit_vehicle_ownership_model_with_sample_weights(calculator):
    """Test model fitting with sample weights."""
    from pytest import approx

    # With fixed random_state=42 and fixed input data, coefficients should be exactly:
    expected_coef = np.array([
        [-0.6273701443561746, -0.5479382238614515, 0.14318094439321308],
        [0.20244784926329684, 0.11506717219026819, 0.4769219156039682],
        [0.42492229509287793, 0.4328710516711835, -0.6201028599971814],
    ])
    pums_data = pl.DataFrame({
        "occupants": [1, 2, 3, 4],
        "income": [30000, 50000, 75000, 100000],
        "metro": ["urban", "urban", "suburban", "suburban"],
        "vehicles": [0, 1, 2, 2],
        "hh_weight": [2.0, 1.5, 1.0, 0.5],  # Different weights
    })

    model = calculator.fit_vehicle_ownership_model(pums_data)
    assert calculator.vehicle_ownership_model is not None
    # Check that model was fitted successfully and has expected coefficients
    assert model.coef_ == approx(expected_coef, rel=1e-6)


def test_fit_vehicle_ownership_model_handles_missing_values(calculator, caplog):
    """Test that missing values are handled appropriately."""
    pums_data = pl.DataFrame({
        "occupants": [1, 2, None, 4, 3, 5],
        "income": [30000, 50000, 75000, None, 60000, 80000],
        "metro": ["urban", None, "suburban", "suburban", "urban", "rural"],
        "vehicles": [0, 1, 2, 2, 1, 2],
        "hh_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    })

    # The function should handle missing values gracefully by dropping incomplete rows
    # After dropping rows with missing values, we should have enough data to fit the model
    model = calculator.fit_vehicle_ownership_model(pums_data)
    assert model is not None
    assert calculator.vehicle_ownership_model is not None

    # Check that the warning message was logged correctly
    assert len(caplog.records) > 0
    warning_message = caplog.records[0].message
    assert "Dropped 3 records with missing values in vehicle ownership model features" in warning_message


def test_predict_num_vehicles_success(calculator, sample_pums_data, sample_metadata):
    """Test successful vehicle prediction."""
    # First fit the model
    calculator.fit_vehicle_ownership_model(sample_pums_data)

    # Then predict
    result_df = calculator.predict_num_vehicles(sample_metadata)

    # Check that result has the expected columns
    assert "vehicles" in result_df.columns
    assert len(result_df) == len(sample_metadata)

    # Check that all original columns are preserved
    for col in sample_metadata.columns:
        assert col in result_df.columns

    # Check that vehicle predictions are integers in expected range
    vehicle_predictions = result_df.get_column("vehicles").to_list()
    assert all(isinstance(v, (int, np.integer)) for v in vehicle_predictions)
    assert all(0 <= v <= 2 for v in vehicle_predictions)


def test_predict_num_vehicles_without_fitted_model(calculator, sample_metadata, sample_pums_data):
    """Test that model is automatically fitted when predicting without a fitted model."""
    # Ensure model isn't fitted
    calculator.vehicle_ownership_model = None
    calculator.pums_df = sample_pums_data  # Set PUMS data for auto-fitting

    # Should not raise an error, but instead fit the model automatically
    result = calculator.predict_num_vehicles(sample_metadata)

    # Verify model was fitted
    assert calculator.vehicle_ownership_model is not None

    # Verify predictions are reasonable
    assert "vehicles" in result.columns
    assert len(result) == len(sample_metadata)
    vehicle_predictions = result.get_column("vehicles").to_list()
    assert all(isinstance(v, (int, np.integer)) for v in vehicle_predictions)
    assert all(0 <= v <= 2 for v in vehicle_predictions)


def test_predict_num_vehicles_without_metadata(calculator, sample_pums_data):
    """Test that error is raised when no metadata is provided."""
    calculator.fit_vehicle_ownership_model(sample_pums_data)
    # Clear the metadata_df to test the None case properly
    calculator.metadata_df = None

    with pytest.raises(MetadataDataFrameError):
        calculator.predict_num_vehicles(None)


def test_predict_num_vehicles_missing_required_features(calculator, sample_pums_data):
    """Test that error is raised when required features are missing."""
    calculator.fit_vehicle_ownership_model(sample_pums_data)

    # Create metadata with missing values
    incomplete_metadata = pl.DataFrame({
        "bldg_id": [1, 2, 3],
        "occupants": [1, None, 3],  # Missing value
        "income": [30000, 50000, None],  # Missing value
        "metro": ["urban", "urban", None],  # Missing value
    })

    with pytest.raises(ValueError) as exc_info:
        calculator.predict_num_vehicles(incomplete_metadata)

    # Check that the error message mentions missing data
    assert "Missing vehicle ownership model input data" in str(exc_info.value)
    assert "building IDs" in str(exc_info.value)


def test_predict_num_vehicles_consistent_predictions(calculator, sample_pums_data, sample_metadata):
    """Test that predictions are consistent for same input."""
    calculator.fit_vehicle_ownership_model(sample_pums_data)

    # Make predictions twice
    result1 = calculator.predict_num_vehicles(sample_metadata)
    result2 = calculator.predict_num_vehicles(sample_metadata)

    # Results should be identical
    assert result1.equals(result2)


def test_predict_num_vehicles_feature_encoding(calculator, sample_pums_data):
    """Test that categorical features are properly encoded during prediction."""
    calculator.fit_vehicle_ownership_model(sample_pums_data)

    # Create metadata with new metro values not seen during training
    new_metadata = pl.DataFrame({
        "bldg_id": [1],
        "occupants": [2],
        "income": [50000],
        "metro": ["urban"],  # This should be encoded properly
    })

    # Should not raise an error
    result = calculator.predict_num_vehicles(new_metadata)
    assert len(result) == 1
    assert "vehicles" in result.columns


def test_predict_num_vehicles_feature_scaling(calculator, sample_pums_data):
    """Test that features are properly scaled during prediction."""
    calculator.fit_vehicle_ownership_model(sample_pums_data)

    # Create metadata with extreme values to test scaling
    extreme_metadata = pl.DataFrame({
        "bldg_id": [1, 2],
        "occupants": [1, 10],  # Extreme values
        "income": [10000, 500000],  # Extreme values
        "metro": ["urban", "suburban"],
    })

    # Should not raise an error and should produce predictions
    result = calculator.predict_num_vehicles(extreme_metadata)
    assert len(result) == 2
    assert "vehicles" in result.columns
    vehicle_predictions = result.get_column("vehicles").to_list()
    assert all(0 <= v <= 2 for v in vehicle_predictions)


def test_random_seed_reproducibility(calculator, sample_pums_data, sample_metadata):
    """Test that the random seed produces consistent results across multiple runs."""
    # Fit the model
    calculator.fit_vehicle_ownership_model(sample_pums_data)

    # Make predictions multiple times
    result1 = calculator.predict_num_vehicles(sample_metadata)
    result2 = calculator.predict_num_vehicles(sample_metadata)
    result3 = calculator.predict_num_vehicles(sample_metadata)

    # All results should be identical
    assert result1.equals(result2)
    assert result2.equals(result3)
    assert result1.equals(result3)

    # Check that the actual predictions are the same
    predictions1 = result1.get_column("vehicles").to_list()
    predictions2 = result2.get_column("vehicles").to_list()
    predictions3 = result3.get_column("vehicles").to_list()

    assert predictions1 == predictions2
    assert predictions2 == predictions3
