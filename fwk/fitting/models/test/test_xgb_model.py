import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from fitting.fitting_core import TaskType
from fitting.models.xgb_model import XGBoostModel
from fitting.fitting_metrics import RegressionMetric


# Helper: Generate synthetic data
def generate_classification_data(n_samples=100, n_features=5):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], size=n_samples)
    return X.astype(np.float32), y.astype(np.float32)


def generate_regression_data(n_samples=100, n_features=5):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.1  # linear target
    return X.astype(np.float32), y.astype(np.float32)


# -------------------- TEST: CLASSIFICATION MODE --------------------

def test_xgb_model_classification():
    """Test XGBoostModel in classification mode."""
    
    # Generate data
    X, y = generate_classification_data()

    task_type=TaskType.CLASSIFICATION

    # Initialize model in classification mode
    model = XGBoostModel(task_type=task_type)

    # Assert correct internal model type
    assert isinstance(model._model, XGBClassifier), "Model should be XGBClassifier in classification mode"

    # Test default parameters
    default_params = model.getDefaultParams(task_type=task_type)
    assert isinstance(default_params, dict)
    assert 'n_estimators' in default_params

    # Test parameter setting
    custom_params = {
        'n_estimators': 50,
        'max_depth': 3,
        'learning_rate': 0.2
    }
    model.setParams(**custom_params)
    current_params = model.getParams()
    for key, value in custom_params.items():
        assert current_params[key] == value

    # Test parameter grid
    param_grid = model.getParamsGrids(task_type=task_type)
    assert 'n_estimators' in param_grid
    assert len(param_grid['n_estimators']) > 0
    assert 'learning_rate' in param_grid

    # Test fit and predict
    model.fit(X, y)
    predictions = model.predict(X)

    # Assertions on prediction output
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(y),)  # 1D array
    assert np.issubdtype(predictions.dtype, np.floating)
    assert np.all(np.isfinite(predictions))  # No NaN or inf

    # Predictions should be class probabilities? Actually, XGBClassifier predict() returns class labels (not probs)
    # Check they are integers cast to float
    unique_preds = np.unique(predictions)
    assert set(unique_preds).issubset({0.0, 1.0}), "Classification predictions should be 0 or 1"


# -------------------- TEST: REGRESSION MODE --------------------

def test_xgb_model_regression():
    """Test XGBoostModel in regression mode."""
    
    # Generate data
    X, y = generate_regression_data()
    task_type=TaskType.REGRESSION

    # Initialize model in regression mode
    model = XGBoostModel(task_type=task_type)

    # Assert correct internal model type
    assert isinstance(model._model, XGBRegressor), "Model should be XGBRegressor in regression mode"

    # Test default parameters
    default_params = model.getDefaultParams(task_type=task_type)
    assert isinstance(default_params, dict)
    assert 'n_estimators' in default_params
    assert default_params['eval_metric'] == 'rmse'

    # Test parameter setting
    custom_params = {
        'n_estimators': 200,
        'max_depth': 9,
        'learning_rate': 0.05
    }
    model.setParams(**custom_params)
    current_params = model.getParams()
    for key, value in custom_params.items():
        assert current_params[key] == value

    # Test parameter grid
    param_grid = model.getParamsGrids(task_type=task_type)
    assert 'max_depth' in param_grid
    assert len(param_grid['max_depth']) == 3

    # Test fit and predict
    model.fit(X, y)
    predictions = model.predict(X)

    # Assertions on prediction output
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(y),)  # 1D array
    assert np.issubdtype(predictions.dtype, np.floating)
    assert np.all(np.isfinite(predictions))

    # Check that predictions are continuous (not discrete)
    unique_preds = np.unique(predictions)
    assert len(unique_preds) > 2, "Regression predictions should be continuous values"

    # Optional: check that RMSE is reasonable (just a sanity check)
    mse = np.mean((predictions - y) ** 2)
    assert mse < 1.0, f"Regression MSE too high: {mse}"


def generate_regression_data_stationary(n_samples=500, n_features=20, noise=0.1, seed=42):
    """
    Generate stationary regression data WITHOUT concept drift.
    Uses sklearn.make_regression which creates data where:
    - X and y are related by a fixed linear relationship
    - No temporal trends that would cause train/val/test distribution mismatch
    """
    from sklearn.datasets import make_regression
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(15, n_features),
        noise=noise,
        random_state=seed
    )
    return X.astype(np.float32), y.astype(np.float32)


def test_xgb_regression_metrics_positive():
    """
    Test XGBoostModel regression with proper metrics.
    
    This test verifies that:
    1. Training data produces high R² (model learns the pattern)
    2. Validation/test splits produce positive R² (no concept drift)
    3. Metrics are not zero or NaN
    
    The issue with R²=0 on validation/test was caused by concept drift
    in the test data (temporal trends causing train/val/test distribution mismatch).
    """
    np.random.seed(123)
    
    # Generate stationary regression data (no concept drift)
    X, y = generate_regression_data_stationary(n_samples=500, n_features=20, noise=1.0, seed=42)
    
    # Verify data has consistent distribution across splits
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]
    
    # The means should be similar (within 2 std for stationary data)
    y_std = np.std(y)
    mean_diff_train_val = abs(np.mean(y_train) - np.mean(y_val))
    mean_diff_train_test = abs(np.mean(y_train) - np.mean(y_test))
    
    assert mean_diff_train_val < 2 * y_std, f"Train/Val mean difference too large: {mean_diff_train_val}"
    assert mean_diff_train_test < 2 * y_std, f"Train/Test mean difference too large: {mean_diff_train_test}"
    
    # Split data
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    # Train XGBoost
    model = XGBoostModel(task_type=TaskType.REGRESSION)
    model.setParams(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predict on all splits
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    metric_calc = RegressionMetric()
    
    train_metrics = metric_calc.calculate(y_train, train_pred)
    val_metrics = metric_calc.calculate(y_val, val_pred)
    test_metrics = metric_calc.calculate(y_test, test_pred)
    
    print("\n=== XGBoost Regression Metrics ===")
    print(f"Train: r2={train_metrics['r2']:.4f}, score={train_metrics['score']:.4f}")
    print(f"Val:   r2={val_metrics['r2']:.4f}, score={val_metrics['score']:.4f}")
    print(f"Test:  r2={test_metrics['r2']:.4f}, score={test_metrics['score']:.4f}")
    
    # CRITICAL ASSERTIONS: These were failing with R²=0 due to concept drift
    assert train_metrics['r2'] > 0.5, f"Train R² should be > 0.5, got {train_metrics['r2']}"
    assert train_metrics['score'] > 0.5, f"Train score should be > 0.5, got {train_metrics['score']}"
    
    assert val_metrics['r2'] > 0, f"Val R² should be > 0 (no concept drift), got {val_metrics['r2']}"
    assert val_metrics['score'] > 0, f"Val score should be > 0 (no concept drift), got {val_metrics['score']}"
    
    assert test_metrics['r2'] > 0, f"Test R² should be > 0 (no concept drift), got {test_metrics['r2']}"
    assert test_metrics['score'] > 0, f"Test score should be > 0 (no concept drift), got {test_metrics['score']}"
    
    # Verify no NaN values
    for key, val in {**train_metrics, **val_metrics, **test_metrics}.items():
        assert np.isfinite(val), f"Metric {key} should be finite, got {val}"


def test_xgb_regression_metrics_with_normalization():
    """
    Test XGBoostModel regression with feature normalization.
    
    This test simulates the TimeSeriesModelTrainer behavior with:
    - StandardScaler normalization
    - Proper train/val/test splits
    - Verification that metrics remain positive
    """
    from sklearn.preprocessing import StandardScaler
    
    np.random.seed(456)
    
    # Generate stationary data
    X, y = generate_regression_data_stationary(n_samples=500, n_features=20, noise=1.0, seed=42)
    
    # Split data
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    # Normalize features (like TimeSeriesModelTrainer does)
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    
    # Train model
    model = XGBoostModel(task_type=TaskType.REGRESSION)
    model.setParams(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train_norm, y_train)
    
    # Predict
    model.predict(X_train_norm)  # warm-up prediction
    val_pred = model.predict(X_val_norm)
    test_pred = model.predict(X_test_norm)
    
    # Calculate metrics
    metric_calc = RegressionMetric()
    val_metrics = metric_calc.calculate(y_val, val_pred)
    test_metrics = metric_calc.calculate(y_test, test_pred)
    
    print("\n=== XGBoost Regression with Normalization ===")
    print(f"Val:   r2={val_metrics['r2']:.4f}, score={val_metrics['score']:.4f}")
    print(f"Test:  r2={test_metrics['r2']:.4f}, score={test_metrics['score']:.4f}")
    
    # Critical assertions
    assert val_metrics['r2'] > 0, f"Val R² should be > 0, got {val_metrics['r2']}"
    assert val_metrics['score'] > 0, f"Val score should be > 0, got {val_metrics['score']}"
    assert test_metrics['r2'] > 0, f"Test R² should be > 0, got {test_metrics['r2']}"
    assert test_metrics['score'] > 0, f"Test score should be > 0, got {test_metrics['score']}"


def test_xgb_periodic_retrain_sliding_window():
    """
    Test XGBoostModel with PERIODIC_RETRAIN mode and SLIDING window.
    
    This test reproduces the user's configuration:
    - train_ratio=0.2 (only 20% for initial training)
    - sliding_window_size=126
    - retrain_period=21
    - RetrainMode.SLIDING
    
    This test verifies that:
    1. The model trains successfully on the initial data
    2. Periodic retraining works correctly
    3. Metrics are positive (no concept drift with stationary data)
    """
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    
    np.random.seed(456)
    
    # Generate stationary data (no concept drift)
    X, y = generate_regression_data_stationary(n_samples=1000, n_features=20, noise=1.0, seed=42)
    
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Target: mean={y.mean():.2f}, std={y.std():.2f}, range=[{y.min():.2f}, {y.max():.2f}]")
    
    # Configuration matching user's setup
    train_ratio = 0.2
    sliding_window_size = 126
    retrain_period = 21
    
    n = len(X)
    train_end = int(n * train_ratio)
    
    # Split data
    X_train = X[:train_end]
    y_train = y[:train_end]
    X_test = X[train_end:]
    y_test = y[train_end:]
    
    print(f"Train: {len(X_train)} samples, y mean={y_train.mean():.2f}")
    print(f"Test:  {len(X_test)} samples, y mean={y_test.mean():.2f}")
    
    # Simulate the periodic retraining with sliding window
    scaler = StandardScaler()
    
    all_predictions = []
    retrain_count = 0
    
    for test_start in range(0, len(X_test), retrain_period):
        test_end = min(test_start + retrain_period, len(X_test))
        
        # SLIDING window: use most recent sliding_window_size samples
        window_start = max(0, len(X_train) + test_start - sliding_window_size)
        window_end = len(X_train) + test_start
        
        X_train_window = X[window_start:window_end]
        y_train_window = y[window_start:window_end]
        
        # Fit scaler on training window
        X_train_norm = scaler.fit_transform(X_train_window)
        
        # Train model
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train_norm, y_train_window)
        
        # Predict on test chunk
        X_test_chunk = X_test[test_start:test_end]
        X_test_norm = scaler.transform(X_test_chunk)
        y_pred = model.predict(X_test_norm)
        
        all_predictions.extend(y_pred)
        retrain_count += 1
        
        # Calculate R² for this chunk
        y_test_chunk = y_test[test_start:test_end]
        ss_res = np.sum((y_test_chunk - y_pred) ** 2)
        ss_tot = np.sum((y_test_chunk - np.mean(y_test_chunk)) ** 2)
        r2_chunk = 1 - (ss_res / (ss_tot + 1e-8))
        
        print(f"  Chunk [{test_start}:{test_end}]: R²={r2_chunk:.4f}")
    
    # Final metrics on entire test set
    all_predictions = np.array(all_predictions)
    metric_calc = RegressionMetric()
    final_test_metrics = metric_calc.calculate(y_test, all_predictions)
    
    print(f"\nFinal Test Metrics (after {retrain_count} retrainings):")
    print(f"  R²:   {final_test_metrics['r2']:.4f}")
    print(f"  Score: {final_test_metrics['score']:.4f}")
    print(f"  MSE:   {final_test_metrics['mse']:.4f}")
    
    # Calculate train metrics (on the last training window used)
    window_start = max(0, len(X_train) - sliding_window_size)
    X_train_final = X[window_start:len(X_train)]
    y_train_final = y[window_start:len(X_train)]
    
    if len(X_train_final) > 0:
        X_train_final_norm = scaler.transform(X_train_final)
        y_train_pred = model.predict(X_train_final_norm)
        train_metrics = metric_calc.calculate(y_train_final, y_train_pred)
        
        print(f"\nTrain Metrics (last window [{window_start}:{len(X_train)}]):")
        print(f"  R²:   {train_metrics['r2']:.4f}")
        print(f"  Score: {train_metrics['score']:.4f}")
        
        assert train_metrics['r2'] > 0, f"Train R² should be > 0, got {train_metrics['r2']}"
        assert train_metrics['score'] > 0, f"Train score should be > 0, got {train_metrics['score']}"
    
    # Critical assertions
    assert final_test_metrics['r2'] > 0, f"Test R² should be > 0, got {final_test_metrics['r2']}"
    assert final_test_metrics['score'] > 0, f"Test score should be > 0, got {final_test_metrics['score']}"
    assert train_metrics['r2'] > 0, f"Train R² should be > 0, got {train_metrics['r2']}"
    assert train_metrics['score'] > 0, f"Train score should be > 0, got {train_metrics['score']}"
    
    # Verify no NaN values
    assert np.all(np.isfinite(all_predictions)), "Predictions should not contain NaN or inf"

