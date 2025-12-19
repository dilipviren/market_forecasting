import numpy as np
import pandas as pd
# print(np.__version__)
# print(pd.__version__)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# import pmdarima as pm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class StockForecaster:
    def __init__(self, train, test, target_col='Close', feature_cols=None, test_size=0.2, look_back=60):
        """
        Initialize the forecaster.
        
        Args:
            data (pd.DataFrame): Input dataframe with DatetimeIndex.
            target_col (str): Name of the column to predict.
            test_size (float): Fraction of data to use for testing (0.2 = 20%).
            look_back (int): Number of past days to consider for LSTM sequences.
        """
        # self.data = data
        self.target_col = target_col
        # self.test_size = test_size
        # self.data = train
        self.look_back = look_back
        
        # Preprocessing: Ensure data is sorted
        self.data = self.data.sort_index()
        
        # Split Index for Time Series (No shuffling)
        # split_idx = int(len(self.data) * (1 - self.test_size))
        self.train_data = train
        self.test_data = test
        # self.train_data = self.data.iloc[:split_idx]
        # self.test_data = self.data.iloc[split_idx:]
        
        # Prepare arrays for Scikit-Learn models (X = Time/Index, y = Price)
        # Note: Using integer index for regression on time
        if feature_cols is None:
            self.X_train_reg = np.arange(len(self.train_data)).reshape(-1, 1)
            self.y_train_reg = self.train_data[self.target_col].values
            self.X_test_reg = np.arange(len(self.train_data), len(self.data)).reshape(-1, 1)
            self.y_test_reg = self.test_data[self.target_col].values

        # Prepare arrays for Models
        else:
            self.X_train_reg = self.train_data[self.feature_cols].values
            self.y_train_reg = self.train_data[self.target_col].values
            self.X_test_reg = self.test_data[self.feature_cols].values
            self.y_test_reg = self.test_data[self.target_col].values

    def _evaluate(self, y_true, y_pred, model_name):
        """Helper to print performance metrics."""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        print(f"[{model_name}] MSE: {mse:.4f} | MAE: {mae:.4f}")
        return {'MSE': mse, 'MAE': mae, 'Predictions': y_pred}

    def linear_regression(self):
        """Standard Linear Regression on the time index."""
        model = LinearRegression()
        model.fit(self.X_train_reg, self.y_train_reg)
        predictions = model.predict(self.X_test_reg)
        return self._evaluate(self.y_test_reg, predictions, "Linear Regression")

    def decision_tree(self):
        """Decision Tree Regressor."""
        model = DecisionTreeRegressor(random_state=42)
        model.fit(self.X_train_reg, self.y_train_reg)
        predictions = model.predict(self.X_test_reg)
        return self._evaluate(self.y_test_reg, predictions, "Decision Tree")

    def support_vector_regression(self):
        """Support Vector Regression with RBF kernel."""
        # SVR requires scaling
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_train_s = scaler_X.fit_transform(self.X_train_reg)
        y_train_s = scaler_y.fit_transform(self.y_train_reg.reshape(-1, 1)).ravel()
        X_test_s = scaler_X.transform(self.X_test_reg)
        
        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        model.fit(X_train_s, y_train_s)
        
        pred_scaled = model.predict(X_test_s)
        predictions = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        return self._evaluate(self.y_test_reg, predictions, "SVR")

    # def auto_arima(self):
    #     """AutoARIMA (Auto-Regressive Integrated Moving Average)."""
    #     # Note: AutoARIMA fits on the target series directly
    #     print("Training AutoARIMA (this may take a moment)...")
    #     model = pm.auto_arima(self.train_data[self.target_col], 
    #                           seasonal=False, 
    #                           stepwise=True, 
    #                           suppress_warnings=True)
        
    #     predictions = model.predict(n_periods=len(self.test_data))
    #     return self._evaluate(self.test_data[self.target_col], predictions, "AutoARIMA")

    def tes_holt_winters(self):
        """Triple Exponential Smoothing (Holt-Winters)."""
        # TES handles seasonality and trend
        model = ExponentialSmoothing(
            self.train_data[self.target_col],
            trend='add',
            seasonal=None, # Set to 'add'/'mul' if seasonality exists and seasonal_periods is known
            initialization_method="estimated"
        ).fit()
        
        predictions = model.forecast(len(self.test_data))
        return self._evaluate(self.test_data[self.target_col], predictions, "TES")

    def lstm_neural_network(self):
        """Long Short-Term Memory (LSTM) Recurrent Neural Network."""
        # 1. Scale Data
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(self.data[[self.target_col]])
        
        # 2. Create Sequences
        def create_dataset(dataset, look_back=1):
            X, Y = [], []
            for i in range(len(dataset) - look_back - 1):
                a = dataset[i:(i + look_back), 0]
                X.append(a)
                Y.append(dataset[i + look_back, 0])
            return np.array(X), np.array(Y)

        # Use only training data to train
        train_scaled = dataset[:len(self.train_data)]
        test_scaled = dataset[len(self.train_data) - self.look_back:] # Include overlap for continuity
        
        X_train, y_train = create_dataset(train_scaled, self.look_back)
        X_test, y_test = create_dataset(test_scaled, self.look_back)
        
        # Reshape input to be [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # 3. Build Model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(self.look_back, 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        # 4. Train (Verbose=0 to hide epoch logs)
        print("Training LSTM...")
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        
        # 5. Predict
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        
        # Invert predictions
        test_predict = scaler.inverse_transform(test_predict).flatten()
        
        # Truncate y_true to match LSTM output shape (lost rows due to look_back)
        y_true = self.data[self.target_col].values[len(self.train_data):]
        # Align lengths if necessary
        min_len = min(len(y_true), len(test_predict))
        
        return self._evaluate(y_true[:min_len], test_predict[:min_len], "LSTM")

    def run_all_models(self):
        """Executes all models and returns a summary."""
        results = {}
        print("--- Starting Model Evaluation ---")
        
        # results['Linear Regression'] = self.linear_regression()
        # results['Decision Tree'] = self.decision_tree()
        # results['SVR'] = self.support_vector_regression()
        results['TES'] = self.tes_holt_winters()
        try:
            results['AutoARIMA'] = self.auto_arima()
        except Exception as e:
            print(f"AutoARIMA failed: {e}")
            
        results['LSTM'] = self.lstm_neural_network()
        
        print("--- Evaluation Complete ---")
        return results

# Example Usage Block (Commented out)
if __name__ == "__main__":

    # Create dummy data
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    values = np.linspace(100, 200, 500) + np.random.normal(0, 5, 500) # Trend + Noise
    df = pd.DataFrame({'Close': values}, index=dates)

    # Instantiate and Run
    forecaster = StockForecaster(df, target_col='Close', look_back=30)
    all_results = forecaster.run_all_models()
    print(pd.DataFrame(all_results).T)
    