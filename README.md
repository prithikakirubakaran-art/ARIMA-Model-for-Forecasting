# Experiment 6: ARIMA Model for Forecasting

## Aim

To evaluate predictive models using performance metrics.

## Algorithm

1.  **START**
2.  Prepare stationary time series.
3.  Identify ARIMA -- (p, d, q) parameters using ACF/PACF.
4.  Fit the ARIMA model on the data.
5.  Forecast future periods and plot the results.
6.  Evaluate forecast accuracy.
7.  **STOP**



## Program

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # Create time series dataset
    data = [112, 118, 132, 129, 121, 135, 148, 148,
            136, 119, 104, 118, 115, 126, 141, 135,
            125, 149, 170, 170, 158, 133, 114, 140,
            145, 150, 178, 163, 172, 178, 199, 199]
    
    # Convert into Pandas Series
    series = pd.Series(data)
    
    # Split data into training and testing sets
    train = series[:-6]
    test = series[-6:]
    
    # Fit ARIMA model
    model = ARIMA(train, order=(2, 1, 2))
    model_fit = model.fit()
    
    # Forecast future values
    forecast = model_fit.forecast(steps=6)
    
    # Display forecast
    print("Forecasted Values:")
    print(forecast)
    
    # Evaluate model performance
    mae = mean_absolute_error(test, forecast)
    mse = mean_squared_error(test, forecast)
    rmse = np.sqrt(mse)
    
    print("\nMean Absolute Error:", round(mae, 2))
    print("Mean Squared Error:", round(mse, 2))
    print("Root Mean Squared Error:", round(rmse, 2))
    
    # Plot actual and forecast values
    plt.figure(figsize=(8, 4))
    plt.plot(series.index, series, label="Actual Data")
    plt.plot(test.index, forecast, label="Forecast", color="red")
    plt.title("ARIMA Time Series Forecasting")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.legend()
    plt.grid()
    plt.show()




## Output
    Forecasted Values:
    26    148.564656
    27    143.747981
    28    140.693361
    29    140.102969
    30    140.721014
    31    141.380003
    Name: predicted_mean, dtype: float64
    
    Mean Absolute Error: 38.97
    Mean Squared Error: 1728.31
    Root Mean Squared Error: 41.57

 <img width="695" height="393" alt="image" src="https://github.com/user-attachments/assets/970b6591-229e-4e06-bede-253346a5d973" />
   



## Result
Thus, the ARIMA time-series forecasting model was successfully implemented using Python. The model was fitted to the training data to forecast future values. Forecast accuracy was evaluated using MAE, MSE, and RMSE, and the actual and predicted values were plotted for comparison.
