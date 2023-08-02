
import numpy as np


def calculate_sharpe_ratio(returns, risk_free_rate):
    """
    Calcular el ratio de Sharpe

    Parámetros:
    returns (np.array): Array de rendimientos de la inversión
    risk_free_rate (float): Tasa de rendimiento sin riesgo

    Devuelve:
    sharpe_ratio (float): Ratio de Sharpe
    """

    # Calcular el rendimiento promedio
    avg_returns = np.mean(returns)

    # Calcular la desviación estándar de los rendimientos
    std_returns = np.std(returns)

    # Calcular el ratio de Sharpe
    sharpe_ratio = (avg_returns - risk_free_rate) / std_returns

    return sharpe_ratio


def h_price_to_data(h_price, days_backward = 150, days_forward = 30, days_steps = 1  ):
    datos_analisis = h_price
    X_DATA = []
    Y_DATA = []

    for i in range(days_backward, len(datos_analisis), days_steps):
        # retornos logaritmicos
        X_data = datos_analisis[i-days_backward:i]
        X_data = np.log(X_data).diff().dropna()
        X_DATA.append(X_data)

        # y
        data_forward = datos_analisis[i:i+days_forward]

        rs = calculate_sharpe_ratio(np.log(data_forward).diff().dropna(), 0)
        Y_data = np.argsort(np.argsort(rs))
        Y_DATA.append(Y_data)

    X_DATA = np.array(X_DATA)
    Y_DATA = np.array(Y_DATA)

    return X_DATA, Y_DATA