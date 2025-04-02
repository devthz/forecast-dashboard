import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def generate_sample_data():
    # Gerando dados de exemplo
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    sales = np.random.normal(loc=1000, scale=200, size=len(dates))
    # Adicionando tendência e sazonalidade
    trend = np.linspace(0, 500, len(dates))
    weekly = 100 * np.sin(np.arange(len(dates)) * (2 * np.pi / 7))
    monthly = 200 * np.sin(np.arange(len(dates)) * (2 * np.pi / 30))
    sales = sales + trend + weekly + monthly
    sales = np.maximum(sales, 0)  # Garantindo valores não negativos
    
    df = pd.DataFrame({
        'ds': dates,
        'y': sales
    })
    return df

def train_prophet_model(df):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(df)
    return model

def create_forecast_plot(df, forecast):
    fig = go.Figure()

    # Dados históricos
    fig.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        name='Historical Data',
        line=dict(color='blue')
    ))

    # Previsão
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        name='Forecast',
        line=dict(color='red')
    ))

    # Intervalo de confiança
    fig.add_trace(go.Scatter(
        x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,0,0,0)'),
        name='Confidence Interval'
    ))

    fig.update_layout(
        title='Sales Forecast',
        xaxis_title='Date',
        yaxis_title='Sales',
        hovermode='x unified'
    )

    return fig

def main():
    st.title("Sales Prediction Dashboard")
    st.write("Analyze and forecast sales data using Prophet")

    # Gerando dados de exemplo
    df = generate_sample_data()

    # Sidebar para controles
    st.sidebar.header("Forecast Parameters")
    forecast_days = st.sidebar.slider("Forecast Days", 30, 365, 90)
    
    if st.sidebar.button("Generate Forecast"):
        with st.spinner("Training model and generating forecast..."):
            # Treinando o modelo
            model = train_prophet_model(df)
            
            # Gerando previsões
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
            
            # Plotando resultados
            fig = create_forecast_plot(df, forecast)
            st.plotly_chart(fig)
            
            # Componentes da previsão
            st.subheader("Forecast Components")
            fig_components = model.plot_components(forecast)
            st.pyplot(fig_components)
            
            # Métricas importantes
            last_date = df['ds'].max()
            next_week = forecast[forecast['ds'] > last_date].head(7)
            next_month = forecast[forecast['ds'] > last_date].head(30)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Average Next Week",
                    f"${next_week['yhat'].mean():,.2f}",
                    f"{((next_week['yhat'].mean() - df['y'].mean()) / df['y'].mean() * 100):+.1f}%"
                )
            with col2:
                st.metric(
                    "Average Next Month",
                    f"${next_month['yhat'].mean():,.2f}",
                    f"{((next_month['yhat'].mean() - df['y'].mean()) / df['y'].mean() * 100):+.1f}%"
                )
            with col3:
                st.metric(
                    "Forecast Trend",
                    "Positive" if forecast['trend'].diff().mean() > 0 else "Negative",
                    f"{forecast['trend'].diff().mean():+.2f}/day"
                )

if __name__ == "__main__":
    main() 