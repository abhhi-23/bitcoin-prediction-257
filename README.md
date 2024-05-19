# BitSmart: Machine Learning for Bitcoin Price Prediction and Trading

## Team Members

Harbans Singh Toor (017405076) - harbanssingh.toor@sjsu.edu
Abhi Kadam (017432194) - abhi.kadam@sjsu.edu
Mohana Moganti (017163497) - mohanasatyanarayana.moganti@sjsu.edu

## Project Overview

BitSmart is a cutting-edge machine learning service designed specifically for Bitcoin price prediction and swing trading strategy formulation. Developed as part of the CMPE 257 Machine Learning course, this tool utilizes historical Bitcoin market data to forecast future price movements and provide actionable trading insights. This project simplifies the complex nature of cryptocurrency trading into a user-friendly web application, making advanced trading strategies accessible to everyone.

## Project URL

https://bitcoin-prediction-257-h9k2oxhgt55baro4ykvenw.streamlit.app/

## Getting Started

These instructions will guide you through setting up BitSmart on your local environment for development and testing. Follow these steps to prepare your system to run the service.

### Prerequisites

Before installing BitSmart, ensure you have the following software installed on your machine:
- Python 3.8 or newer
- pip (Python package installer)

This project depends on several Python libraries, including TensorFlow for machine learning models, Pandas for data manipulation, and Plotly for interactive charts.

### Installation

Follow these steps to get a development environment running:

1. **Clone the GitHub repository:**
   ```
   git clone https://github.com/yourusername/bitsmart.git
   cd bitsmart
   ```

2. **Install Python dependencies:**
   ```
   pip install -r requirements.txt
   ```

This will install all necessary libraries to run BitSmart, including TensorFlow, Scikit-Learn, and Plotly.

## Usage

Once installed, you can run BitSmart using the following steps:

1. **Launch the application:**
   ```
   python app.py
   ```

2. **Access the Web Interface:**
   Navigate to `http://localhost:5000` using any web browser to interact with the BitSmart interface. Here, you can select a date, predict Bitcoin prices for the next seven days, and view recommended trading strategies.

## Features

### Bitcoin Price Prediction
- **Daily Forecasts:** Predicts the highest, lowest, and average prices for the next seven days.
- **Data-driven Insights:** Utilizes LSTM neural networks to analyze historical price data for predictions.

### Swing Trading Strategy
- **Automated Trading Decisions:** Offers a strategic plan involving selling and buying instructions to maximize potential returns.
- **Responsive Strategy Adjustments:** Adjusts recommendations based on the latest predictions and market conditions.

### User Interface
- **Interactive Dashboard:** A clean and intuitive web interface that allows users to easily interact with the prediction system.
- **Real-time Data Visualization:** Charts and graphs that update in real-time to reflect the latest predictions and strategies.

## Data Sources

BitSmart uses publicly available data:
- **Primary Data Source:** Historical daily prices of Bitcoin from Yahoo Finance.
- **Optional Data:** Incorporates additional economic indicators like the S&P500 Index and US Consumer Confidence to enhance prediction accuracy.

## Model Details

### Architecture
- **LSTM Network:** Employs Long Short-Term Memory units capable of learning from sequences of historical price data, ideal for time-series prediction.

### Training and Evaluation
- **Data Preprocessing:** Involves normalization and handling of missing data to prepare the dataset for training.
- **Model Training:** Conducts extensive training sessions using backpropagation and optimization techniques to fine-tune the neural network.
- **Performance Metrics:** Utilizes metrics such as RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error) to evaluate prediction accuracy.

## Project Management

### Team Collaboration
- **Source Control:** Uses Git for version control, ensuring that all team members can collaborate effectively on the codebase.
- **Task Management:** Utilizes tools like Jira or Trello for tracking progress and managing tasks across the team.

### Deliverables
- **Codebase:** Complete source code and resources necessary to run BitSmart.
- **Documentation:** Detailed project report and presentations outlining the design, implementation, and performance of the system.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

- Professors and TA's of CMPE 257 for guidance and support.
- Open-source contributors whose tools and libraries facilitated this project.
