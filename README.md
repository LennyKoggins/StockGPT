# StockGPT

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Data Preprocessing](#data-preprocessing)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Introduction
StockGPT is a comprehensive stock analysis and prediction dashboard built using Streamlit. It allows users to load stock data from an Excel file, preprocess the data, and visualize various metrics and trends. The application also includes features for predicting the best-performing stocks and recommending similar stocks based on historical data.

## Features
- Load and preprocess stock data from an Excel file
- Configure analysis parameters via a sidebar
- Visualize feature importance, correlation heatmap, and distribution of scores
- Display periodic market cap trends and best stock counts
- Recommend similar stocks based on optimization
- Generate classification reports and predict the best stock for future periods

## Installation
To install the necessary dependencies, run the following command:
```bash
pip install -r requirements.txt
```

## Usage
To run the application, execute the following command:
```bash
streamlit run main.py
```

## Configuration
The application allows users to configure various parameters via the sidebar:
- **Number of Sheets to Process**: Specify how many sheets from the Excel file to process.
- **Minimum Market Cap (Crore)**: Set the minimum Market Capitalization in Crore.
- **Maximum Market Cap (Crore)**: Set the maximum Market Capitalization in Crore.
- **Select Analysis Period**: Choose the periodicity for analysis (Monthly, Quarterly, Yearly).

## Data Preprocessing
The data preprocessing steps include:
1. Loading data from the specified number of sheets in the Excel file.
2. Checking if preprocessing is needed based on metadata and file modification time.
3. Preprocessing data by calculating various metrics such as earnings growth rate, volatility, revenue trend, and moving averages.
4. Cleaning data by converting columns to numeric, filling missing values, and filtering based on market cap.
5. Aggregating data to the selected periodicity.

## Visualization
The application provides various visualizations:
- **Feature Importance**: Bar chart showing the importance score of each feature.
- **Correlation Heatmap**: Heatmap displaying the correlation between features.
- **Distribution of Total Scores**: Histogram showing the distribution of total scores.
- **Market Cap Trend**: Line chart showing the average market capitalization over time.
- **Best Stocks Count by Ticker**: Bar chart showing the number of times each ticker was selected as the best.
- **Recommended Similar Stocks**: Scatter matrix of recommended similar stocks.

## Contributing
Contributions are welcome! Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

## License
This project is licensed. See the [LICENSE](LICENSE) file for more details.