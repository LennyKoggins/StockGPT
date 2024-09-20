# streamlit_app.py
import pandas as pd
import numpy as np
import logging
import os
import json
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from collections import Counter
import streamlit as st
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# ======== Setup Logging ========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# ======== Load Data ========
excel_file = 'stock_data_with_calculations.xlsx'
preprocessed_file_df = 'preprocessed_df.parquet'
preprocessed_file_df_periodic = 'preprocessed_df.parquet'
preprocessed_metadata = 'preprocessed_metadata.json'

# Define the sheets to process (modify the slice as needed)
sheets_to_process = []

# ======== Streamlit Configuration ========
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

st.title("📈 Stock Analysis and Prediction Dashboard")

# ======== Sidebar Configuration ========
st.sidebar.header("Configuration")

# Number of Sheets to Process
sheets_count_input = st.sidebar.number_input(
    "Number of Sheets to Process",
    min_value=1,
    step=1,
    value=15,
    help="Specify how many sheets from the Excel file to process."
)

# Market Cap Filter Inputs
st.sidebar.subheader("Market Capitalization Filter (Crore)")
min_market_cap = st.sidebar.number_input(
    "Minimum Market Cap (Crore)",
    min_value=0.0,
    value=500.0,
    step=50.0,
    format="%.2f",
    help="Set the minimum Market Capitalization in Crore."
)

max_market_cap = st.sidebar.number_input(
    "Maximum Market Cap (Crore)",
    min_value=0.0,
    value=1000.0,
    step=50.0,
    format="%.2f",
    help="Set the maximum Market Capitalization in Crore."
)

# Periodicity Selection
st.sidebar.subheader("Analysis Periodicity")
periodicity_options = {'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y'}
selected_periodicity = st.sidebar.selectbox("Select Analysis Period:", list(periodicity_options.keys()))
periodicity = periodicity_options[selected_periodicity]

st.sidebar.header("Data Loading")
if st.sidebar.button("Load Data"):
    logging.info("Load Data button clicked.")
    preprocess_needed = True
    current_sheets = []

    # Check if metadata exists
    if os.path.exists(preprocessed_metadata):
        try:
            with open(preprocessed_metadata, 'r') as meta_file:
                saved_metadata = json.load(meta_file)
                saved_sheets = saved_metadata.get('sheets_to_process', [])
                saved_excel_mtime = saved_metadata.get('excel_mtime', 0)
                saved_periodicity = saved_metadata.get('periodicity', 'M')
        except Exception as e:
            logging.error(f"Error loading metadata: {e}")
            st.error(f"Error loading metadata: {e}")
            saved_sheets = []
            saved_excel_mtime = 0
            saved_periodicity = 'M'
    else:
        saved_sheets = []
        saved_excel_mtime = 0
        saved_periodicity = 'M'

    # Get current Excel file modification time
    if os.path.exists(excel_file):
        current_excel_mtime = os.path.getmtime(excel_file)
    else:
        st.error(f"Excel file '{excel_file}' not found. Please ensure the file is in the correct directory.")
        st.stop()

    # Determine sheets to process based on user input
    try:
        xls = pd.ExcelFile(excel_file)
        all_sheet_names = xls.sheet_names
        if sheets_count_input <= len(all_sheet_names):
            sheets_current = all_sheet_names[:sheets_count_input]
        else:
            st.warning(f"Requested {sheets_count_input} sheets, but only {len(all_sheet_names)} sheets are available.")
            sheets_current = all_sheet_names
        logging.info(f"Sheets to process: {sheets_current}")
    except Exception as e:
        logging.error(f"Error reading Excel file: {e}")
        st.error(f"Error reading Excel file: {e}")
        st.stop()

    # Check if preprocessing is needed
    if (set(sheets_current).issubset(set(saved_sheets))) and (current_excel_mtime <= saved_excel_mtime) and (
            saved_periodicity == periodicity):
        try:
            logging.info("Loading preprocessed data from parquet files.")
            df = pd.read_parquet(preprocessed_file_df)
            df_periodic = pd.read_parquet(preprocessed_file_df_periodic)
            st.success(
                f"Preprocessed data loaded successfully from '{preprocessed_file_df}' and '{preprocessed_file_df_periodic}'.")
            preprocess_needed = False
        except Exception as e:
            logging.error(f"Error loading preprocessed data: {e}")
            st.error(f"Error loading preprocessed data: {e}")
            preprocess_needed = True
    else:
        preprocess_needed = True

    if preprocess_needed:
        logging.info("Preprocessed data not up-to-date or incomplete. Loading data from Excel file.")
        try:
            xls = pd.ExcelFile(excel_file)
            st.success(f"Excel file '{excel_file}' loaded successfully.")
        except FileNotFoundError:
            st.error(f"Excel file '{excel_file}' not found. Please ensure the file is in the correct directory.")
            st.stop()

        all_data = []

        for sheet_name in sheets_current:
            df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
            df_sheet['Ticker'] = sheet_name  # Add ticker name

            logging.info(f"Processing sheet: {sheet_name}")
            if df_sheet.empty:
                logging.warning(f"Sheet {sheet_name} is empty. Skipping.")
                continue

            # Ensure 'Date' column exists and is formatted properly
            if 'Date' in df_sheet.columns:
                df_sheet['Date'] = pd.to_datetime(df_sheet['Date'])
                df_sheet.sort_values('Date', inplace=True)
            else:
                logging.warning(f"'Date' column missing in sheet {sheet_name}. Skipping date processing.")
                continue

            # Compute earnings growth rate
            if 'earnings' in df_sheet.columns:
                df_sheet['earnings_growth_rate'] = df_sheet['earnings'].pct_change()

            # Calculate volatility of stock returns
            if 'Close' in df_sheet.columns:
                df_sheet['returns'] = df_sheet['Close'].pct_change()
                df_sheet['volatility'] = df_sheet['returns'].rolling(window=20).std()

            # Calculate revenue trend using linear regression
            if 'revenue' in df_sheet.columns:
                df_sheet = df_sheet.dropna(subset=['revenue'])
                if len(df_sheet) > 1:
                    df_sheet['TimeIndex'] = np.arange(len(df_sheet))
                    X_trend = df_sheet[['TimeIndex']]
                    y_trend = df_sheet['revenue']
                    model_trend = LinearRegression()
                    model_trend.fit(X_trend, y_trend)
                    df_sheet['revenue_trend'] = model_trend.coef_[0]
            else:
                df_sheet['revenue_trend'] = np.nan

            # Momentum indicators (Moving Averages)
            if 'Close' in df_sheet.columns:
                df_sheet['50_day_MA'] = df_sheet['Close'].rolling(window=50).mean()
                df_sheet['200_day_MA'] = df_sheet['Close'].rolling(window=200).mean()

            all_data.append(df_sheet)

        # Concatenate all stocks' metrics into a single DataFrame
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            logging.info(f"Data loaded successfully. Total records: {len(df)}")
            st.success(f"Data loaded successfully. Total records: {len(df)}")
        else:
            logging.error("No data available after processing sheets.")
            st.error("No data available after processing sheets.")
            st.stop()

        # ======== Data Cleaning ========
        numeric_columns = [
            'Market_Cap', 'Enterprise_Value', 'Price_To_Book', 'Price_To_Sales',
            'PE_Ratio', 'PEG_Ratio', '52WeekChange', 'Dividend_Yield', 'Payout_Ratio',
            'Beta', 'profitMargins', 'earningsGrowth', 'revenueGrowth', 'grossMargins',
            'ebitdaMargins', 'operatingMargins', 'enterpriseToRevenue', 'enterpriseToEbitda',
            'auditRisk', 'boardRisk', 'compensationRisk', 'shareHolderRightsRisk',
            'avg_earnings_growth', 'volatility', 'revenue_trend', '50_day_MA', '200_day_MA'
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(axis=1, how='all', inplace=True)
        numeric_columns = [col for col in numeric_columns if col in df.columns]

        df.fillna(df.median(numeric_only=True), inplace=True)

        # ======== Market Cap Conversion and Filtering ========
        # Convert Market Cap to Crores
        if 'Market_Cap' in df.columns:
            df['Market_Cap'] = df['Market_Cap'] / 10000000
            logging.info("Market Cap converted to crores.")
            st.info("Market Cap converted to crores.")
        else:
            logging.error("'Market_Cap' column is missing from the data.")
            st.error("'Market_Cap' column is missing from the data.")
            st.stop()

        # Filter: Market Cap based on user input
        df = df[(df['Market_Cap'] >= min_market_cap) & (df['Market_Cap'] <= max_market_cap)]

        if df.empty:
            logging.error(
                f"No stocks found with Market Cap between {min_market_cap} and {max_market_cap} crore. Exiting.")
            st.error(
                f"No stocks found with Market Cap between {min_market_cap} and {max_market_cap} crore. Please adjust your data or filters.")
            st.stop()

        # ======== Aggregate Data to Selected Periodicity ========
        df['Period'] = df['Date'].dt.to_period(periodicity)
        df_periodic = df.copy()
        df_periodic['Period'] = df_periodic['Date'].dt.to_period(periodicity)

        # For aggregation, take the last available data for each ticker in each period
        df_periodic = df_periodic.sort_values('Date').groupby(['Ticker', 'Period']).last().reset_index()

        # Convert Period back to datetime for consistency
        df_periodic['Period'] = df_periodic['Period'].dt.to_timestamp()

        # Save preprocessed data
        try:
            df.to_parquet(preprocessed_file_df)
            df_periodic.to_parquet(preprocessed_file_df_periodic)
            logging.info(f"Preprocessed data saved to '{preprocessed_file_df}' and '{preprocessed_file_df_periodic}'.")
            st.success(f"Preprocessed data saved to '{preprocessed_file_df}' and '{preprocessed_file_df_periodic}'.")
        except Exception as e:
            logging.error(f"Error saving preprocessed data: {e}")
            st.error(f"Error saving preprocessed data: {e}")
            st.stop()

        # Save metadata
        metadata = {
            'sheets_to_process': sheets_current,
            'excel_mtime': current_excel_mtime,
            'periodicity': periodicity
        }
        try:
            with open(preprocessed_metadata, 'w') as meta_file:
                json.dump(metadata, meta_file)
            logging.info(f"Metadata saved to '{preprocessed_metadata}'.")
        except Exception as e:
            logging.error(f"Error saving metadata: {e}")
            st.error(f"Error saving metadata: {e}")
            st.stop()

    # ======== Rolling Optimization ========
    metrics_to_maximize = ['Market_Cap', 'avg_earnings_growth', 'revenue_trend', 'earningsGrowth',
                           'revenueGrowth', 'grossMargins', 'ebitdaMargins', 'operatingMargins']
    metrics_to_minimize = ['PE_Ratio', 'Price_To_Book', 'Price_To_Sales', 'auditRisk', 'volatility']

    metrics_to_maximize = [metric for metric in metrics_to_maximize if metric in df_periodic.columns]
    metrics_to_minimize = [metric for metric in metrics_to_minimize if metric in df_periodic.columns]

    df_periodic.sort_values('Period', inplace=True)
    df_periodic.reset_index(drop=True, inplace=True)

    best_stocks_history = []

    lambda_param = 0.5
    gamma_param = 0.5  # Parameter for volatility weighting

    unique_periods = df_periodic['Period'].sort_values().unique()

    for current_period in unique_periods:
        df_past = df_periodic[df_periodic['Period'] <= current_period]

        if df_past.empty:
            continue

        df_past = df_past.dropna(subset=metrics_to_maximize + metrics_to_minimize)

        if df_past.empty:
            continue

        scaler_max = MinMaxScaler()
        df_max_scaled = pd.DataFrame(scaler_max.fit_transform(df_past[metrics_to_maximize]),
                                     columns=[f"{col}_scaled" for col in metrics_to_maximize])

        scaler_min = MinMaxScaler()
        df_min_scaled = pd.DataFrame(scaler_min.fit_transform(df_past[metrics_to_minimize]),
                                     columns=[f"{col}_scaled" for col in metrics_to_minimize])
        df_min_scaled = 1 - df_min_scaled  # Invert to make lower values better

        df_scaled = pd.concat([df_max_scaled, df_min_scaled], axis=1)
        df_scaled.reset_index(drop=True, inplace=True)
        df_past.reset_index(drop=True, inplace=True)

        scaled_metrics = df_scaled.columns.tolist()


        def complex_objective(weights):
            weighted_scores = df_scaled[scaled_metrics].dot(weights)
            mean_score = weighted_scores.mean()
            variance_score = weighted_scores.var()
            if 'volatility_scaled' in scaled_metrics:
                volatility_score = df_past['volatility'].mean()
            else:
                volatility_score = 0
            diversity_pen = df_past['Ticker'].nunique() / len(df_past)
            return -(mean_score - lambda_param * variance_score - gamma_param * volatility_score + 0.1 * diversity_pen)


        bounds = [(0.05, 0.5)] * len(scaled_metrics)
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        initial_weights = np.array([1 / len(scaled_metrics)] * len(scaled_metrics))

        result = minimize(complex_objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            optimized_weights = result.x
        else:
            optimized_weights = np.array([1 / len(scaled_metrics)] * len(scaled_metrics))

        df_scaled['Total_Score'] = df_scaled[scaled_metrics].dot(optimized_weights)
        df_past['Total_Score'] = df_scaled['Total_Score']

        best_stock = df_past.loc[df_past['Total_Score'].idxmax()]
        best_stocks_history.append({
            'Period': current_period,
            'Best_Ticker': best_stock['Ticker'],
            'Total_Score': best_stock['Total_Score']
        })

        logging.info(
            f"Period: {current_period.strftime('%Y-%m')} - Best Stock: {best_stock['Ticker']} with Score: {best_stock['Total_Score']:.4f}")

    best_stocks_df = pd.DataFrame(best_stocks_history)

    # ======== Building the Predictive Model ========
    df_periodic = df_periodic.merge(best_stocks_df, on='Period', how='left', suffixes=('', '_Best'))

    # Assuming periodic data, shift by 2 periods for the target (e.g., next 6 months if quarterly)
    periods_to_shift = 2 if periodicity == 'Q' else 6
    df_periodic['Target_Best_Ticker'] = df_periodic.groupby('Ticker')['Best_Ticker'].shift(-periods_to_shift)

    df_model = df_periodic.dropna(subset=['Target_Best_Ticker'])

    feature_columns = [
        'Market_Cap', 'avg_earnings_growth', 'volatility', 'revenue_trend',
        'earningsGrowth', 'revenueGrowth', 'grossMargins', 'ebitdaMargins',
        'operatingMargins', 'PE_Ratio', 'Price_To_Book', 'Price_To_Sales',
        'auditRisk', '50_day_MA', '200_day_MA'
    ]

    feature_columns = [col for col in feature_columns if col in df_model.columns]

    X = df_model[feature_columns]
    y = df_model['Target_Best_Ticker']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # ======== Making Predictions for the Next Periods ========
    latest_period = df_periodic['Period'].max()
    df_latest = df_periodic[df_periodic['Period'] == latest_period]

    if df_latest.empty:
        logging.error("No latest data available for prediction.")
        st.error("No latest data available for prediction.")
        st.stop()

    X_latest = df_latest[feature_columns]

    X_latest.fillna(X_latest.median(), inplace=True)

    y_latest_pred = model.predict(X_latest)
    best_ticker_pred = le.inverse_transform(y_latest_pred)

    ticker_counts = Counter(best_ticker_pred)
    predicted_best_ticker = ticker_counts.most_common(1)[0][0]

    if periodicity == 'Q':
        prediction_date = latest_period + pd.DateOffset(months=6)
    elif periodicity == 'M':
        prediction_date = latest_period + pd.DateOffset(months=6)
    elif periodicity == 'Y':
        prediction_date = latest_period + pd.DateOffset(years=1)
    else:
        prediction_date = latest_period

    # ======== Identify Top Performing Stocks ========
    top_n = 10  # You can adjust this number
    top_performers = best_stocks_df['Best_Ticker'].value_counts().head(top_n).index.tolist()

    # Extract feature data for top performers
    df_top_performers = df_periodic[df_periodic['Best_Ticker'].isin(top_performers)]

    # ======== Trend Feature Extraction ========
    trend_features = [
        '50_day_MA', '200_day_MA', 'revenue_trend', 'earningsGrowth',
        'revenueGrowth', 'grossMargins', 'ebitdaMargins', 'operatingMargins'
    ]

    # Ensure all trend features are present
    trend_features = [col for col in trend_features if col in df_top_performers.columns]

    # Handle missing values
    df_top_performers = df_top_performers.dropna(subset=trend_features)

    # Pivot the data to have one row per ticker with trend feature averages
    df_trend = df_top_performers.groupby('Ticker')[trend_features].mean().reset_index()

    # ======== Similarity Computation ========
    # Scale trend features
    scaler_trend = MinMaxScaler()
    trend_scaled = scaler_trend.fit_transform(df_trend[trend_features])

    # Compute cosine similarity matrix
    similarity_matrix_trend = cosine_similarity(trend_scaled)

    # Create a DataFrame for similarity scores
    similarity_df_trend = pd.DataFrame(similarity_matrix_trend, index=df_trend['Ticker'], columns=df_trend['Ticker'])

    # ======== Recommend Similar Stocks ========
    similar_stocks = set()
    for ticker in top_performers:
        if ticker not in similarity_df_trend.index:
            continue
        # Get similarity scores for the ticker
        sim_scores = similarity_df_trend.loc[ticker].sort_values(ascending=False)
        # Exclude the ticker itself
        sim_scores = sim_scores.drop(labels=[ticker])
        # Select top similar stocks (e.g., top 5)
        top_similar = sim_scores.head(5).index.tolist()
        similar_stocks.update(top_similar)

    # Remove top performers from similar stocks if present
    similar_stocks = similar_stocks - set(top_performers)

    # ======== Optimize Similar Stocks Selection ========
    recommended_optimized_similar_stocks = []
    if similar_stocks:
        # Filter the original periodic data to include only similar stocks
        df_similar = df_periodic[df_periodic['Ticker'].isin(similar_stocks)]

        # Ensure there is sufficient data for optimization
        if not df_similar.empty:
            # Re-scale metrics for optimization
            metrics_to_maximize_sim = ['Market_Cap', 'avg_earnings_growth', 'revenue_trend', 'earningsGrowth',
                                       'revenueGrowth', 'grossMargins', 'ebitdaMargins', 'operatingMargins']
            metrics_to_minimize_sim = ['PE_Ratio', 'Price_To_Book', 'Price_To_Sales', 'auditRisk', 'volatility']

            metrics_to_maximize_sim = [metric for metric in metrics_to_maximize_sim if metric in df_similar.columns]
            metrics_to_minimize_sim = [metric for metric in metrics_to_minimize_sim if metric in df_similar.columns]

            df_similar = df_similar.dropna(subset=metrics_to_maximize_sim + metrics_to_minimize_sim)

            if not df_similar.empty:
                scaler_max_sim = MinMaxScaler()
                df_max_scaled_sim = pd.DataFrame(scaler_max_sim.fit_transform(df_similar[metrics_to_maximize_sim]),
                                                 columns=[f"{col}_scaled" for col in metrics_to_maximize_sim])

                scaler_min_sim = MinMaxScaler()
                df_min_scaled_sim = pd.DataFrame(scaler_min_sim.fit_transform(df_similar[metrics_to_minimize_sim]),
                                                 columns=[f"{col}_scaled" for col in metrics_to_minimize_sim])
                df_min_scaled_sim = 1 - df_min_scaled_sim  # Invert to make lower values better

                df_scaled_sim = pd.concat([df_max_scaled_sim, df_min_scaled_sim], axis=1)
                df_scaled_sim.reset_index(drop=True, inplace=True)
                df_similar.reset_index(drop=True, inplace=True)

                scaled_metrics_sim = df_scaled_sim.columns.tolist()


                def similar_objective(weights):
                    weighted_scores = df_scaled_sim[scaled_metrics_sim].dot(weights)
                    mean_score = weighted_scores.mean()
                    variance_score = weighted_scores.var()
                    if 'volatility_scaled' in scaled_metrics_sim:
                        volatility_score = df_similar['volatility'].mean()
                    else:
                        volatility_score = 0
                    diversity_pen = df_similar['Ticker'].nunique() / len(df_similar)
                    # Add similarity factor to the objective
                    # Higher similarity to top performers should increase the score
                    similarity_scores = df_similar['Ticker'].apply(lambda x: similarity_df_trend.loc[
                        x, top_performers].max() if x in similarity_df_trend.columns else 0)
                    similarity_pen = similarity_scores.mean()
                    return -(
                                mean_score - lambda_param * variance_score - gamma_param * volatility_score + 0.1 * diversity_pen + 0.2 * similarity_pen)


                bounds_sim = [(0.05, 0.5)] * len(scaled_metrics_sim)
                constraints_sim = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
                initial_weights_sim = np.array([1 / len(scaled_metrics_sim)] * len(scaled_metrics_sim))

                result_sim = minimize(similar_objective, initial_weights_sim, method='SLSQP', bounds=bounds_sim,
                                      constraints=constraints_sim)

                if result_sim.success:
                    optimized_weights_sim = result_sim.x
                else:
                    optimized_weights_sim = np.array([1 / len(scaled_metrics_sim)] * len(scaled_metrics_sim))

                df_scaled_sim['Total_Score'] = df_scaled_sim[scaled_metrics_sim].dot(optimized_weights_sim)
                df_similar['Total_Score'] = df_scaled_sim['Total_Score']

                # Select the top similar stocks based on the optimized total score
                top_similar_stocks = df_similar.sort_values('Total_Score', ascending=False).head(10)
                recommended_optimized_similar_stocks = top_similar_stocks['Ticker'].unique().tolist()

                logging.info(f"Recommended Optimized Similar Stocks: {recommended_optimized_similar_stocks}")
            else:
                logging.warning("No sufficient data available for similar stocks optimization.")
        else:
            logging.warning("No similar stocks found after filtering.")
    else:
        logging.info("No similar stocks identified based on trends.")

    # ======== Visualization ========
    st.header("📊 Visualizations")

    # Classification Report
    st.subheader("Classification Report")
    st.dataframe(report_df.style.background_gradient(cmap='viridis'))

    # Historical Best Stocks Over Time
    st.subheader("Historical Best Stocks Over Time")
    fig1 = px.line(best_stocks_df, x='Period', y='Best_Ticker',
                   title='Historical Best Stocks',
                   labels={'Period': 'Date', 'Best_Ticker': 'Best Ticker'},
                   markers=True)
    fig1.update_layout(xaxis=dict(tickformat="%Y-%m"))
    st.plotly_chart(fig1, use_container_width=True)

    # Feature Importance Plot
    st.subheader("Feature Importance from Random Forest Model")
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    fig2 = px.bar(feature_importances,
                  x=feature_importances.values,
                  y=feature_importances.index,
                  orientation='h',
                  title='Feature Importance',
                  labels={'x': 'Importance Score', 'y': 'Features'},
                  color=feature_importances.values,
                  color_continuous_scale='magma')
    fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig2, use_container_width=True)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap of Features")
    corr = df_model[feature_columns].corr()
    fig3 = px.imshow(corr,
                     text_auto=True,
                     aspect="auto",
                     color_continuous_scale='Viridis',
                     title='Feature Correlation Heatmap')
    st.plotly_chart(fig3, use_container_width=True)

    # Distribution of Total Scores
    st.subheader("Distribution of Total Scores")
    fig4 = px.histogram(df_scaled, x='Total_Score', nbins=50,
                        title='Distribution of Total Scores',
                        labels={'Total_Score': 'Total Score'},
                        opacity=0.75)
    st.plotly_chart(fig4, use_container_width=True)

    # Periodic Market Cap Trend
    st.subheader(f"{selected_periodicity} Market Cap Trend")
    market_cap_trend = df_periodic.groupby('Period')['Market_Cap'].mean().reset_index()
    fig5 = px.line(market_cap_trend, x='Period', y='Market_Cap',
                   title='Average Market Capitalization Over Time',
                   labels={'Period': 'Date', 'Market_Cap': 'Average Market Cap (Crores)'},
                   markers=True)
    fig5.update_layout(xaxis=dict(tickformat="%Y-%m"))
    st.plotly_chart(fig5, use_container_width=True)

    # Best Stocks Count by Ticker
    st.subheader("Best Stocks Count by Ticker")
    ticker_counts_df = best_stocks_df['Best_Ticker'].value_counts().reset_index()
    ticker_counts_df.columns = ['Ticker', 'Count']
    fig6 = px.bar(ticker_counts_df, x='Ticker', y='Count',
                  title='Number of Times Each Ticker Was Selected as Best',
                  labels={'Ticker': 'Ticker', 'Count': 'Selection Count'},
                  color='Count',
                  color_continuous_scale='Blues')
    st.plotly_chart(fig6, use_container_width=True)

    # ======== Recommended Similar Stocks Visualization ========
    st.subheader("📌 Recommended Similar Stocks")
    if recommended_optimized_similar_stocks:
        recommended_df = df[df['Ticker'].isin(recommended_optimized_similar_stocks)]

        # Select relevant features for visualization
        vis_features = ['Market_Cap', 'earningsGrowth', 'revenueGrowth', 'volatility', 'PE_Ratio']
        vis_features = [col for col in vis_features if col in recommended_df.columns]

        fig7 = px.scatter_matrix(recommended_df, dimensions=vis_features, color='Ticker',
                                 title='Scatter Matrix of Recommended Similar Stocks',
                                 labels={col: col.replace('_', ' ') for col in vis_features},
                                 height=800)
        st.plotly_chart(fig7, use_container_width=True)

        # Display the list of recommended optimized similar stocks
        st.write("### Top Recommended Optimized Similar Stocks:")
        st.write(", ".join(recommended_optimized_similar_stocks))
    else:
        st.info("No similar stocks found to recommend.")

    # ======== Output ========
    st.header("📋 Results")

    st.subheader("Classification Report")
    st.write(report_df)

    st.subheader("Predicted Best Stock for the Next Periods")
    st.write(f"**{predicted_best_ticker}**")
    st.write(f"**Prediction Date:** {prediction_date.strftime('%Y-%m')}")

    # Optionally, display the best_stocks_history
    with st.expander("View Best Stocks History"):
        st.dataframe(best_stocks_df)

    # ======== Recommended Similar Stocks Output ========
    st.header("🔍 Recommended Similar Stocks")
    st.write(
        "Based on the top-performing stocks identified through optimization, the following stocks have been recommended due to their similar trend characteristics and potential to perform well:")

    if recommended_optimized_similar_stocks:
        recommended_data = df[df['Ticker'].isin(recommended_optimized_similar_stocks)]
        st.dataframe(recommended_data[['Ticker'] + trend_features].drop_duplicates(subset=['Ticker']))
    else:
        st.write("No similar stocks found to recommend.")

