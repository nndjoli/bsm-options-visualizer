import numpy as np
import altair as alt
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from models_class import PricingModels

# Page configuration
st.set_page_config(
    page_title="Black-Scholes & Merton Options Visualizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title
st.markdown("# Black-Scholes & Merton Options Visualizer")

st.markdown(
    """
    This Streamlit app builds a two‑tab dashboard for exploring both real and theoretical option data. 
    It allows users to view option prices, Greeks, and payoffs for both Call and Put options. 
    Both the “Real underlying data” and “Theoretical parameters” tabs automatically adjust the pricing 
    model (Black‑Scholes or Black‑Scholes‑Merton) based on the dividend yield, then display a one‑year 
    price and volume chart (real data tab) or accept user‑defined inputs (theoretical tab) to plot 
    option prices, payoffs, and Greeks. Each tab also includes an interactive heatmap showing how the 
    selected metric varies with any two chosen parameters. 

    """
)
st.markdown("###")

# Create tabs
real_data_tab, theoretical_tab = st.tabs(
    ["Real underlying data", "Theoretical parameters"]
)

# Mapping of metrics for Call and Put
metrics_mapping = {
    "Call": {
        "Price": "c",
        "Payoff": "call_payoff",
        "Delta": "delta_c",
        "Gamma": "gamma",
        "Vega": "vega",
        "Theta": "theta_c",
        "Rho": "rho_c",
    },
    "Put": {
        "Price": "p",
        "Payoff": "put_payoff",
        "Delta": "delta_p",
        "Gamma": "gamma",
        "Vega": "vega",
        "Theta": "theta_p",
        "Rho": "rho_p",
    },
}

# --- Real underlying data tab ---
with real_data_tab:
    st.markdown(
        '### <span style="text-decoration: underline">Underlying Stock Overview:</span>',
        unsafe_allow_html=True,
    )

    # Columns for info and charts
    info_col, chart_col = st.columns([0.25, 0.75], gap="large")

    # Stock ticker input
    ticker_symbol = info_col.text_input("Symbol:", "NVDA", key="stock_ticker")

    # Retrieve stock data
    option_pricer = PricingModels(ticker_symbol, 100, 0.03, 1)
    current_price = float(option_pricer.s)
    dividend_yield = float(option_pricer.q)
    realized_volatility = float(option_pricer.sigma)

    # Display key info
    info_col.markdown(f"*Current Price:* {current_price:.2f}")
    info_col.markdown(f"*Dividend Yield:* {dividend_yield*100:.2f}%")
    info_col.markdown(f"*Realized Volatility (1Y):* {realized_volatility:.2f}%")

    # 1-year history
    chart_col.markdown(
        '<div style="text-align:center;"><span style="text-decoration: underline;">One Year History -- Daily Close & Volume</span></div>',
        unsafe_allow_html=True,
    )
    one_year_history_df = option_pricer.ticker_history_1y
    currency = option_pricer.ticker.info.get("currency", "USD")

    price_chart = (
        alt.Chart(one_year_history_df.reset_index())
        .mark_line(color="blue")
        .encode(
            x="Date:T",
            y=alt.Y("Close:Q", title=f"Close ({currency})"),
            tooltip=[
                "Date:T",
                "Open:Q",
                "High:Q",
                "Low:Q",
                "Close:Q",
                "Volume:Q",
            ],
        )
    )

    volume_chart = (
        alt.Chart(one_year_history_df.reset_index())
        .mark_bar(opacity=0.4, color="orange")
        .encode(
            x="Date:T",
            y=alt.Y("Volume:Q", title="Volume"),
            tooltip=["Date:T", "Volume:Q"],
        )
        .interactive()
    )

    combined_chart = (
        alt.layer(
            price_chart,
            volume_chart.encode(
                y=alt.Y("Volume:Q", scale=alt.Scale(zero=True))
            ),
        )
        .resolve_scale(y="independent")
        .properties(height=225)
    )

    chart_col.altair_chart(combined_chart, use_container_width=True)

    # Description and additional metrics
    description_tab, additional_metrics_tab = st.tabs(
        ["Company Description", "Additional metrics"]
    )

    with description_tab:
        st.markdown(
            option_pricer.ticker.info.get(
                "longBusinessSummary", "No description available"
            )
        )

    with additional_metrics_tab:
        company_info = option_pricer.ticker.info
        metrics_cols = st.columns(4)

        metrics_cols[0].metric(
            "Day Open", f"${company_info.get('open', 0):.2f}"
        )
        metrics_cols[1].metric(
            "Day High", f"${company_info.get('dayHigh', 0):.2f}"
        )
        metrics_cols[2].metric(
            "Day Low", f"${company_info.get('dayLow', 0):.2f}"
        )
        metrics_cols[3].metric(
            "Market Cap", f"{company_info.get('marketCap', 0)/1e9:.2f} B"
        )

        metrics_cols[0].metric(
            "P/E Ratio", f"{company_info.get('trailingPE', 0):.2f}"
        )
        metrics_cols[1].metric(
            "52W High", f"${company_info.get('fiftyTwoWeekHigh', 0):.2f}"
        )
        metrics_cols[2].metric(
            "52W Low", f"${company_info.get('fiftyTwoWeekLow', 0):.2f}"
        )

    st.markdown("#")
    st.markdown(
        '### <span style="text-decoration: underline">Option Price, Greeks & Payoff:</span>',
        unsafe_allow_html=True,
    )

    # --- Option Parameters ---
    input_col, plot_col = st.columns([0.25, 0.75], gap="large")

    with input_col:
        contract_type = st.selectbox(
            "Contract Type:", ["Call", "Put"], key="stock_type"
        )
        strike_price = float(
            st.number_input(
                "Strike Price:",
                min_value=0.0,
                value=round(current_price, -1),
                step=2.5,
                key="stock_k",
            )
        )
        risk_free_rate = float(
            st.number_input(
                "Risk-Free Rate:",
                min_value=0.0,
                value=0.03,
                step=0.005,
                key="stock_r",
            )
        )
        days_to_expiry = float(
            st.number_input(
                "Days to Expiration:",
                min_value=1,
                value=30,
                step=1,
                key="stock_dte",
            )
        )
        volatility = float(
            st.number_input(
                "Volatility:",
                min_value=0.0,
                max_value=2.0,
                value=realized_volatility,
                step=0.01,
                key="stock_sigma",
            )
        )
        model_name = (
            "Black-Scholes-Merton" if dividend_yield != 0 else "Black-Scholes"
        )
        st.markdown(f"*Model:* {model_name}")

    time_to_maturity = days_to_expiry / 365
    theoretical_pricer = PricingModels(
        current_price,
        strike_price,
        risk_free_rate,
        time_to_maturity,
        volatility,
        dividend_yield,
    )

    expiration_date = (
        pd.to_datetime("now") + pd.Timedelta(days=days_to_expiry)
    ).strftime("%B %d, %Y")
    plot_col.markdown(
        f'<div style="text-align:center;"><span style="text-decoration: underline">{ticker_symbol} @{current_price:.2f} '
        f"{contract_type} Strike {strike_price} {expiration_date} (DTE {days_to_expiry})</span></div>",
        unsafe_allow_html=True,
    )

    # Scenarios for Underlying Price and Strike Price
    underlying_price_range = np.linspace(
        0.5 * current_price, 1.5 * current_price, 100
    )
    strike_price_range = np.linspace(
        0.5 * current_price, 1.5 * current_price, 100
    )

    price_scenarios = [
        PricingModels(
            float(price_sample),
            strike_price,
            risk_free_rate,
            time_to_maturity,
            volatility,
            dividend_yield,
        ).options
        for price_sample in underlying_price_range
    ]

    strike_scenarios = [
        PricingModels(
            current_price,
            float(strike_sample),
            risk_free_rate,
            time_to_maturity,
            volatility,
            dividend_yield,
        ).options
        for strike_sample in strike_price_range
    ]

    df_underlying_scenarios = pd.DataFrame(price_scenarios)
    df_underlying_scenarios["call_payoff"] = np.maximum(
        0, df_underlying_scenarios["s"] - strike_price
    )
    df_underlying_scenarios["put_payoff"] = np.maximum(
        0, strike_price - df_underlying_scenarios["s"]
    )

    df_strike_scenarios = pd.DataFrame(strike_scenarios)
    df_strike_scenarios["call_payoff"] = np.maximum(
        0, current_price - df_strike_scenarios["k"]
    )
    df_strike_scenarios["put_payoff"] = np.maximum(
        0, df_strike_scenarios["k"] - current_price
    )

    # Select variables for the chart
    selected_x_axis = input_col.selectbox(
        "X-Axis:", ["Underlying Price", "Strike Price"], key="stock_xvar"
    )
    selected_y_group = input_col.selectbox(
        "Y-Axis:", ["Price & Payoff", "Greeks"], key="stock_ygrp"
    )

    selected_greeks = {}
    if selected_y_group == "Greeks":
        greek_checkbox_cols = plot_col.columns(5)
        for idx, greek_name in enumerate(
            ["Delta", "Gamma", "Vega", "Theta", "Rho"]
        ):
            selected_greeks[greek_name] = greek_checkbox_cols[idx].checkbox(
                greek_name, True, key=f"stock_{greek_name}"
            )

    # Choose DataFrame and Y-Axis
    if selected_x_axis == "Underlying Price":
        df_to_plot = df_underlying_scenarios
        x_values = underlying_price_range
        reference_value = current_price
        x_axis_label = f"Underlying Price (Cur: {current_price:.2f})"
    else:
        df_to_plot = df_strike_scenarios
        x_values = strike_price_range
        reference_value = strike_price
        x_axis_label = f"Strike Price (Cur: {strike_price:.2f})"

    # Build Plotly chart
    price_payoff_fig = go.Figure()

    if selected_y_group == "Price & Payoff":
        chart_height = 512.5
        option_price_key = metrics_mapping[contract_type]["Price"]
        price_payoff_fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df_to_plot[option_price_key],
                mode="lines",
                name="Price",
            )
        )

        payoff_key = "call_payoff" if contract_type == "Call" else "put_payoff"
        price_payoff_fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df_to_plot[payoff_key],
                mode="lines",
                name="Payoff",
            )
        )
    else:
        chart_height = 465
        for greek_name, is_checked in selected_greeks.items():
            if is_checked:
                metric_key = metrics_mapping[contract_type][greek_name]
                price_payoff_fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=df_to_plot[metric_key],
                        mode="lines",
                        name=greek_name,
                    )
                )

    price_payoff_fig.add_vline(
        x=reference_value,
        line=dict(color="red", dash="dash"),
        name="Current Price",
    )

    price_payoff_fig.add_vline(
        x=strike_price,
        line=dict(color="green", dash="dash"),
        name="Strike Price",
    )

    price_payoff_fig.update_layout(
        xaxis_title=x_axis_label,
        yaxis_title=selected_y_group,
        hovermode="x",
        margin=dict(t=0, b=0),
        height=chart_height,
    )
    plot_col.plotly_chart(price_payoff_fig)

    # Display numeric metrics
    metric_cols_top = plot_col.columns(6)
    option_price_key = "c" if contract_type == "Call" else "p"
    metric_cols_top[0].metric(
        "Price", f"{theoretical_pricer.options[option_price_key]:.4f}"
    )

    for idx, greek_name in enumerate(
        ["Delta", "Gamma", "Vega", "Theta", "Rho"], start=1
    ):
        metric_key = metrics_mapping[contract_type][greek_name]
        metric_cols_top[idx].metric(
            greek_name, f"{theoretical_pricer.options[metric_key]:.6f}"
        )

    st.markdown("#")
    st.markdown(
        '### <span style="text-decoration: underline">Price & Greeks Heatmap Visualization:</span> ',
        unsafe_allow_html=True,
    )

    heatmap_input_col, heatmap_chart_col = st.columns([0.25, 0.75], gap="large")

    with heatmap_input_col:
        # Metric selection
        selected_metric = heatmap_input_col.selectbox(
            "Metric:",
            ["Price", "Delta", "Gamma", "Vega", "Theta", "Rho"],
            key="stock_heatmap_metric",
        )

        # X and Y axis selection among variable parameters
        variable_list = [
            "Underlying Price",
            "Strike Price",
            "Time to Expiration",
            "Volatility",
        ]
        x_var = heatmap_input_col.selectbox(
            "X-Axis:", variable_list, key="stock_heatmap_x"
        )
        y_var = heatmap_input_col.selectbox(
            "Y-Axis:",
            [v for v in variable_list if v != x_var],
            key="stock_heatmap_y",
        )

        # Grid definition
        def make_grid(var):
            if var == "Underlying Price":
                return np.linspace(0.5 * current_price, 1.5 * current_price, 50)
            if var == "Strike Price":
                return np.arange(
                    0.25 * current_price, 1.75 * current_price + 1, 10
                )
            if var == "Time to Expiration":
                return np.arange(0.25, 2.01, 0.25)
            # Volatility
            return np.linspace(0.01, 1.0, 50)

        x_vals = make_grid(x_var)
        y_vals = make_grid(y_var)

        with heatmap_chart_col:
            st.markdown(
                f'<div style="text-align:center;"><span style="text-decoration: underline">'
                f"{ticker_symbol} | S={current_price:.2f}, K={strike_price:.2f}, r={risk_free_rate:.4f}, σ={volatility:.2f}, q={dividend_yield:.2f}, T={time_to_maturity:.2f} | "
                f"{contract_type} {selected_metric} Heatmap vs {x_var} & {y_var}"
                "</span></div>",
                unsafe_allow_html=True,
            )
            # Build Z matrix
            Z = np.zeros((len(y_vals), len(x_vals)))

            for i, y in enumerate(y_vals):
                for j, x in enumerate(x_vals):
                    # Assign parameters
                    S = (
                        x
                        if x_var == "Underlying Price"
                        else (
                            y if y_var == "Underlying Price" else current_price
                        )
                    )
                    K = (
                        x
                        if x_var == "Strike Price"
                        else (y if y_var == "Strike Price" else strike_price)
                    )
                    T = (
                        x
                        if x_var == "Time to Expiration"
                        else (
                            y
                            if y_var == "Time to Expiration"
                            else time_to_maturity
                        )
                    )
                    sigma = (
                        x
                        if x_var == "Volatility"
                        else (y if y_var == "Volatility" else volatility)
                    )

                    pr = PricingModels(
                        float(S),
                        float(K),
                        risk_free_rate,
                        float(T),
                        float(sigma),
                        dividend_yield,
                    )
                    key = metrics_mapping[contract_type][selected_metric]
                    Z[i, j] = pr.options[key]

            # Display heatmap
            hm = go.Figure(
                data=go.Heatmap(x=x_vals, y=y_vals, z=Z, coloraxis="coloraxis")
            )

            hm.update_layout(
                xaxis_title=x_var,
                yaxis_title=y_var,
                coloraxis={"colorscale": "Viridis"},
                margin=dict(t=0, b=0),
                height=550,
            )

            heatmap_chart_col.plotly_chart(hm)

with theoretical_tab:
    # Section Header
    st.markdown(
        '### <span style="text-decoration: underline">Option Price, Greeks & Payoff:</span>',
        unsafe_allow_html=True,
    )

    # Input and plot columns
    th_input_col, th_plot_col = st.columns([0.25, 0.75], gap="large")

    # Theoretical inputs
    with th_input_col:
        contract_type_th = st.selectbox(
            "Contract Type:", ["Call", "Put"], key="th_type"
        )
        underlying_price_th = float(
            st.number_input(
                "Underlying Price:",
                min_value=0.0,
                value=100.0,
                step=1.0,
                key="th_s",
            )
        )
        strike_price_th = float(
            st.number_input(
                "Strike Price:",
                min_value=0.0,
                value=round(underlying_price_th, -1),
                step=2.5,
                key="th_k",
            )
        )
        risk_free_rate_th = float(
            st.number_input(
                "Risk-Free Rate:",
                min_value=0.0,
                value=0.03,
                step=0.005,
                key="th_r",
            )
        )
        volatility_th = float(
            st.number_input(
                "Volatility:",
                min_value=0.0,
                max_value=2.0,
                value=0.2,
                step=0.01,
                key="th_sigma",
            )
        )
        dividend_yield_th = float(
            st.number_input(
                "Dividend Yield:",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                key="th_q",
            )
        )
        days_to_expiry_th = float(
            st.number_input(
                "Days to Expiration:",
                min_value=1,
                value=30,
                step=1,
                key="th_dte",
            )
        )
        model_name_th = (
            "Black‑Scholes‑Merton"
            if dividend_yield_th != 0
            else "Black‑Scholes"
        )
        th_input_col.markdown(f"*Model:* {model_name_th}")

        # Axis selection
        selected_x_axis_th = st.selectbox(
            "X-Axis:", ["Underlying Price", "Strike Price"], key="th_xvar"
        )
        selected_y_group_th = st.selectbox(
            "Y-Axis:", ["Price & Payoff", "Greeks"], key="th_ygrp"
        )
        selected_greeks_th = {}
        if selected_y_group_th == "Greeks":
            cb_cols = th_plot_col.columns(5)
            for idx, name in enumerate(
                ["Delta", "Gamma", "Vega", "Theta", "Rho"]
            ):
                selected_greeks_th[name] = cb_cols[idx].checkbox(
                    name, True, key=f"th_{name}"
                )

    # Theoretical calculation
    time_to_maturity_th = days_to_expiry_th / 365
    pricer_th = PricingModels(
        underlying_price_th,
        strike_price_th,
        risk_free_rate_th,
        time_to_maturity_th,
        volatility_th,
        dividend_yield_th,
    )

    # Scenarios for curves
    base_underlying_price_th = np.linspace(
        0.5 * underlying_price_th, 1.5 * underlying_price_th, 100
    )
    base_strike_price_th = np.linspace(
        0.5 * underlying_price_th, 1.5 * underlying_price_th, 100
    )

    df_underlying_price_th = pd.DataFrame(
        [
            PricingModels(
                float(s),
                strike_price_th,
                risk_free_rate_th,
                time_to_maturity_th,
                volatility_th,
                dividend_yield_th,
            ).options
            for s in base_underlying_price_th
        ]
    )
    df_underlying_price_th["call_payoff"] = np.maximum(
        0, df_underlying_price_th["s"] - strike_price_th
    )
    df_underlying_price_th["put_payoff"] = np.maximum(
        0, strike_price_th - df_underlying_price_th["s"]
    )

    df_strike_price_th = pd.DataFrame(
        [
            PricingModels(
                underlying_price_th,
                float(k),
                risk_free_rate_th,
                time_to_maturity_th,
                volatility_th,
                dividend_yield_th,
            ).options
            for k in base_strike_price_th
        ]
    )
    df_strike_price_th["call_payoff"] = np.maximum(
        0, underlying_price_th - df_strike_price_th["k"]
    )
    df_strike_price_th["put_payoff"] = np.maximum(
        0, df_strike_price_th["k"] - underlying_price_th
    )

    # Build theoretical chart
    fig_th = go.Figure()
    if selected_x_axis_th == "Underlying Price":
        df_plot_th = df_underlying_price_th
        x_vals_th = base_underlying_price_th
        ref_val_th = underlying_price_th
    else:
        df_plot_th = df_strike_price_th
        x_vals_th = base_strike_price_th
        ref_val_th = strike_price_th

    if selected_y_group_th == "Price & Payoff":
        price_key_th = metrics_mapping[contract_type_th]["Price"]
        payoff_key_th = (
            "call_payoff" if contract_type_th == "Call" else "put_payoff"
        )
        fig_th.add_trace(
            go.Scatter(
                x=x_vals_th,
                y=df_plot_th[price_key_th],
                mode="lines",
                name="Price",
            )
        )
        fig_th.add_trace(
            go.Scatter(
                x=x_vals_th,
                y=df_plot_th[payoff_key_th],
                mode="lines",
                name="Payoff",
            )
        )
        height_th = 705
    else:
        for greek, checked in selected_greeks_th.items():
            if checked:
                key = metrics_mapping[contract_type_th][greek]
                fig_th.add_trace(
                    go.Scatter(
                        x=x_vals_th, y=df_plot_th[key], mode="lines", name=greek
                    )
                )
        height_th = 655

    fig_th.add_vline(x=ref_val_th, line=dict(color="red", dash="dash"))
    fig_th.update_layout(
        xaxis_title=selected_x_axis_th,
        yaxis_title=selected_y_group_th,
        hovermode="x",
        height=height_th,
    )
    th_plot_col.plotly_chart(fig_th, use_container_width=True)

    # Display numeric metrics
    metric_cols_th = th_plot_col.columns(6)
    price_key_th = metrics_mapping[contract_type_th]["Price"]
    metric_cols_th[0].metric("Price", f"{pricer_th.options[price_key_th]:.4f}")
    for idx, greek in enumerate(
        ["Delta", "Gamma", "Vega", "Theta", "Rho"], start=1
    ):
        key = metrics_mapping[contract_type_th][greek]
        metric_cols_th[idx].metric(greek, f"{pricer_th.options[key]:.6f}")

    # Theoretical heatmap
    st.markdown("#")
    st.markdown(
        '### <span style="text-decoration: underline">Price & Greeks Heatmap Visualization:</span>',
        unsafe_allow_html=True,
    )

    heatmap_input_col_th, heatmap_chart_col_th = st.columns(
        [0.25, 0.75], gap="large"
    )
    with heatmap_input_col_th:
        metric_th = heatmap_input_col_th.selectbox(
            "Metric:",
            ["Price", "Delta", "Gamma", "Vega", "Theta", "Rho"],
            key="th_heatmap_metric",
        )
        variable_list_th = [
            "Underlying Price",
            "Strike Price",
            "Time to Expiration",
            "Volatility",
        ]
        x_var_th = heatmap_input_col_th.selectbox(
            "X-Axis:", variable_list_th, key="th_heatmap_x"
        )
        y_var_th = heatmap_input_col_th.selectbox(
            "Y-Axis:",
            [v for v in variable_list_th if v != x_var_th],
            key="th_heatmap_y",
        )

        def make_grid_th(var):
            if var == "Underlying Price":
                return np.linspace(
                    0.5 * underlying_price_th, 1.5 * underlying_price_th, 50
                )
            if var == "Strike Price":
                return np.arange(
                    0.25 * underlying_price_th,
                    1.75 * underlying_price_th + 1,
                    10,
                )
            if var == "Time to Expiration":
                return np.arange(0.25, 2.01, 0.25)
            return np.linspace(0.01, 1.0, 50)

        x_vals_h_th = make_grid_th(x_var_th)
        y_vals_h_th = make_grid_th(y_var_th)

    with heatmap_chart_col_th:
        st.markdown(
            f'<div style="text-align:center;"><span style="text-decoration: underline">'
            f"Theoretical | S={underlying_price_th:.2f}, K={strike_price_th:.2f}, r={risk_free_rate_th:.4f}, σ={volatility_th:.2f}, q={dividend_yield_th:.2f}, T={time_to_maturity_th:.2f} | "
            f"{contract_type_th} {metric_th} Heatmap vs {x_var_th} & {y_var_th}"
            "</span></div>",
            unsafe_allow_html=True,
        )
        Z_th = np.zeros((len(y_vals_h_th), len(x_vals_h_th)))
        for i, y in enumerate(y_vals_h_th):
            for j, x in enumerate(x_vals_h_th):
                S = (
                    x
                    if x_var_th == "Underlying Price"
                    else (
                        y
                        if y_var_th == "Underlying Price"
                        else underlying_price_th
                    )
                )
                K = (
                    x
                    if x_var_th == "Strike Price"
                    else (y if y_var_th == "Strike Price" else strike_price_th)
                )
                T = (
                    x
                    if x_var_th == "Time to Expiration"
                    else (
                        y
                        if y_var_th == "Time to Expiration"
                        else time_to_maturity_th
                    )
                )
                sigma_val = (
                    x
                    if x_var_th == "Volatility"
                    else (y if y_var_th == "Volatility" else volatility_th)
                )
                pr = PricingModels(
                    float(S),
                    float(K),
                    risk_free_rate_th,
                    float(T),
                    float(sigma_val),
                    dividend_yield_th,
                )
                key = metrics_mapping[contract_type_th][metric_th]
                Z_th[i, j] = pr.options[key]

        hm_th = go.Figure(
            data=go.Heatmap(
                x=x_vals_h_th, y=y_vals_h_th, z=Z_th, coloraxis="coloraxis"
            )
        )
        hm_th.update_layout(
            xaxis_title=x_var_th,
            yaxis_title=y_var_th,
            coloraxis={"colorscale": "Viridis"},
            margin=dict(t=0, b=0),
            height=550,
        )
        heatmap_chart_col_th.plotly_chart(hm_th)
