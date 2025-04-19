# Black-Scholes & Merton Options Visualizer

This Streamlit app provides an interactive dashboard to explore both real and theoretical option data using the Black-Scholes and Black-Scholes-Merton models. Users can visualize option prices, Greeks, and payoffs for Calls and Puts, with real market data or custom parameters.

## Features
- Two-tab dashboard: Real underlying data and Theoretical parameters
- Interactive charts for price, volume, and option metrics
- Supports Black-Scholes and Black-Scholes-Merton models
- Visualizes Greeks and payoffs

## Installation
1. Clone the repository or copy the `bsm_options_visualizer` folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the Streamlit app from the `bsm_options_visualizer` directory:
```bash
streamlit run app.py
```

## Folder structure
- `app.py`: Main Streamlit application
- `models_class.py`: Option pricing models and logic
- `models/`: Additional model files (if any)

## Requirements
- Python 3.8+

## License
MIT License