import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from scipy.stats import (poisson, nbinom, norm, lognorm, weibull_min, gamma)
try:
    from scipy.stats import skewt
    SKEWT_AVAILABLE = True
except ImportError:
    SKEWT_AVAILABLE = False
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from math import exp, factorial
import io

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
DEFAULT_MAE_THRESHOLD = 0.05
# A 7.1% vig means the probabilities sum to 107.1%, or 1.071
DEFAULT_VIG_MARKET_TOTAL = 1.071

# ==============================================================================
# 1. UTILITY & MODELING FUNCTIONS
# ==============================================================================

def american_to_prob(odds):
    """Converts American odds to a probability."""
    if odds is None or pd.isna(odds): return np.nan
    odds = float(odds)
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)

def prob_to_american(prob):
    """Converts a probability to American odds."""
    if prob is None or pd.isna(prob) or not (0 < prob < 1): return np.nan
    return round((100 / prob) - 100) if prob < 0.5 else round(prob / (1 - prob) * -100)

def zip_cdf(k, pi, lam):
    """Custom Cumulative Distribution Function for a Zero-Inflated Poisson model."""
    if k < 0: return 0
    k = int(k)
    p_zero = pi + (1 - pi) * exp(-lam)
    if k == 0: return p_zero
    cdf_val = p_zero
    for i in range(1, k + 1):
        try:
            cdf_val += (1 - pi) * (lam**i * exp(-lam)) / factorial(i)
        except (OverflowError, ValueError):
            return 1.0
    return cdf_val

def devig_market_data(df, market_total=None):
    """Applies a given market_total (vig) to a dataframe."""
    df['prob'] = df['odds'].apply(american_to_prob)
    
    if market_total:
        df['fair_prob'] = df['prob'] / market_total
    else:
        # This case should no longer be hit if the new logic works
        st.warning("Proceeding without devigging.")
        df['fair_prob'] = df['prob']
    return df

def get_prob_from_model(params, line, dist_name):
    """Calculates the 'over' probability for a given line from any specified distribution."""
    if dist_name in ['poisson', 'nbinom', 'zip']:
        k = np.floor(line)
        if dist_name == 'poisson': prob_le_k = poisson.cdf(k, mu=params[0])
        elif dist_name == 'nbinom': prob_le_k = nbinom.cdf(k, n=params[0], p=params[1])
        elif dist_name == 'zip': prob_le_k = zip_cdf(k, pi=params[0], lam=params[1])
    else:
        k = line
        if dist_name == 'norm': prob_le_k = norm.cdf(k, loc=params[0], scale=params[1])
        elif dist_name == 'lognorm': prob_le_k = lognorm.cdf(k, s=params[0], loc=params[1], scale=params[2])
        elif dist_name == 'weibull': prob_le_k = weibull_min.cdf(k, c=params[0], loc=params[1], scale=params[2])
        elif dist_name == 'gamma': prob_le_k = gamma.cdf(k, a=params[0], loc=params[1], scale=params[2])
        elif dist_name == 'skewt' and SKEWT_AVAILABLE: prob_le_k = skewt.cdf(k, a=params[0], df=params[1], loc=params[2], scale=params[3])
    return 1 - prob_le_k

def calculate_fit_error(params, market_df, use_anchor, anchor_line, anchor_prob, dist_name):
    """Generic error function, with optional anchor point."""
    if any(pd.isna(params)): return 1e9
    
    total_error = 0
    over_data = market_df[market_df['type'] == 'over']
    model_probs = over_data['line'].apply(lambda x: get_prob_from_model(params, x, dist_name))
    total_error = np.sum((model_probs - over_data['fair_prob'])**2)
    
    anchor_error = 0
    if use_anchor:
        anchor_model_prob = get_prob_from_model(params, anchor_line, dist_name)
        anchor_error = (anchor_model_prob - anchor_prob)**2
    
    return total_error + 100 * anchor_error

def fit_model(market_df, use_anchor, anchor_line, anchor_prob, dist_name):
    """Finds the best-fit parameters for any given distribution."""
    if not use_anchor and not market_df.empty:
        mean_estimate = market_df.iloc[(market_df['fair_prob']-0.5).abs().argsort()[:1]]['line'].values[0]
    else:
        mean_estimate = anchor_line

    models = {
        'poisson': {'guess': [mean_estimate], 'bounds': [(0.1, None)]},
        'nbinom': {'guess': [20, 0.5], 'bounds': [(0.1, None), (0.01, 0.99)]},
        'zip': {'guess': [0.1, mean_estimate], 'bounds': [(0.01, 0.99), (0.1, None)]},
        'norm': {'guess': [mean_estimate, 5], 'bounds': [(None, None), (0.1, None)]},
        'lognorm': {'guess': [0.5, 0, mean_estimate], 'bounds': [(0.01, None), (None, None), (0.1, None)]},
        'weibull': {'guess': [1.5, 0, mean_estimate], 'bounds': [(0.1, None), (None, None), (0.1, None)]},
        'gamma': {'guess': [2, 0, mean_estimate / 2], 'bounds': [(0.1, None), (None, None), (0.1, None)]},
        'skewt': {'guess': [0, 10, mean_estimate, 5], 'bounds': [(None, None), (1, None), (None, None), (0.1, None)]}
    }
    config = models[dist_name]
    result = minimize(calculate_fit_error, config['guess'], args=(market_df, use_anchor, anchor_line, anchor_prob, dist_name), bounds=config['bounds'], method='L-BFGS-B')
    if not result.success:
        st.warning(f"Optimizer failed to converge for {dist_name}. Results may be unreliable.")
    return result.x

# ==============================================================================
# 4. MAIN ANALYSIS FUNCTION
# ==============================================================================
def run_analysis(df, use_anchor, anchor_line, anchor_odds, target_line, dist_type, mae_threshold):
    """Contains the core analysis logic with vig sharing and a default vig fallback."""
    
    df.columns = ['textsm', 'fd_over', 'fd_under', 'dk_over', 'dk_under']
    df.replace(['-', ''], np.nan, inplace=True)
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if dist_type == 'Discrete':
        models_to_test = ['poisson', 'nbinom', 'zip']
    else:
        models_to_test = ['norm', 'lognorm', 'weibull', 'gamma']
        if SKEWT_AVAILABLE:
            models_to_test.append('skewt')
    
    st.write(f"##### Testing {dist_type} distributions: `{', '.join(models_to_test)}`")

    books = ['fd', 'dk']
    all_book_results = []
    all_market_data = []
    anchor_fair_prob = american_to_prob(anchor_odds)
    
    # --- Vig Sharing Logic ---
    book_dataframes = {}
    book_vigs = {}
    for book in books:
        book_df = df[['textsm', f'{book}_over', f'{book}_under']].copy()
        book_df.columns = ['line', 'over', 'under']
        market_df = pd.melt(book_df, id_vars=['line'], value_vars=['over', 'under'], var_name='type', value_name='odds').dropna(subset=['odds'])
        book_dataframes[book] = market_df
        
        two_way_market = market_df.pivot_table(index='line', columns='type', values='odds').dropna().applymap(american_to_prob)
        if not two_way_market.empty:
            market_total = two_way_market['over'].iloc[0] + two_way_market['under'].iloc[0]
            book_vigs[book] = market_total
            st.info(f"Found 2-way market for **{book.upper()}** (Vig: {market_total:.4f}).")

    shared_vig = next(iter(book_vigs.values()), None)
    
    for book in books:
        with st.expander(f"Processing for {book.upper()}..."):
            market_df = book_dataframes.get(book)
            if market_df is None or market_df.empty:
                st.warning(f"No data for {book.upper()}. Skipping.")
                continue

            # --- MODIFICATION START ---
            # Vig application hierarchy
            if book in book_vigs:
                vig_to_use = book_vigs[book]
                st.write(f"Using **{book.upper()}'s own vig** for devigging.")
            elif shared_vig:
                vig_to_use = shared_vig
                st.write(f"Using **shared vig** ({vig_to_use:.4f}) from another book.")
            else:
                vig_to_use = DEFAULT_VIG_MARKET_TOTAL
                st.warning(f"No 2-way market found. Using **default vig** of {vig_to_use:.4f}.")
            # --- MODIFICATION END ---
                
            devigged_df = devig_market_data(market_df, vig_to_use)
            all_market_data.append(devigged_df)
            
            best_model_for_book = None
            min_mae = float('inf')

            for dist_name in models_to_test:
                try:
                    params = fit_model(devigged_df, use_anchor, anchor_line, anchor_fair_prob, dist_name)
                    model_probs = devigged_df['line'].apply(lambda x: get_prob_from_model(params, x, dist_name))
                    mae = (devigged_df[devigged_df['type']=='over']['fair_prob'] - model_probs[devigged_df['type']=='over']).abs().mean()
                    
                    if mae < min_mae:
                        min_mae = mae
                        best_model_for_book = {'book': book, 'model': dist_name, 'params': params, 'mae': mae}
                except (ValueError, TypeError) as e:
                    st.error(f"Could not fit '{dist_name}' for {book.upper()}. Optimizer failed. Skipping.")
                    continue

            if best_model_for_book:
                target_prob = get_prob_from_model(best_model_for_book['params'], target_line, best_model_for_book['model'])
                best_model_for_book['target_prob_over'] = target_prob
                all_book_results.append(best_model_for_book)
                st.write(f"Best fit: **{best_model_for_book['model'].capitalize()}** (MAE: {min_mae:.4f})")
                st.write(f"Predicted O{target_line} probability: {target_prob:.4f}")
            else:
                st.error(f"Could not find any suitable model for {book.upper()}.")
    
    if not all_book_results:
        st.error("No data could be processed. Cannot provide a final prediction.")
        return
    
    results_df = pd.DataFrame(all_book_results)
    reliable_results_df = results_df[results_df['mae'] <= mae_threshold].copy()
    
    st.header("Final Results")
    st.write("Best fit model found for each book (all results):")
    st.dataframe(results_df[['book', 'model', 'mae', 'target_prob_over']])
    
    if reliable_results_df.empty:
        st.warning(f"No models met the quality threshold (MAE <= {mae_threshold}). Cannot provide a reliable final prediction.")
    else:
        avg_fair_prob_over = reliable_results_df['target_prob_over'].mean()
        avg_fair_odds_over = prob_to_american(avg_fair_prob_over)
        avg_fair_prob_under = 1 - avg_fair_prob_over
        avg_fair_odds_under = prob_to_american(avg_fair_prob_under)
        
        st.subheader(f"Final Averaged Prediction for line {target_line}")
        st.info(f"Based on **{len(reliable_results_df)}** reliable book(s) where MAE ≤ {mae_threshold}")
        
        col1, col2 = st.columns(2)
        col1.metric("Fair Odds (Over)", f"{avg_fair_odds_over:+.0f}")
        col2.metric("Fair Odds (Under)", f"{avg_fair_odds_under:+.0f}")

    # --- Visualization ---
    st.header("Visualization")
    if not all_market_data:
        st.warning("No data available for plotting.")
        return
        
    fig, ax = plt.subplots(figsize=(10, 6))
    full_market_df = pd.concat(all_market_data)
    colors = {'fd': 'blue', 'dk': 'green'}
    for book in ['fd', 'dk']:
        book_data = full_market_df[(full_market_df['book'] == book) & (full_market_df['type'] == 'over')]
        if not book_data.empty:
            ax.scatter(book_data['line'], book_data['fair_prob'], color=colors.get(book, 'gray'), label=f'{book.upper()} Market "Over" (Devigged)', alpha=0.7, s=50)
    
    for _, res in results_df.iterrows():
        line_style = '--' if res['mae'] <= mae_threshold else ':'
        label_suffix = " (Best)" if res['mae'] <= mae_threshold else f" (Poor Fit, MAE={res['mae']:.2f})"
        plot_lines = np.linspace(full_market_df['line'].min() - 2, full_market_df['line'].max() + 2, 200)
        model_probs = [get_prob_from_model(res['params'], l, res['model']) for l in plot_lines]
        ax.plot(plot_lines, model_probs, color=colors.get(res['book'], 'gray'), ls=line_style, label=f"{res['book'].upper()} {res['model'].capitalize()} Fit{label_suffix}")

    if use_anchor:
        ax.scatter(anchor_line, anchor_fair_prob, c='red', s=200, marker='*', label=f'Anchor: Over {anchor_line} @ {anchor_odds:+.0f}', zorder=5)
    
    ax.axvline(x=target_line, color='purple', linestyle=':', label=f'Target Line: {target_line}')
    if not reliable_results_df.empty:
        avg_fair_prob_over = reliable_results_df['target_prob_over'].mean()
        avg_fair_odds_over = prob_to_american(avg_fair_prob_over)
        ax.axhline(y=avg_fair_prob_over, color='purple', linestyle=':', label=f'Final Avg. Odds: {avg_fair_odds_over:+.0f}')
    
    ax.set_title(f'Per-Book Model Fits ({dist_type} Distributions)', fontsize=16)
    ax.set_xlabel('Player Prop Line', fontsize=12)
    ax.set_ylabel('Fair American Odds ("Over")', fontsize=12)
    ax.set_ylim(bottom=0.02, top=0.98) 
    ticks = ax.get_yticks()
    valid_ticks = [t for t in ticks if 0 < t < 1]
    ax.set_yticks(valid_ticks)
    ax.set_yticklabels([f'{prob_to_american(t):+.0f}' for t in valid_ticks])
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)

# ==============================================================================
# 5. STREAMLIT UI
# ==============================================================================

st.set_page_config(layout="wide")
st.title("🎯 Derivative Prop Line Calculator")

with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    st.header("2. Set Parameters")
    
    use_anchor = st.checkbox("Use Anchor Point for Calibration", value=True)
    
    if use_anchor:
        anchor_line = st.number_input("Anchor Line", value=21.5, step=1.0, format="%.1f")
        anchor_odds = st.number_input("Anchor Odds (American)", value=109)
    else:
        anchor_line = 0
        anchor_odds = 0
        
    target_line = st.number_input("Target Line", value=18.5, step=1.0, format="%.1f")
    dist_type = st.radio("Distribution Type", ('Discrete', 'Continuous'))
    mae_threshold = st.slider("Max MAE Threshold", min_value=0.01, max_value=0.1, value=DEFAULT_MAE_THRESHOLD, step=0.01, format="%.2f")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=',', engine='python', on_bad_lines='skip', encoding='utf-8-sig')
        st.header("Data Preview")
        st.dataframe(df.head())

        if st.button("Run Analysis", use_container_width=True):
            with st.spinner("Calculating..."):
                run_analysis(df, use_anchor, anchor_line, anchor_odds, target_line, dist_type, mae_threshold)

    except Exception as e:
        st.error(f"An error occurred while reading or processing the file: {e}")
else:
    st.info("Please upload a CSV file to begin.")
