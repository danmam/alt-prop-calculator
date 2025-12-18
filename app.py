import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from scipy.stats import (poisson, nbinom, norm, lognorm, weibull_min, gamma, skewnorm)
try:
    from scipy.stats import skewt
    SKEWT_AVAILABLE = True
except ImportError:
    SKEWT_AVAILABLE = False
from scipy.optimize import minimize, brentq
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import exp, factorial

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
DEFAULT_MAE_THRESHOLD = 0.05
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

def devig_market_data(df, market_total=None, method='multiplicative'):
    """
    Applies a given market_total (vig) to a dataframe.
    method: 'multiplicative' (divide by vig) or 'additive' (subtract half-vig).
    """
    df['prob'] = df['odds'].apply(american_to_prob)
    
    if market_total:
        if method == 'multiplicative':
            df['fair_prob'] = df['prob'] / market_total
        elif method == 'additive':
            # Additive: P_fair = P_raw - (Vig - 1) / 2
            # This assumes equal vig distribution on both sides of a 2-way, 
            # applied constantly to 1-way lines.
            vig_diff = (market_total - 1) / 2
            df['fair_prob'] = df['prob'] - vig_diff
            # Clip to ensure valid probabilities (though unlikely to hit < 0 in standard props)
            df['fair_prob'] = df['fair_prob'].clip(lower=0.001, upper=0.999)
    else:
        # st.warning("Proceeding without devigging.")
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
        elif dist_name == 'skewnorm': prob_le_k = skewnorm.cdf(k, a=params[0], loc=params[1], scale=params[2])
        elif dist_name == 'skewt' and SKEWT_AVAILABLE: prob_le_k = skewt.cdf(k, a=params[0], df=params[1], loc=params[2], scale=params[3])
    return 1 - prob_le_k

def calculate_fit_error(params, market_df, use_anchor, anchor_line, anchor_prob, dist_name, target_col='fair_prob'):
    """Generic error function, with optional anchor point."""
    if any(pd.isna(params)): return 1e9
    
    total_error = 0
    over_data = market_df[market_df['type'] == 'over']
    model_probs = over_data['line'].apply(lambda x: get_prob_from_model(params, x, dist_name))
    total_error = np.sum((model_probs - over_data[target_col])**2)
    
    anchor_error = 0
    if use_anchor and anchor_prob is not None:
        anchor_model_prob = get_prob_from_model(params, anchor_line, dist_name)
        anchor_error = (anchor_model_prob - anchor_prob)**2
    
    return total_error + 100 * anchor_error

def fit_model(market_df, use_anchor, anchor_line, anchor_prob, dist_name, target_col='fair_prob'):
    """Finds the best-fit parameters for any given distribution."""
    if not use_anchor and not market_df.empty:
        mean_estimate = market_df.iloc[(market_df[target_col]-0.5).abs().argsort()[:1]]['line'].values[0]
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
        'skewnorm': {'guess': [0, mean_estimate, 5], 'bounds': [(None, None), (None, None), (0.1, None)]},
        'skewt': {'guess': [0, 10, mean_estimate, 5], 'bounds': [(None, None), (1, None), (None, None), (0.1, None)]}
    }
    config = models[dist_name]
    result = minimize(calculate_fit_error, config['guess'], args=(market_df, use_anchor, anchor_line, anchor_prob, dist_name, target_col), bounds=config['bounds'], method='L-BFGS-B')
    if not result.success:
        return None
    return result.x

def solve_location_shift(params, dist_name, target_line, target_prob):
    """
    Shifts the location parameter of a distribution so that P(X > target_line) = target_prob.
    Keeps shape and scale parameters constant (Analysis 3 - Shape Retention).
    """
    # Helper to reconstruct params with new loc
    def make_params(loc_val):
        p = list(params)
        if dist_name == 'norm': p[0] = loc_val
        elif dist_name == 'lognorm': p[1] = loc_val
        elif dist_name == 'weibull': p[1] = loc_val
        elif dist_name == 'gamma': p[1] = loc_val
        elif dist_name == 'skewnorm': p[1] = loc_val
        elif dist_name == 'skewt': p[2] = loc_val
        return p

    def objective(loc_val):
        return get_prob_from_model(make_params(loc_val), target_line, dist_name) - target_prob

    # Bracket search for root
    try:
        # Initial guess is the current loc
        current_loc = params[0] if dist_name == 'norm' else params[1] if dist_name in ['lognorm', 'weibull', 'gamma', 'skewnorm'] else params[2]
        
        # Simple search for bounds
        a, b = current_loc - 20, current_loc + 20
        new_loc = brentq(objective, a, b)
        return make_params(new_loc)
    except:
        return params # Fallback if solver fails

# ==============================================================================
# 4. MAIN ANALYSIS FUNCTION
# ==============================================================================
def run_analysis(df, use_anchor, anchor_line, anchor_odds, target_line, dist_type, mae_threshold):
    """Contains the core analysis logic with robust vig calculation and dynamic book handling."""
    
    # 1. Dynamic Column Handling
    num_cols = df.shape[1]
    if (num_cols - 1) % 2 != 0:
        st.error(f"Invalid column format. Expected 1 Line column + Pairs of Over/Under columns. Found {num_cols} columns.")
        return

    num_books = (num_cols - 1) // 2
    
    new_cols = ['line']
    books = []
    for i in range(num_books):
        book_name = f"book_{i+1}"
        books.append(book_name)
        new_cols.extend([f"{book_name}_over", f"{book_name}_under"])
        
    df.columns = new_cols
    df.replace(['-', ''], np.nan, inplace=True)
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if dist_type == 'Discrete':
        models_to_test = ['poisson', 'nbinom', 'zip']
    else:
        models_to_test = ['norm', 'lognorm', 'weibull', 'gamma', 'skewnorm']
        if SKEWT_AVAILABLE:
            models_to_test.append('skewt')
    
    st.write(f"##### Testing {dist_type} distributions: `{', '.join(models_to_test)}`")

    # Data collection
    all_results = []
    plot_data = [] # Stores (book, method, params, dist_name) for plotting

    anchor_fair_prob_user = american_to_prob(anchor_odds)
    
    book_dataframes = {}
    book_vigs = {}
    book_main_lines = {} # Store the specific line and fair prob for Analysis 3
    
    # 2. Extract Data Per Book & Find Vigs
    for book in books:
        book_df = df[['line', f'{book}_over', f'{book}_under']].copy()
        book_df.columns = ['line', 'over', 'under']
        market_df = pd.melt(book_df, id_vars=['line'], value_vars=['over', 'under'], var_name='type', value_name='odds').dropna(subset=['odds'])
        book_dataframes[book] = market_df
        
        pivot = market_df.pivot_table(index='line', columns='type', values='odds')
        if 'over' in pivot.columns and 'under' in pivot.columns:
            two_way_market = pivot[['over', 'under']].dropna().applymap(american_to_prob)
            if not two_way_market.empty:
                # Calculate vig from the first available 2-way line
                over_prob = two_way_market['over'].iloc[0]
                under_prob = two_way_market['under'].iloc[0]
                market_total = over_prob + under_prob
                book_vigs[book] = market_total
                
                # Store main line details for Analysis 3 (Shape Retention)
                main_line_val = two_way_market.index[0]
                # Fair prob for main line is typically Multiplicative devig
                main_line_fair = over_prob / market_total 
                book_main_lines[book] = {'line': main_line_val, 'fair_prob': main_line_fair}
                
                st.info(f"Found 2-way market for **{book.replace('_', ' ').upper()}** @ {main_line_val} (Vig: {market_total:.4f}).")

    shared_vig = next(iter(book_vigs.values()), None)
    
    # 3. Process Each Book with 3 Analyses
    analyses = ['Additive', 'Multiplicative', 'Shape Retention']
    
    for book in books:
        clean_book_name = book.replace('_', ' ').upper()
        market_df = book_dataframes.get(book)
        
        if market_df is None or market_df.empty:
            continue
            
        vig_to_use = book_vigs.get(book, shared_vig if shared_vig else DEFAULT_VIG_MARKET_TOTAL)
        main_line_info = book_main_lines.get(book)

        with st.expander(f"Analysis for {clean_book_name}", expanded=True):
            
            for analysis_type in analyses:
                best_model = None
                min_mae = float('inf')
                
                # --- PREPARE DATA ---
                working_df = market_df.copy()
                
                if analysis_type == 'Additive':
                    # Method 1: Constant Vig Subtraction
                    working_df = devig_market_data(working_df, vig_to_use, method='additive')
                    target_col = 'fair_prob'
                    fit_anchor_prob = anchor_fair_prob_user # Use user anchor for fitting
                    
                elif analysis_type == 'Multiplicative':
                    # Method 2: Constant Ratio Division (Standard)
                    working_df = devig_market_data(working_df, vig_to_use, method='multiplicative')
                    target_col = 'fair_prob'
                    fit_anchor_prob = anchor_fair_prob_user
                    
                elif analysis_type == 'Shape Retention':
                    # Method 3: Fit Raw, then Shift
                    # We treat 'prob' (raw) as the target for the initial fit
                    working_df['prob'] = working_df['odds'].apply(american_to_prob)
                    target_col = 'prob'
                    fit_anchor_prob = None # Don't anchor to user point during raw fit
                
                # --- FIT MODELS ---
                for dist_name in models_to_test:
                    try:
                        # 1. Fit
                        params = fit_model(working_df, use_anchor if analysis_type != 'Shape Retention' else False, 
                                         anchor_line, fit_anchor_prob, dist_name, target_col)
                        if params is None: continue

                        # 2. Process / Shift based on Method
                        final_params = params
                        
                        if analysis_type == 'Shape Retention':
                            # Step A: Shift to remove vig (Anchor to Main Line Fair Prob)
                            if main_line_info:
                                final_params = solve_location_shift(final_params, dist_name, main_line_info['line'], main_line_info['fair_prob'])
                            
                            # Step B: Shift to User Anchor (Calibration)
                            if use_anchor:
                                final_params = solve_location_shift(final_params, dist_name, anchor_line, anchor_fair_prob_user)
                        
                        # 3. Calculate Error (MAE) against Fair Data (Devigged via Multiplicative for standardized comparison)
                        # Note: For Shape Retention, we compare the shifted model against standard devigged points to see fit quality
                        comparison_df = devig_market_data(market_df.copy(), vig_to_use, method='multiplicative')
                        model_probs = comparison_df['line'].apply(lambda x: get_prob_from_model(final_params, x, dist_name))
                        mae = (comparison_df[comparison_df['type']=='over']['fair_prob'] - model_probs[comparison_df['type']=='over']).abs().mean()
                        
                        if mae < min_mae:
                            min_mae = mae
                            best_model = {'book': book, 'method': analysis_type, 'model': dist_name, 'params': final_params, 'mae': mae}
                            
                    except Exception as e:
                        continue

                # --- STORE RESULTS ---
                if best_model:
                    target_prob = get_prob_from_model(best_model['params'], target_line, best_model['model'])
                    best_model['target_prob_over'] = target_prob
                    all_results.append(best_model)
                    plot_data.append(best_model)
    
    # --- OUTPUT RESULTS ---
    if not all_results:
        st.error("No valid models found.")
        return

    res_df = pd.DataFrame(all_results)
    
    st.header("Comparative Results")
    st.write("Comparison of Fair Odds for Target Line using different devigging methodologies:")
    
    # Pivot for clean display
    display_df = res_df.pivot_table(index='book', columns='method', values='target_prob_over', aggfunc='first')
    
    # Add formatting
    formatted_df = display_df.applymap(lambda p: f"{prob_to_american(p):+.0f} ({p:.1%})")
    st.dataframe(formatted_df)

    # --- VISUALIZATION ---
    st.header("Visual Comparison")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot Scatter Points (Standard Multiplicative Devig for reference)
    cmap = cm.get_cmap('tab10')
    colors = {b: cmap(i) for i, b in enumerate(books)}
    styles = {'Additive': ':', 'Multiplicative': '--', 'Shape Retention': '-'}
    
    for book in books:
        # Plot data points (using Multiplicative devig as visual baseline)
        m_df = book_dataframes[book].copy()
        vig = book_vigs.get(book, DEFAULT_VIG_MARKET_TOTAL)
        m_df = devig_market_data(m_df, vig, method='multiplicative')
        over_data = m_df[m_df['type']=='over']
        ax.scatter(over_data['line'], over_data['fair_prob'], color=colors[book], alpha=0.3, label=f"{book} Data (Ref)")

    # Plot Curves
    x_range = np.linspace(df['line'].min() - 2, df['line'].max() + 2, 200)
    
    for res in plot_data:
        # Only plot if MAE is reasonable to avoid clutter
        if res['mae'] <= mae_threshold * 1.5: 
            y_vals = [get_prob_from_model(res['params'], x, res['model']) for x in x_range]
            c = colors[res['book']]
            s = styles[res['method']]
            lbl = f"{res['book']} {res['method']} ({res['model']})"
            ax.plot(x_range, y_vals, color=c, linestyle=s, label=lbl, linewidth=2 if res['method'] == 'Shape Retention' else 1.5)

    if use_anchor:
        ax.scatter(anchor_line, anchor_fair_prob_user, c='red', s=150, marker='*', label='Anchor Point', zorder=10)
        
    ax.axvline(target_line, color='black', alpha=0.3, linestyle='-')
    ax.set_title("Methodology Comparison: Shape Retention vs. Standard Devig")
    ax.set_xlabel("Line")
    ax.set_ylabel("Fair Probability (Over)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

# ==============================================================================
# 5. STREAMLIT UI
# ==============================================================================

st.set_page_config(layout="wide")
st.title("🎯 Advanced Prop Line Calculator")

with st.sidebar:
    st.header("1. Upload Data")
    st.markdown("""
    **Format Requirements:**
    * **Column 1:** Prop Line (e.g., 18.5, 19.5)
    * **Column 2:** Book 1 Over Odds
    * **Column 3:** Book 1 Under Odds
    * **Column 4:** Book 2 Over Odds...
    """)
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

        if st.button("Run Advanced Analysis", use_container_width=True):
            with st.spinner("Calculating Multi-Method Models..."):
                run_analysis(df, use_anchor, anchor_line, anchor_odds, target_line, dist_type, mae_threshold)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to begin.")
