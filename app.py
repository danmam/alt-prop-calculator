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
import matplotlib
from math import exp, factorial

# ==============================================================================
# 0. CONFIGURATION & SESSION STATE
# ==============================================================================
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False

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
            vig_diff = (market_total - 1) / 2
            df['fair_prob'] = df['prob'] - vig_diff
            df['fair_prob'] = df['fair_prob'].clip(lower=0.001, upper=0.999)
    else:
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

    try:
        current_loc = params[0] if dist_name == 'norm' else params[1] if dist_name in ['lognorm', 'weibull', 'gamma', 'skewnorm'] else params[2]
        a, b = current_loc - 20, current_loc + 20
        new_loc = brentq(objective, a, b)
        return make_params(new_loc)
    except:
        return params 

# ==============================================================================
# 4. MAIN ANALYSIS FUNCTION
# ==============================================================================
def run_analysis(df, use_anchor, anchor_line, anchor_odds, target_line, dist_type, mae_threshold, show_individual_plots):
    """Contains the core analysis logic with robust vig calculation and dynamic book handling."""
    
    # 1. ROBUST COLUMN HANDLING
    # Safe extraction of the Line column to avoid KeyError later
    line_col = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    total_cols = df.shape[1]
    
    book_dataframes = {}
    book_vigs = {}
    book_main_lines = {} 
    books = []
    
    # Duplicate Detection
    loaded_book_signatures = set()

    book_idx = 1
    for i in range(1, total_cols, 2):
        if i + 1 >= total_cols:
            break 
            
        over_raw = pd.to_numeric(df.iloc[:, i], errors='coerce')
        under_raw = pd.to_numeric(df.iloc[:, i+1], errors='coerce')
        
        if over_raw.isna().all() and under_raw.isna().all():
            continue

        # Prevent duplicate columns (ghost books)
        sig_df = pd.DataFrame({'o': over_raw, 'u': under_raw}).fillna(-9999)
        book_signature = pd.util.hash_pandas_object(sig_df).sum()
        
        if book_signature in loaded_book_signatures:
            continue
            
        loaded_book_signatures.add(book_signature)
            
        book_name = f"Book {book_idx}"
        books.append(book_name)
        
        temp_df = pd.DataFrame({'line': line_col, 'over': over_raw, 'under': under_raw})
        market_df = pd.melt(temp_df, id_vars=['line'], value_vars=['over', 'under'], var_name='type', value_name='odds').dropna(subset=['odds'])
        book_dataframes[book_name] = market_df
        
        pivot = market_df.pivot_table(index='line', columns='type', values='odds')
        if 'over' in pivot.columns and 'under' in pivot.columns:
            # Replaced applymap with map to fix deprecation warning
            two_way_market = pivot[['over', 'under']].dropna().map(american_to_prob)
            if not two_way_market.empty:
                over_prob = two_way_market['over'].iloc[0]
                under_prob = two_way_market['under'].iloc[0]
                market_total = over_prob + under_prob
                book_vigs[book_name] = market_total
                
                main_line_val = two_way_market.index[0]
                main_line_fair = over_prob / market_total 
                book_main_lines[book_name] = {'line': main_line_val, 'fair_prob': main_line_fair}
        
        book_idx += 1

    if not books:
        st.error("No valid book data found. Please check column format.")
        return

    st.success(f"Successfully processed {len(books)} unique book(s).")

    if dist_type == 'Discrete':
        models_to_test = ['poisson', 'nbinom', 'zip']
    else:
        models_to_test = ['norm', 'lognorm', 'weibull', 'gamma', 'skewnorm']
        if SKEWT_AVAILABLE:
            models_to_test.append('skewt')
    
    shared_vig = next(iter(book_vigs.values()), None)
    
    # 2. ANALYSIS LOOP
    all_results = []
    analyses = ['Additive', 'Multiplicative', 'Shape Retention']
    anchor_fair_prob_user = american_to_prob(anchor_odds)

    progress_bar = st.progress(0)
    
    for i, book in enumerate(books):
        progress_bar.progress((i + 1) / len(books))
        market_df = book_dataframes.get(book)
        vig_to_use = book_vigs.get(book, shared_vig if shared_vig else DEFAULT_VIG_MARKET_TOTAL)
        main_line_info = book_main_lines.get(book)

        for analysis_type in analyses:
            best_model = None
            min_mae = float('inf')
            
            working_df = market_df.copy()
            
            if analysis_type == 'Additive':
                working_df = devig_market_data(working_df, vig_to_use, method='additive')
                target_col = 'fair_prob'
                fit_anchor_prob = anchor_fair_prob_user 
                
            elif analysis_type == 'Multiplicative':
                working_df = devig_market_data(working_df, vig_to_use, method='multiplicative')
                target_col = 'fair_prob'
                fit_anchor_prob = anchor_fair_prob_user
                
            elif analysis_type == 'Shape Retention':
                working_df['prob'] = working_df['odds'].apply(american_to_prob)
                target_col = 'prob'
                fit_anchor_prob = None 
            
            for dist_name in models_to_test:
                try:
                    params = fit_model(working_df, use_anchor if analysis_type != 'Shape Retention' else False, 
                                     anchor_line, fit_anchor_prob, dist_name, target_col)
                    if params is None: continue

                    final_params = params
                    
                    if analysis_type == 'Shape Retention':
                        if main_line_info:
                            final_params = solve_location_shift(final_params, dist_name, main_line_info['line'], main_line_info['fair_prob'])
                        if use_anchor:
                            final_params = solve_location_shift(final_params, dist_name, anchor_line, anchor_fair_prob_user)
                    
                    comparison_df = devig_market_data(market_df.copy(), vig_to_use, method='multiplicative')
                    model_probs = comparison_df['line'].apply(lambda x: get_prob_from_model(final_params, x, dist_name))
                    mae = (comparison_df[comparison_df['type']=='over']['fair_prob'] - model_probs[comparison_df['type']=='over']).abs().mean()
                    
                    if mae < min_mae:
                        min_mae = mae
                        best_model = {'book': book, 'method': analysis_type, 'model': dist_name, 'params': final_params, 'mae': mae}
                        
                except Exception:
                    continue

            if best_model:
                target_prob = get_prob_from_model(best_model['params'], target_line, best_model['model'])
                best_model['target_prob_over'] = target_prob
                all_results.append(best_model)
    
    progress_bar.empty()
    
    if not all_results:
        st.error("No valid models found.")
        return

    res_df = pd.DataFrame(all_results)
    
    # ==========================================================================
    # 3. RESULTS TABLE WITH AVERAGES
    # ==========================================================================
    
    pivot_df = res_df.pivot_table(index='book', columns='method', values='target_prob_over', aggfunc='first')
    
    mean_row = pivot_df.mean().to_frame().T
    mean_row.index = ['AVERAGE']
    
    final_table = pd.concat([pivot_df, mean_row])
    
    st.header("Results Table")
    # Replaced applymap with map to fix deprecation warning
    formatted_table = final_table.map(lambda p: f"{prob_to_american(p):+.0f}" if pd.notnull(p) else "-")
    # Updated use_container_width to width='stretch' to fix deprecation warning
    st.dataframe(formatted_table, width=1500) 

    mult_mean_prob = mean_row['Multiplicative'].iloc[0] if 'Multiplicative' in mean_row.columns else None

    # ==========================================================================
    # 4. PLOTTING - DEFAULT (Multiplicative Consensus)
    # ==========================================================================
    
    st.header("Visualizations")
    
    # --- PLOT 1: Market Consensus (Multiplicative Only) ---
    st.subheader("Market Consensus (Multiplicative Models)")
    
    fig_main, ax_main = plt.subplots(figsize=(12, 6))
    
    # Replaced get_cmap with matplotlib.colormaps to fix deprecation warning
    try:
        cmap = matplotlib.colormaps['tab10']
    except:
        cmap = plt.get_cmap('tab10') # Fallback for older mpl versions
        
    colors = {b: cmap(i % 10) for i, b in enumerate(books)}
    
    # Use line_col instead of df['line'] to prevent KeyError
    x_min = line_col.min()
    x_max = line_col.max()
    x_range = np.linspace(x_min - 2, x_max + 2, 200)
    
    # A. Plot Curves (Multiplicative ONLY)
    mult_results = [r for r in all_results if r['method'] == 'Multiplicative']
    
    for res in mult_results:
        alpha = 0.8 
        if res['mae'] > mae_threshold:
            ls = ':' 
            alpha = 0.6
        else:
            ls = '-'
        y_vals = [get_prob_from_model(res['params'], x, res['model']) for x in x_range]
        ax_main.plot(x_range, y_vals, color=colors[res['book']], linestyle=ls, alpha=alpha, label=f"{res['book']} ({res['model']})", linewidth=1.5)

    # B. Plot Scatter Points (Multiplicative Devigged ONLY)
    for book in books:
        if book not in book_dataframes: continue
        m_df = book_dataframes[book].copy()
        vig = book_vigs.get(book, DEFAULT_VIG_MARKET_TOTAL)
        
        # Devig
        devig_df = devig_market_data(m_df, vig, method='multiplicative')
        devig_over = devig_df[devig_df['type']=='over']
        
        # Plot Scatter (matching curve color)
        ax_main.scatter(devig_over['line'], devig_over['fair_prob'], color=colors[book], marker='o', s=30, alpha=0.5)

    # C. Plot Average Horizontal Line
    if mult_mean_prob:
        # axhline draws a line across the entire plot axes
        ax_main.axhline(y=mult_mean_prob, color='black', linestyle='-', linewidth=2, label=f'Avg Multiplicative Prob ({prob_to_american(mult_mean_prob):+.0f})')
        # Use line_col.min() here too to prevent error
        ax_main.text(x_min, mult_mean_prob + 0.01, f" Avg: {prob_to_american(mult_mean_prob):+.0f}", fontsize=10, fontweight='bold')

    # D. Plot Target Line & Anchor
    ax_main.axvline(x=target_line, color='purple', linestyle=':', linewidth=2, label=f'Target Line {target_line}')
    
    if use_anchor:
        ax_main.scatter(anchor_line, anchor_fair_prob_user, c='red', s=150, marker='*', label='Anchor Point', zorder=10)

    # Axis Formatting
    ax_main.set_ylabel("Fair American Odds")
    ax_main.set_xlabel("Line")
    ax_main.set_ylim(0.02, 0.98)
    ticks = np.arange(0.1, 1.0, 0.1)
    ax_main.set_yticks(ticks)
    ax_main.set_yticklabels([f'{prob_to_american(t):+.0f}' for t in ticks])
    
    ax_main2 = ax_main.twinx()
    ax_main2.set_ylabel("Probability (%)")
    ax_main2.set_ylim(0.02, 0.98)
    ax_main2.set_yticks(ticks)
    ax_main2.set_yticklabels([f'{t:.0%}' for t in ticks])
    
    handles, labels = ax_main.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_main.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.08, 1), loc='upper left')
    ax_main.grid(True, alpha=0.3)
    
    st.pyplot(fig_main)

    # ==========================================================================
    # 5. PLOTTING - INDIVIDUAL DETAILS (Optional)
    # ==========================================================================
    
    if show_individual_plots:
        st.markdown("---")
        st.subheader("Individual Book Details")
        
        for book in books:
            book_results = [r for r in all_results if r['book'] == book]
            if not book_results: continue
            
            fig_sub, ax_sub = plt.subplots(figsize=(10, 5))
            
            # Plot Raw/Devig Data Points
            m_df = book_dataframes[book].copy()
            vig = book_vigs.get(book, DEFAULT_VIG_MARKET_TOTAL)
            
            # Raw
            m_df['raw_prob'] = m_df['odds'].apply(american_to_prob)
            raw_over = m_df[m_df['type']=='over']
            ax_sub.scatter(raw_over['line'], raw_over['raw_prob'], color='gray', marker='x', s=50, alpha=0.6, label='Raw Odds (With Vig)')
            
            # Devigged (Multiplicative Reference)
            devig_df = devig_market_data(m_df.copy(), vig, method='multiplicative')
            devig_over = devig_df[devig_df['type']=='over']
            ax_sub.scatter(devig_over['line'], devig_over['fair_prob'], color='black', marker='o', s=50, alpha=0.8, label='Fair Odds (No Vig)')
            
            # Plot Curves
            styles = {'Additive': ':', 'Multiplicative': '--', 'Shape Retention': '-'}
            
            for res in book_results:
                y_vals = [get_prob_from_model(res['params'], x, res['model']) for x in x_range]
                ls = styles.get(res['method'], '-')
                ax_sub.plot(x_range, y_vals, linestyle=ls, linewidth=2, label=f"{res['method']} ({res['model']})")
                
            ax_sub.set_title(f"{book} Analysis")
            ax_sub.set_ylabel("Fair Odds")
            ax_sub.set_ylim(0.02, 0.98)
            ax_sub.set_yticks(ticks)
            ax_sub.set_yticklabels([f'{prob_to_american(t):+.0f}' for t in ticks])
            ax_sub.axvline(x=target_line, color='purple', linestyle=':', alpha=0.5)
            ax_sub.legend()
            ax_sub.grid(True, alpha=0.3)
            
            st.pyplot(fig_sub)


# ==============================================================================
# 5. STREAMLIT UI
# ==============================================================================

st.set_page_config(layout="wide")
st.title("🎯 Advanced Prop Line Calculator")

with st.sidebar:
    st.header("1. Upload Data")
    st.markdown("""
    **Format Requirements:**
    * **Column 1:** Prop Line
    * **Column 2/3:** Book 1 Over/Under
    * **Column 4/5:** Book 2 Over/Under
    * ...
    """)
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    st.header("2. Set Parameters")
    
    use_anchor = st.checkbox("Use Anchor Point", value=True)
    
    if use_anchor:
        anchor_line = st.number_input("Anchor Line", value=21.5, step=1.0, format="%.1f")
        anchor_odds = st.number_input("Anchor Odds", value=109)
    else:
        anchor_line = 0
        anchor_odds = 0
        
    target_line = st.number_input("Target Line", value=18.5, step=1.0, format="%.1f")
    dist_type = st.radio("Distribution Type", ('Discrete', 'Continuous'))
    mae_threshold = st.slider("Max MAE Threshold", 0.01, 0.1, DEFAULT_MAE_THRESHOLD, 0.01)
    
    # Checkbox placed PROMINENTLY in Sidebar for visibility
    st.markdown("---")
    show_individual = st.checkbox("Show Individual Book Plots (All Methods)", value=False)
    
    # Button logic
    if st.button("Run Analysis", use_container_width=True):
        st.session_state.analysis_run = True

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=',', engine='python', on_bad_lines='skip', encoding='utf-8-sig')
        df.replace(['-', ''], np.nan, inplace=True)
        
        st.header("Data Preview")
        # Updated to use width='stretch' to avoid deprecation warning
        st.dataframe(df.head(), width=1500)

        # Execute if button was clicked OR if already run (allows toggling)
        if st.session_state.analysis_run:
            run_analysis(df, use_anchor, anchor_line, anchor_odds, target_line, dist_type, mae_threshold, show_individual)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        # Helpful debugging info
        st.warning("If this is a Key Error, it likely means the code tried to find a specific column name. I have updated the code to be agnostic to column names.")
else:
    st.info("Please upload a CSV file to begin.")
