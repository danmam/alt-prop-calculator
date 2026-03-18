import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import (poisson, nbinom, norm, lognorm, gamma, skewnorm)
try:
    from scipy.stats import skewt
    SKEWT_AVAILABLE = True
except ImportError:
    SKEWT_AVAILABLE = False
from scipy.optimize import minimize, brentq
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 0. CONFIGURATION & SESSION STATE
# ==============================================================================
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'available_books' not in st.session_state:
    st.session_state.available_books = []
if 'selected_books' not in st.session_state:
    st.session_state.selected_books = []
if 'analysis_params' not in st.session_state:
    st.session_state.analysis_params = {}

DEFAULT_MAE_THRESHOLD = 0.05
DEFAULT_VIG_MARKET_TOTAL = 1.071   # fallback if no two-sided lines found anywhere
MIN_ALT_LINES = 2

# ==============================================================================
# 1. UTILITY & MODELING FUNCTIONS
# ==============================================================================

def american_to_prob(odds):
    """Converts American odds to implied probability."""
    if odds is None or pd.isna(odds): return np.nan
    odds = float(odds)
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)

def prob_to_american(prob):
    """Converts a probability to American odds."""
    if prob is None or pd.isna(prob) or not (0 < prob < 1): return np.nan
    return round((100 / prob) - 100) if prob < 0.5 else round(prob / (1 - prob) * -100)

def find_power_k(p_over, p_under):
    """
    Find power-devig exponent k such that p_over^k + p_under^k = 1.
    Concentrates more vig on longshots (longshot bias), less on favourites.
    Falls back to k=1 (no adjustment) if solving fails.
    """
    try:
        return brentq(lambda k: p_over**k + p_under**k - 1, 0.001, 50)
    except (ValueError, RuntimeError):
        return 1.0

def find_most_balanced_two_way_market(market_df):
    """
    Finds the two-way market (over/under pair) closest to 50/50.
    Uses power devigging to compute the fair over probability.
    Returns: (line, over_odds, under_odds, market_total, fair_over_prob, k)
             or None if no two-way markets exist.
    """
    pivot = market_df.pivot_table(index='line', columns='type', values='odds')
    if 'over' not in pivot.columns or 'under' not in pivot.columns:
        return None

    two_way_markets = pivot[['over', 'under']].dropna()
    if two_way_markets.empty:
        return None

    balanced_scores = []
    for idx, row in two_way_markets.iterrows():
        over_prob  = american_to_prob(row['over'])
        under_prob = american_to_prob(row['under'])
        balance_score = abs(over_prob - 0.5) + abs(under_prob - 0.5)

        k = find_power_k(over_prob, under_prob)
        # p_over^k + p_under^k = 1 by construction, so p_over^k IS the fair prob
        fair_over_prob = over_prob ** k

        balanced_scores.append({
            'line': idx, 'over_odds': row['over'], 'under_odds': row['under'],
            'over_prob': over_prob, 'under_prob': under_prob,
            'balance_score': balance_score,
            'fair_over_prob': fair_over_prob, 'k': k
        })

    best = min(balanced_scores, key=lambda x: x['balance_score'])
    market_total = best['over_prob'] + best['under_prob']
    return (best['line'], best['over_odds'], best['under_odds'],
            market_total, best['fair_over_prob'], best['k'])


def calculate_consensus_fair_value(book_dataframes):
    """
    Equal-weight interpolation-based consensus anchor.

    Algorithm:
      1. For each book with a two-sided line: record (line, power-devigged fair_prob).
      2. Group by unique line; compute simple average fair_prob per unique line.
      3. Find the 50% crossing via linear interpolation between adjacent unique-line points.
      4. Fallback A — all points exactly 50%: equal-weighted average of all book lines.
      5. Fallback B — no crossing but not all 50%: use the point closest to 50%
         (keeps actual fair_prob rather than forcing 50%).

    Returns: (consensus_line, consensus_prob, contributions_list)
    """
    book_points  = []
    contributions = []

    for book_name, market_df in book_dataframes.items():
        result = find_most_balanced_two_way_market(market_df)
        if result is None:
            continue
        line, over_odds, under_odds, market_total, fair_over_prob, k = result
        book_points.append({'book': book_name, 'line': line, 'fair_prob': fair_over_prob})
        contributions.append({
            'book': book_name, 'line': line, 'fair_prob': fair_over_prob,
            'k': round(k, 4) if k else None,
            'american_odds': prob_to_american(fair_over_prob)
        })

    if not book_points:
        return None, None, []

    # Group by unique line — equal weight per book
    by_line = defaultdict(list)
    for pt in book_points:
        by_line[pt['line']].append(pt['fair_prob'])

    unique_points = sorted(
        [(line, sum(probs) / len(probs)) for line, probs in by_line.items()]
    )
    lines = [p[0] for p in unique_points]
    probs = [p[1] for p in unique_points]

    # Attempt interpolation for 50% crossing
    for i in range(len(unique_points) - 1):
        L0, p0 = lines[i],   probs[i]
        L1, p1 = lines[i+1], probs[i+1]
        if abs(p1 - p0) < 1e-9:
            continue  # flat segment — no directional info
        if (p0 < 0.50 < p1) or (p1 < 0.50 < p0):
            t = (0.50 - p0) / (p1 - p0)
            consensus_line = L0 + t * (L1 - L0)
            return consensus_line, 0.50, contributions

    # Fallback A: every unique-line point is exactly at 50%
    if all(abs(p - 0.50) < 1e-9 for p in probs):
        all_lines = [pt['line'] for pt in book_points]
        consensus_line = sum(all_lines) / len(all_lines)
        return consensus_line, 0.50, contributions

    # Fallback B: all on one side of 50% — use closest point with its actual fair_prob
    best_idx = min(range(len(probs)), key=lambda i: abs(probs[i] - 0.50))
    return lines[best_idx], probs[best_idx], contributions


def get_prob_from_model(params, line, dist_name):
    """Calculates P(X > line) for any supported distribution."""
    if dist_name in ('poisson', 'nbinom'):
        k = int(np.floor(line))
        if dist_name == 'poisson': prob_le_k = poisson.cdf(k, mu=params[0])
        else:                       prob_le_k = nbinom.cdf(k, n=params[0], p=params[1])
    else:
        k = line
        if   dist_name == 'norm':    prob_le_k = norm.cdf(k, loc=params[0], scale=params[1])
        elif dist_name == 'lognorm': prob_le_k = lognorm.cdf(k, s=params[0], loc=params[1], scale=params[2])
        elif dist_name == 'gamma':   prob_le_k = gamma.cdf(k, a=params[0], loc=params[1], scale=params[2])
        elif dist_name == 'skewnorm':prob_le_k = skewnorm.cdf(k, a=params[0], loc=params[1], scale=params[2])
        elif dist_name == 'skewt' and SKEWT_AVAILABLE:
            prob_le_k = skewt.cdf(k, a=params[0], df=params[1], loc=params[2], scale=params[3])
        else:
            return np.nan
    return 1.0 - prob_le_k


def fit_model(market_df, dist_name, target_col='fair_prob'):
    """
    Fits a distribution to power-devigged fair probabilities (over side only).
    No anchor penalty — location is corrected afterwards via solve_location_shift().
    """
    over_data    = market_df[market_df['type'] == 'over'].copy()
    lines        = over_data['line'].values
    target_probs = over_data[target_col].values

    if len(lines) == 0:
        return None

    closest_idx   = np.argmin(np.abs(target_probs - 0.5))
    mean_estimate = lines[closest_idx]

    models = {
        'poisson':  {'guess': [mean_estimate],              'bounds': [(0.1, None)]},
        'nbinom':   {'guess': [20, 0.5],                    'bounds': [(0.1, None), (0.01, 0.99)]},
        'norm':     {'guess': [mean_estimate, 5],           'bounds': [(None, None), (0.1, None)]},
        'lognorm':  {'guess': [0.5, 0, mean_estimate],      'bounds': [(0.01, None), (None, None), (0.1, None)]},
        'gamma':    {'guess': [2, 0, mean_estimate / 2],    'bounds': [(0.1, None), (None, None), (0.1, None)]},
        'skewnorm': {'guess': [0, mean_estimate, 5],        'bounds': [(None, None), (None, None), (0.1, None)]},
    }
    if SKEWT_AVAILABLE and dist_name == 'skewt':
        models['skewt'] = {
            'guess':  [0, 10, mean_estimate, 5],
            'bounds': [(None, None), (1, None), (None, None), (0.1, None)]
        }

    if dist_name not in models:
        return None

    config = models[dist_name]

    def error_fn(params):
        if any(np.isnan(params)): return 1e9
        model_probs = np.array([get_prob_from_model(params, x, dist_name) for x in lines])
        return float(np.sum((model_probs - target_probs) ** 2))

    result = minimize(error_fn, config['guess'], bounds=config['bounds'],
                      method='L-BFGS-B', options={'maxiter': 100})
    return result.x if result.success else None


def solve_location_shift(params, dist_name, target_line, target_prob):
    """
    Shifts the location parameter so that P(X > target_line) = target_prob.
    Shape and scale are preserved.
    """
    def make_params(loc_val):
        p = list(params)
        if   dist_name == 'norm':     p[0] = loc_val
        elif dist_name == 'lognorm':  p[1] = loc_val
        elif dist_name == 'gamma':    p[1] = loc_val
        elif dist_name == 'skewnorm': p[1] = loc_val
        elif dist_name == 'skewt':    p[2] = loc_val
        return p

    def objective(loc_val):
        return get_prob_from_model(make_params(loc_val), target_line, dist_name) - target_prob

    try:
        current_loc = (params[0] if dist_name == 'norm'
                       else params[2] if dist_name == 'skewt'
                       else params[1])
        new_loc = brentq(objective, current_loc - 30, current_loc + 30)
        return make_params(new_loc)
    except Exception:
        return params


# ==============================================================================
# 2. DISPLAY FUNCTIONS
# ==============================================================================

def display_results_table(results_df, selected_books):
    """Display results table; average row computed from selected books only."""
    if results_df is None or results_df.empty:
        return None

    res_df_actual = results_df[~results_df.get('is_single_line', False)]
    if res_df_actual.empty:
        st.warning("No model results available.")
        return None

    pivot_df = res_df_actual.pivot_table(
        index='book', columns='method', values='target_prob_over', aggfunc='first'
    )

    selected_pivot = pivot_df[pivot_df.index.isin(selected_books)]
    if not selected_pivot.empty:
        mean_row = selected_pivot.mean().to_frame().T
        mean_row.index = [f'AVERAGE ({len(selected_books)} books)']
    else:
        mean_row = pd.DataFrame(index=['AVERAGE (0 books)'], columns=pivot_df.columns)

    display_df = pivot_df.copy()
    display_df.index = [
        f"{'✓ ' if b in selected_books else '✗ '}{b}" for b in display_df.index
    ]

    final_table = pd.concat([display_df, mean_row])
    formatted   = final_table.map(
        lambda p: f"{prob_to_american(p):+.0f}" if pd.notnull(p) else "-"
    )
    st.dataframe(formatted, use_container_width=True)

    if 'Power' in mean_row.columns and not selected_pivot.empty:
        return mean_row['Power'].iloc[0]
    return None


# ==============================================================================
# 3. MAIN ANALYSIS FUNCTION
# ==============================================================================

def run_analysis(df, use_anchor, anchor_line, anchor_odds,
                 target_line, dist_type, mae_threshold, show_individual_plots):
    """
    Core analysis pipeline — v2.3:
      1. Load books; compute per-book power-devig k from two-sided market.
      2. Over-only books use the highest k observed across all other books.
      3. Compute consensus anchor via interpolation (or use manual anchor).
      4. For each book: power-devig all alt lines → fit distribution →
         shift location to anchor → report P(X > target_line).
    """
    # ── 1. Load book data ──────────────────────────────────────────────────────
    line_col   = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    total_cols = df.shape[1]

    book_dataframes    = {}
    book_alt_line_counts = {}
    books = []
    loaded_sigs = set()
    book_idx = 1

    for i in range(1, total_cols, 2):
        if i + 1 >= total_cols:
            break
        over_raw  = pd.to_numeric(df.iloc[:, i],   errors='coerce')
        under_raw = pd.to_numeric(df.iloc[:, i+1], errors='coerce')
        if over_raw.isna().all() and under_raw.isna().all():
            continue

        sig_df = pd.DataFrame({'o': over_raw, 'u': under_raw}).fillna(-9999)
        sig    = pd.util.hash_pandas_object(sig_df).sum()
        if sig in loaded_sigs:
            continue
        loaded_sigs.add(sig)

        col_name  = df.columns[i] if i < len(df.columns) else f"Book {book_idx}"
        book_name = col_name.replace(' over', '').replace(' Over', '').strip()
        if not book_name or book_name.lower() in ('', 'unnamed'):
            book_name = f"Book {book_idx}"
        books.append(book_name)

        temp_df   = pd.DataFrame({'line': line_col, 'over': over_raw, 'under': under_raw})
        market_df = pd.melt(temp_df, id_vars=['line'],
                            value_vars=['over', 'under'],
                            var_name='type', value_name='odds').dropna(subset=['odds'])
        book_dataframes[book_name]      = market_df
        book_alt_line_counts[book_name] = market_df['line'].nunique()
        book_idx += 1

    if not books:
        st.error("No valid book data found. Please check column format.")
        return

    st.success(f"Successfully processed {len(books)} unique book(s).")
    st.info("**Book Analysis:**\n" +
            "\n".join([f"- {b}: {book_alt_line_counts[b]} line(s)" for b in books]))

    # ── 2. Compute power k per book ────────────────────────────────────────────
    book_ks = {}   # book_name -> k  (only for books with a two-sided line)

    for book_name, market_df in book_dataframes.items():
        result = find_most_balanced_two_way_market(market_df)
        if result is not None:
            _, _, _, _, _, k = result
            if k is not None:
                book_ks[book_name] = k

    # Over-only books use the highest k (most conservative / highest vig)
    if book_ks:
        fallback_k = max(book_ks.values())
    else:
        # No two-sided lines anywhere — derive k from default vig assumption
        p_each = DEFAULT_VIG_MARKET_TOTAL / 2
        fallback_k = find_power_k(p_each, p_each)

    over_only_books = [b for b in books if b not in book_ks]
    if over_only_books:
        k_src = (f"max observed k = {fallback_k:.4f} "
                 f"(from {max(book_ks, key=book_ks.get)})" if book_ks
                 else f"default vig k = {fallback_k:.4f}")
        st.warning(
            f"⚠️ Over-only books (no two-sided line): {', '.join(over_only_books)}\n"
            f"   Using {k_src} for devigging — results are approximate upper bounds on fair prob."
        )

    # ── 3. Determine fair-value anchor ─────────────────────────────────────────
    if use_anchor:
        fair_anchor_line = anchor_line
        fair_anchor_prob = american_to_prob(anchor_odds)
        st.success(
            f"Manual anchor: Line {fair_anchor_line} @ {anchor_odds:+.0f} "
            f"({fair_anchor_prob:.2%})"
        )
    else:
        consensus_line, consensus_prob, contributions = \
            calculate_consensus_fair_value(book_dataframes)
        if consensus_line is not None:
            fair_anchor_line = consensus_line
            fair_anchor_prob = consensus_prob
            st.success(
                f"Consensus anchor: Line {fair_anchor_line:.3f} "
                f"@ {prob_to_american(fair_anchor_prob):+.0f} "
                f"({fair_anchor_prob:.2%})"
            )
            with st.expander("Consensus Calculation Details"):
                contrib_df = pd.DataFrame(contributions)
                st.dataframe(contrib_df)
        else:
            st.warning("Could not calculate consensus. Proceeding without anchor.")
            fair_anchor_line = None
            fair_anchor_prob = None

    # ── 4. Distributions to test ───────────────────────────────────────────────
    if dist_type == 'Discrete':
        models_to_test = ['poisson', 'nbinom']
    else:
        models_to_test = ['norm', 'lognorm', 'gamma', 'skewnorm']
        if SKEWT_AVAILABLE and max(book_alt_line_counts.values(), default=0) >= 5:
            models_to_test.append('skewt')

    # ── 5. Fitting loop ────────────────────────────────────────────────────────
    all_results      = []
    single_line_books = []

    progress_bar     = st.progress(0)
    total_iterations = len(books)

    for i, book in enumerate(books):
        progress_bar.progress(min((i + 1) / total_iterations, 1.0))

        market_df  = book_dataframes[book]
        k          = book_ks.get(book, fallback_k)
        num_lines  = book_alt_line_counts[book]

        if num_lines < MIN_ALT_LINES:
            single_line_books.append(book)
            if fair_anchor_line is not None and fair_anchor_prob is not None:
                all_results.append({
                    'book': book, 'method': 'Power', 'model': 'single_line',
                    'params': None, 'mae': 0,
                    'target_prob_over': fair_anchor_prob, 'is_single_line': True
                })
            continue

        # Power-devig all lines for this book
        working_df          = market_df.copy()
        working_df['prob']      = working_df['odds'].apply(american_to_prob)
        working_df['fair_prob'] = working_df['prob'].apply(lambda p: p ** k)

        best_model = None
        min_mae    = float('inf')

        for dist_name in models_to_test:
            try:
                params = fit_model(working_df, dist_name, target_col='fair_prob')
                if params is None:
                    continue

                # Shift distribution to consensus / manual anchor
                final_params = params
                if fair_anchor_prob is not None:
                    final_params = solve_location_shift(
                        params, dist_name, fair_anchor_line, fair_anchor_prob
                    )

                # MAE against power-devigged over probs
                over_df     = working_df[working_df['type'] == 'over']
                model_probs = np.array([
                    get_prob_from_model(final_params, x, dist_name)
                    for x in over_df['line'].values
                ])
                mae = float(np.mean(np.abs(over_df['fair_prob'].values - model_probs)))

                if mae < min_mae:
                    min_mae    = mae
                    best_model = {
                        'book': book, 'method': 'Power', 'model': dist_name,
                        'params': final_params, 'mae': mae, 'is_single_line': False
                    }
            except Exception:
                continue

        if best_model:
            best_model['target_prob_over'] = get_prob_from_model(
                best_model['params'], target_line, best_model['model']
            )
            all_results.append(best_model)

    progress_bar.empty()

    if single_line_books:
        st.warning(f"⚠️ Books with only 1 line (no model fitted): {', '.join(single_line_books)}")

    if not all_results:
        st.error("No valid models found.")
        return

    res_df = pd.DataFrame(all_results)

    # Store in session state for dynamic display
    st.session_state.results_df      = res_df
    st.session_state.available_books = [b for b in books if b not in single_line_books]
    st.session_state.selected_books  = st.session_state.available_books.copy()
    st.session_state.analysis_params = {
        'book_dataframes':    book_dataframes,
        'book_ks':            book_ks,
        'fallback_k':         fallback_k,
        'book_alt_line_counts': book_alt_line_counts,
        'fair_anchor_line':   fair_anchor_line,
        'fair_anchor_prob':   fair_anchor_prob,
        'target_line':        target_line,
        'line_col':           line_col,
        'mae_threshold':      mae_threshold,
        'single_line_books':  single_line_books,
    }


# ==============================================================================
# 4. PLOTTING
# ==============================================================================

def create_main_plot(results_df, selected_books, params):
    """Main visualisation: power-devigged scatter + fitted curves + anchor + target."""
    if results_df is None or params is None:
        return None

    book_dataframes = params['book_dataframes']
    book_ks         = params['book_ks']
    fallback_k      = params['fallback_k']
    fair_anchor_line = params['fair_anchor_line']
    fair_anchor_prob = params['fair_anchor_prob']
    target_line      = params['target_line']
    line_col         = params['line_col']
    mae_threshold    = params['mae_threshold']

    books = list(book_dataframes.keys())

    fig, ax = plt.subplots(figsize=(12, 6))
    try:
        cmap = matplotlib.colormaps['tab10']
    except Exception:
        cmap = plt.get_cmap('tab10')
    colors = {b: cmap(i % 10) for i, b in enumerate(books)}

    x_min   = line_col.min()
    x_max   = line_col.max()
    x_range = np.linspace(x_min - 2, x_max + 2, 200)

    res_df_actual = results_df[~results_df.get('is_single_line', False)]
    power_results = res_df_actual[res_df_actual['method'] == 'Power'].to_dict('records')

    # Fitted curves
    for res in power_results:
        if res['book'] not in selected_books:
            continue
        ls    = '-' if res['mae'] <= mae_threshold else ':'
        alpha = 0.8 if res['mae'] <= mae_threshold else 0.6
        y_vals = [get_prob_from_model(res['params'], x, res['model']) for x in x_range]
        ax.plot(x_range, y_vals, color=colors[res['book']], linestyle=ls, alpha=alpha,
                linewidth=1.5, label=f"{res['book']} ({res['model']})")

    # Power-devigged scatter points
    for book in books:
        k   = book_ks.get(book, fallback_k)
        mdf = book_dataframes[book].copy()
        mdf['prob']      = mdf['odds'].apply(american_to_prob)
        mdf['fair_prob'] = mdf['prob'].apply(lambda p: p ** k)
        over_pts  = mdf[mdf['type'] == 'over']
        m_alpha   = 0.5 if book in selected_books else 0.15
        ax.scatter(over_pts['line'], over_pts['fair_prob'],
                   color=colors[book], marker='o', s=30, alpha=m_alpha)

    # Average horizontal line
    sel_results = res_df_actual[
        (res_df_actual['method'] == 'Power') &
        (res_df_actual['book'].isin(selected_books))
    ]
    if not sel_results.empty:
        mean_prob = sel_results['target_prob_over'].mean()
        ax.axhline(y=mean_prob, color='black', linestyle='-', linewidth=2,
                   label=f'Average ({prob_to_american(mean_prob):+.0f})')
        ax.text(x_min, mean_prob + 0.01,
                f" Avg: {prob_to_american(mean_prob):+.0f}",
                fontsize=10, fontweight='bold')

    # Target line & anchor
    ax.axvline(x=target_line, color='purple', linestyle=':', linewidth=2,
               label=f'Target {target_line}')
    if fair_anchor_line is not None and fair_anchor_prob is not None:
        ax.scatter(fair_anchor_line, fair_anchor_prob, c='red', s=150,
                   marker='*', label='Anchor', zorder=10)

    # Axes
    ticks = np.arange(0.1, 1.0, 0.1)
    ax.set_ylabel("Fair American Odds")
    ax.set_xlabel("Line")
    ax.set_ylim(0.02, 0.98)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{prob_to_american(t):+.0f}' for t in ticks])

    ax2 = ax.twinx()
    ax2.set_ylabel("Probability (%)")
    ax2.set_ylim(0.02, 0.98)
    ax2.set_yticks(ticks)
    ax2.set_yticklabels([f'{t:.0%}' for t in ticks])

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              bbox_to_anchor=(1.08, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    return fig


# ==============================================================================
# 5. STREAMLIT UI
# ==============================================================================

st.set_page_config(layout="wide")
st.title("🎯 Advanced Prop Line Calculator v2.3")

with st.expander("ℹ️ v2.3 — Power-Method Devigging"):
    st.markdown("""
    **What changed in v2.3:**
    - **Power-method devigging** replaces the old constant-vig multiplicative approach.
      Exponent *k* is solved from the most balanced two-sided market at each book
      (`p_over^k + p_under^k = 1`), then applied to every alt line for that book.
      This correctly concentrates more vig on longshots and less on heavy favourites.
    - **Single analysis method** — removed redundant Multiplicative / Shape Retention
      columns. All books now go through: power devig → fit distribution → location
      shift to anchor.
    - **Over-only books** (no two-sided line) use the *highest k* seen across other
      books — the most conservative assumption about their vig level.
    - **Consensus anchor** uses equal-weight linear interpolation across book lines
      to find the 50 % crossing, rather than a weighted average of lines and probs.
    """)

with st.sidebar:
    st.header("1. Upload Data")
    st.markdown("""
    **CSV format:**
    * Col 1: Prop line value
    * Col 2/3: Book 1 Over / Under odds
    * Col 4/5: Book 2 Over / Under odds  *(repeat)*

    Under odds may be omitted (over-only books are supported).
    """)
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    st.header("2. Parameters")
    use_anchor = st.checkbox("Use Manual Anchor Point", value=False,
                             help="Provide a specific line + fair-value odds as the anchor.")
    if use_anchor:
        anchor_line = st.number_input("Anchor Line",  value=21.5, step=1.0, format="%.1f")
        anchor_odds = st.number_input("Anchor Odds (fair / devigged)", value=109)
    else:
        st.info("Consensus anchor calculated automatically from most balanced markets.")
        anchor_line = 0
        anchor_odds = 0

    target_line   = st.number_input("Target Line", value=18.5, step=1.0, format="%.1f")
    dist_type     = st.radio("Distribution Type", ('Discrete', 'Continuous'))
    mae_threshold = st.slider("Max MAE Threshold", 0.01, 0.1, DEFAULT_MAE_THRESHOLD, 0.01)

    st.markdown("---")
    show_individual = st.checkbox("Show Individual Book Plots", value=False)

    if st.button("Run Analysis", use_container_width=True):
        st.session_state.analysis_run   = True
        st.session_state.results_df     = None
        st.session_state.available_books = []
        st.session_state.selected_books  = []

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=',', engine='python',
                         on_bad_lines='skip', encoding='utf-8-sig')
        df.replace(['-', ''], np.nan, inplace=True)

        st.header("Data Preview")
        st.dataframe(df.head(), width=1500)

        if st.session_state.analysis_run:
            if st.session_state.results_df is None:
                run_analysis(df, use_anchor, anchor_line, anchor_odds,
                             target_line, dist_type, mae_threshold, show_individual)

            if (st.session_state.results_df is not None
                    and len(st.session_state.available_books) > 0):

                st.header("Results")
                st.subheader("Select Books for Average")
                selected = st.multiselect(
                    "Books to include in average:",
                    options=st.session_state.available_books,
                    default=st.session_state.selected_books,
                    key="book_selector"
                )
                st.session_state.selected_books = selected
                if not selected:
                    st.warning("⚠️ No books selected.")

                st.subheader("Fair Value Results")
                display_results_table(st.session_state.results_df, selected)

                st.header("Visualisations")
                fig = create_main_plot(
                    st.session_state.results_df,
                    selected,
                    st.session_state.analysis_params
                )
                if fig:
                    st.pyplot(fig)

                # Individual book detail plots
                if show_individual and st.session_state.analysis_params:
                    params          = st.session_state.analysis_params
                    book_dataframes = params['book_dataframes']
                    book_ks         = params['book_ks']
                    fallback_k      = params['fallback_k']
                    book_alt_line_counts = params['book_alt_line_counts']
                    single_line_books    = params['single_line_books']
                    line_col = params['line_col']
                    x_min    = line_col.min()
                    x_max    = line_col.max()
                    x_range  = np.linspace(x_min - 2, x_max + 2, 200)
                    ticks    = np.arange(0.1, 1.0, 0.1)

                    st.markdown("---")
                    st.subheader("Individual Book Details")

                    for book in book_dataframes:
                        if book in single_line_books:
                            st.info(f"**{book}**: 1 line — no model fitted")
                            continue

                        res_df_actual = st.session_state.results_df[
                            ~st.session_state.results_df.get('is_single_line', False)
                        ]
                        book_res = res_df_actual[
                            res_df_actual['book'] == book
                        ].to_dict('records')
                        if not book_res:
                            continue

                        k   = book_ks.get(book, fallback_k)
                        mdf = book_dataframes[book].copy()
                        mdf['prob']      = mdf['odds'].apply(american_to_prob)
                        mdf['fair_prob'] = mdf['prob'].apply(lambda p: p ** k)

                        fig_sub, ax_sub = plt.subplots(figsize=(10, 5))

                        raw_over = mdf[mdf['type'] == 'over']
                        ax_sub.scatter(raw_over['line'], raw_over['prob'],
                                       color='gray', marker='x', s=50, alpha=0.6,
                                       label='Raw implied (with vig)')
                        ax_sub.scatter(raw_over['line'], raw_over['fair_prob'],
                                       color='black', marker='o', s=50, alpha=0.8,
                                       label=f'Power-devigged fair (k={k:.3f})')

                        for res in book_res:
                            y_vals = [get_prob_from_model(res['params'], x, res['model'])
                                      for x in x_range]
                            ax_sub.plot(x_range, y_vals, linewidth=2,
                                        label=f"{res['method']} ({res['model']}, MAE={res['mae']:.3f})")

                        ax_sub.set_title(
                            f"{book} — {book_alt_line_counts[book]} lines  |  k={k:.4f}"
                        )
                        ax_sub.set_ylim(0.02, 0.98)
                        ax_sub.set_yticks(ticks)
                        ax_sub.set_yticklabels([f'{prob_to_american(t):+.0f}' for t in ticks])
                        ax_sub.axvline(x=params['target_line'],
                                       color='purple', linestyle=':', alpha=0.5)
                        if params['fair_anchor_line'] is not None:
                            ax_sub.scatter(params['fair_anchor_line'],
                                           params['fair_anchor_prob'],
                                           c='red', s=100, marker='*',
                                           label='Anchor', zorder=10)
                        ax_sub.legend()
                        ax_sub.grid(True, alpha=0.3)
                        st.pyplot(fig_sub)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        import traceback
        st.code(traceback.format_exc())
else:
    st.info("Upload a CSV file to begin.")
