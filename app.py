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
if 'last_excluded' not in st.session_state:
    st.session_state.last_excluded = set()

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


def _logit(p, eps=1e-6):
    """Log-odds of a probability, clamped away from 0/1 for numerical safety."""
    p = min(max(float(p), eps), 1.0 - eps)
    return np.log(p / (1.0 - p))


def estimate_book_fair_line(points, prob_target=0.50):
    """
    Estimate the line at which a single book prices `prob_target`, using its
    FULL power-devigged alt-line ladder rather than a single point.

    `points` is a list of (line, fair_prob) tuples for the over side.

    Rationale: for a wide class of underlying distributions, logit(P(X > line))
    is approximately linear in `line` through the central region. A weighted
    least-squares fit of logit(p) on line therefore gives a robust estimate of
    the median/mainline (where logit = 0) that uses the whole spread of the
    ladder, and degrades gracefully to interpolation when only a couple of lines
    are present. Points are weighted by p*(1-p) (Bernoulli variance, peaking at
    0.5), which trusts lines near the median and discounts noisy tails.

    Returns a dict describing the estimate (line50, slope, n, brackets, r2, ...).
    """
    # Collapse duplicate lines (average fair prob per unique line).
    by_line = defaultdict(list)
    for L, p in points:
        if pd.notna(L) and pd.notna(p):
            by_line[float(L)].append(float(p))
    uniq = sorted((L, sum(ps) / len(ps)) for L, ps in by_line.items())
    if not uniq:
        return None

    Ls = np.array([u[0] for u in uniq], dtype=float)
    ps = np.array([u[1] for u in uniq], dtype=float)
    n  = len(uniq)

    info = {
        'n': n,
        'line_min': float(Ls.min()),
        'line_max': float(Ls.max()),
        'brackets': bool(ps.max() >= prob_target >= ps.min()),
        'slope': None, 'line50': None, 'r2': None,
        'point': (float(Ls[0]), float(ps[0])) if n == 1 else None,
    }
    target_logit = _logit(prob_target)

    # Single line: can only self-anchor if it already sits at the target.
    if n < 2:
        info['line50'] = float(Ls[0]) if abs(ps[0] - prob_target) < 1e-6 else None
        return info

    # Weighted log-odds regression: logit(p) = a + b * line.
    y = np.array([_logit(p) for p in ps])
    w = np.clip(ps * (1.0 - ps), 1e-6, None)

    W    = w.sum()
    xbar = (w * Ls).sum() / W
    ybar = (w * y).sum() / W
    sxx  = (w * (Ls - xbar) ** 2).sum()
    sxy  = (w * (Ls - xbar) * (y - ybar)).sum()
    if sxx < 1e-12:
        return info

    b = sxy / sxx
    a = ybar - b * xbar
    info['slope'] = float(b)

    y_hat  = a + b * Ls
    ss_res = (w * (y - y_hat) ** 2).sum()
    ss_tot = (w * (y - ybar) ** 2).sum()
    info['r2'] = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else None

    if b < -1e-9:                       # well-behaved: prob decreases with line
        info['line50'] = float((target_logit - a) / b)
    else:                               # anomalous slope — fall back to nearest line
        info['line50'] = float(Ls[int(np.argmin(np.abs(ps - prob_target)))])
    return info


def calculate_consensus_fair_value(book_dataframes, book_ks=None,
                                   fallback_k=None, prob_target=0.50,
                                   discrete=False, book_fits=None,
                                   line_bounds=None):
    """
    Distribution-based consensus anchor.

    Two distinct kinds of information are separated and recombined:

      * LOCATION (mean) — only books that post a genuine two-sided market
        contribute to where the consensus line sits. Their power-devig is
        reliable (both sides present), so their fair probability at the posted
        line is trustworthy. One-sided ladders never move the location, since
        their absolute level is biased by the approximate over-only devig.

      * SHAPE (variance/skew) — taken from the actual fitted distributions, NOT
        a logit-linear slope. Every multi-line book (including over-only ladders)
        is fit upstream by run_analysis; those fits carry curvature, skew and —
        for discrete families — integer mass that a logit line cannot. The
        best-fitting multi-line book is the reference SHAPE donor.

    Per two-way book the implied fair line is the MEDIAN of a fitted distribution
    (where P(X > L) = prob_target):
      * a two-way book with its own multi-line fit uses that fit directly;
      * a single-line two-way book borrows the donor SHAPE, relocated to its own
        posted point via solve_location_shift(), then read for its median.
    The consensus line is the EQUAL-weighted average of those medians — every
    two-way book is one vote, regardless of how balanced its posted line is or
    how many alt lines it has.

    Continuous models anchor at the (fractional) consensus median.

    Discrete models (`discrete=True`) anchor on a REAL posted two-way line rather
    than an invented value: the nearest posted two-way line to the consensus
    centre, preferring half (X.5) lines and only using whole-number two-way lines
    if no half-line two-way market exists. Whole-number lines carry a push, so a
    book balanced at a whole number N is relocated to be symmetric about N (so
    Over N-0.5 == Under N+0.5) via relocate_to_two_way(). The anchor probability
    is the confidence-weighted P(X > anchor_line) across the two-way books.

    A logit-linear slope (estimate_book_fair_line) is retained only as a fallback
    when no distribution fit is available.

    `book_ks` / `fallback_k` / `book_fits` / `line_bounds` come from run_analysis.

    Returns: (consensus_line, consensus_prob, contributions_list)
    """
    book_ks      = book_ks or {}
    book_fits    = book_fits or {}
    target_logit = _logit(prob_target)

    # Widen the median search range beyond the observed lines.
    all_lines = [float(L) for m in book_dataframes.values()
                 for L in m['line'].dropna().tolist()]
    if line_bounds is not None:
        lo0, hi0 = line_bounds
    elif all_lines:
        lo0, hi0 = min(all_lines), max(all_lines)
    else:
        lo0, hi0 = 0.0, 100.0
    search_bounds = (lo0 - 30.0, hi0 + 30.0)

    # ── 1. Per-book devigged ladder (for fallback slope) and two-way point ─────
    per_book = {}
    for book_name, market_df in book_dataframes.items():
        k = book_ks.get(book_name, fallback_k)
        if k is None:
            res0 = find_most_balanced_two_way_market(market_df)
            k = res0[5] if res0 else 1.0

        over_df = market_df[market_df['type'] == 'over']
        pts = [(row['line'], american_to_prob(row['odds']) ** k)
               for _, row in over_df.iterrows() if pd.notna(row['odds'])]
        est = estimate_book_fair_line(pts, prob_target) if pts else None
        balanced = find_most_balanced_two_way_market(market_df)  # None if one-sided
        per_book[book_name] = {'k': k, 'est': est, 'balanced': balanced}

    # ── 2. Choose the reference SHAPE donor: best multi-line fit ───────────────
    #    (most lines, tie-broken by lowest unshifted MAE).
    donor = None
    for book_name, fit in book_fits.items():
        if not fit:
            continue
        n_lines = per_book.get(book_name, {}).get('est')
        n_lines = n_lines['n'] if n_lines else 0
        key = (n_lines, -fit.get('mae', float('inf')))
        if donor is None or key > donor['key']:
            donor = {'key': key, 'book': book_name,
                     'dist': fit['dist_name'], 'params': fit['params']}

    # Pooled logit slope — fallback only.
    slope_donors = [(e['slope'], e['n']) for info in per_book.values()
                    if (e := info['est']) and e['slope'] is not None
                    and e['slope'] < 0 and e['n'] >= 2]
    global_slope = (sum(s * n for s, n in slope_donors) / sum(n for _, n in slope_donors)
                    if slope_donors else None)

    # ── 3. LOCATION from two-way books only, via fitted-distribution medians ───
    contributions = []
    weighted      = []
    for book_name, info in per_book.items():
        balanced = info['balanced']
        e        = info['est']
        own_fit  = book_fits.get(book_name)

        if balanced is None:
            # One-sided book — contributes shape only, never the location.
            if own_fit or (e and e['slope'] is not None):
                contributions.append({
                    'book': book_name,
                    'role': 'shape donor' if own_fit else 'dispersion only',
                    'lines_used': e['n'] if e else 0,
                    'posted_line': None, 'fair_prob': None, 'fair_line_50%': None,
                    'model': own_fit['dist_name'] if own_fit else None,
                    'k': round(info['k'], 4) if info['k'] else None,
                    'weight': 0.0,
                })
            continue

        line, _o, _u, _mt, fair_prob, _k = balanced

        # Pick this book's distribution: its own fit, else the borrowed donor.
        dist = params = None
        if own_fit:
            dist, params, src = own_fit['dist_name'], own_fit['params'], 'own fit'
        elif donor is not None:
            dist = donor['dist']
            params = relocate_to_two_way(donor['params'], dist, line, fair_prob, discrete)
            src = f"borrowed {donor['book']}"

        median = model_median(params, dist, search_bounds, prob_target) \
                 if params is not None else None

        # Fallback to the logit-linear slope if no usable distribution median.
        if median is None:
            own_slope = (e['slope'] if e and e['slope'] is not None
                         and e['slope'] < 0 and e['n'] >= 2 else None)
            slope_fb = own_slope if own_slope is not None else global_slope
            if slope_fb is not None:
                median = line + (target_logit - _logit(fair_prob)) / slope_fb
                src = 'logit slope'
            else:
                median = line
                src = 'posted line'

        # Every two-way book is one equal input to the consensus location,
        # regardless of its posted line's balance or its own alt-line depth.
        n_lines = e['n'] if e else 1
        weight  = 1.0

        contrib = {
            'book': book_name,
            'role': f'mean ({src})',
            'lines_used': n_lines,
            'posted_line': round(line, 3),
            'fair_prob': round(fair_prob, 4),
            'fair_line_50%': round(median, 3),
            'model': dist,
            'k': round(info['k'], 4) if info['k'] else None,
            'weight': round(weight, 3),
        }
        contributions.append(contrib)
        weighted.append({'median': median, 'dist': dist, 'params': params,
                         'weight': weight, 'contrib': contrib})

    if not weighted:
        return None, None, contributions

    total_w = sum(w['weight'] for w in weighted)
    if total_w > 0:
        consensus_line = sum(w['median'] * w['weight'] for w in weighted) / total_w
    else:
        consensus_line = sum(w['median'] for w in weighted) / len(weighted)
        for w in weighted:
            w['weight'] = 1.0

    # Continuous models anchor at the fractional consensus median.
    if not discrete:
        return consensus_line, prob_target, contributions

    # Discrete: anchor on a REAL posted two-way line rather than an invented X.5.
    # Whole-number lines carry a push, so a half (X.5) line is unambiguous and
    # preferred; only fall back to whole-number two-way lines if no half-line
    # two-way market exists anywhere.
    two_way_lines = set()
    for book_name, info in per_book.items():
        if info['balanced'] is None:
            continue
        piv = book_dataframes[book_name].pivot_table(
            index='line', columns='type', values='odds')
        if 'over' in piv.columns and 'under' in piv.columns:
            for L, row in piv.iterrows():
                if pd.notna(row.get('over')) and pd.notna(row.get('under')):
                    two_way_lines.add(float(L))

    half_lines  = sorted(L for L in two_way_lines if abs(L - round(L)) > 1e-9)
    whole_lines = sorted(L for L in two_way_lines if abs(L - round(L)) <= 1e-9)
    candidates  = half_lines if half_lines else whole_lines

    if candidates:
        # Nearest posted two-way line to the consensus centre.
        anchor_line = min(candidates, key=lambda L: abs(L - consensus_line))
    else:
        anchor_line = float(np.floor(consensus_line) + 0.5)

    # Anchor probability = confidence-weighted P(X > anchor_line) across the
    # two-way books, read from each book's (push-aware) fitted CDF.
    num = den = 0.0
    for w in weighted:
        p_at = (get_prob_from_model(w['params'], anchor_line, w['dist'])
                if w['params'] is not None else None)
        if p_at is None or not (0.0 <= p_at <= 1.0) or np.isnan(p_at):
            p_at = prob_target   # degenerate — fall back to the target
        w['contrib']['prob_at_anchor'] = round(float(p_at), 4)
        num += p_at * w['weight']
        den += w['weight']
    consensus_prob = num / den if den > 0 else prob_target
    return float(anchor_line), consensus_prob, contributions


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


def model_median(params, dist_name, line_bounds, prob_target=0.50, n_grid=1200):
    """
    Line at which a fitted distribution prices P(X > line) == prob_target.

    Evaluated on a grid and linearly interpolated rather than root-solved, which
    stays robust for the step-shaped CDFs of discrete distributions (where a
    smooth solver would stall on flat segments). Returns None if the target is
    never crossed within line_bounds.
    """
    lo, hi = line_bounds
    xs = np.linspace(lo, hi, n_grid)
    ps = np.array([get_prob_from_model(params, float(x), dist_name) for x in xs])

    # Discrete CDFs are flat across each integer interval, so the target can be
    # hit over a whole range — return the centre of that range (e.g. the 27.5
    # mid-point of [27, 28) rather than its left edge).
    at_target = np.isclose(ps, prob_target, atol=1e-6)
    if at_target.any():
        idx = np.where(at_target)[0]
        return float((xs[idx[0]] + xs[idx[-1]]) / 2.0)

    # Otherwise interpolate the strict crossing (continuous distributions).
    for i in range(len(xs) - 1):
        a, b = ps[i] - prob_target, ps[i + 1] - prob_target
        if a * b < 0 and ps[i] != ps[i + 1]:
            t = (prob_target - ps[i]) / (ps[i + 1] - ps[i])
            return float(xs[i] + t * (xs[i + 1] - xs[i]))
    return None


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
    Shifts the location (or rate) parameter so that P(X > target_line) = target_prob.
    Shape and scale are preserved.

    For continuous distributions: solves for the location parameter.
    For discrete distributions (poisson, nbinom): solves for the rate/probability
    parameter directly, since they have no location parameter.
    """
    # ── Discrete distributions — solve for rate parameter ─────────────────────
    if dist_name == 'poisson':
        k_floor = int(np.floor(target_line))
        def obj_pois(mu):
            return (1.0 - poisson.cdf(k_floor, mu=mu)) - target_prob
        try:
            new_mu = brentq(obj_pois, 0.01, 500)
            return [new_mu]
        except Exception:
            return params

    if dist_name == 'nbinom':
        k_floor = int(np.floor(target_line))
        n = params[0]
        def obj_nb(p_val):
            return (1.0 - nbinom.cdf(k_floor, n=n, p=p_val)) - target_prob
        try:
            new_p = brentq(obj_nb, 0.001, 0.999)
            return [n, new_p]
        except Exception:
            return params

    # ── Continuous distributions — solve for location parameter ───────────────
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

    # P(X > target_line) is monotonic in the location parameter, so widen the
    # search window around the current location until the root is bracketed.
    # (A fixed ±30 window silently failed when relocating a borrowed donor shape
    # whose centre was far from the target line.)
    current_loc = (params[0] if dist_name == 'norm'
                   else params[2] if dist_name == 'skewt'
                   else params[1])
    for half in (30, 60, 120, 250, 500, 1000):
        lo, hi = current_loc - half, current_loc + half
        try:
            f_lo, f_hi = objective(lo), objective(hi)
        except Exception:
            continue
        if np.isnan(f_lo) or np.isnan(f_hi) or f_lo * f_hi > 0:
            continue
        try:
            return make_params(brentq(objective, lo, hi))
        except Exception:
            break
    return params


def relocate_to_two_way(params, dist_name, line, fair_over_prob, discrete):
    """
    Relocate a distribution to match a posted two-way market at `line`.

    Half (X.5) lines have no push, so this is just P(X > line) = fair_over_prob
    (delegated to solve_location_shift).

    Whole-number lines N carry a push at X = N: the two-way market prices
    Over N (X ≥ N+1) vs Under N (X ≤ N-1) and refunds X = N. The fair over price
    is therefore P(X≥N+1) / (P(X≥N+1) + P(X≤N-1)), NOT P(X≥N+1) alone. Matching
    that ratio keeps a balanced book symmetric about N (median ≈ N) instead of
    being pushed off the whole number. Only meaningful for discrete families.
    """
    is_whole = abs(line - round(line)) < 1e-9
    if discrete and is_whole and dist_name in ('poisson', 'nbinom'):
        N = int(round(line))

        def ratio_resid(test_params):
            a = get_prob_from_model(test_params, N, dist_name)            # P(X ≥ N+1)
            b = 1.0 - get_prob_from_model(test_params, N - 1, dist_name)  # P(X ≤ N-1)
            tot = a + b
            return (a / tot if tot > 0 else 0.0) - fair_over_prob

        try:
            if dist_name == 'poisson':
                return [brentq(lambda mu: ratio_resid([mu]), 0.01, 500)]
            n = params[0]   # nbinom: preserve shape n, solve success prob p
            return [n, brentq(lambda p: ratio_resid([n, p]), 1e-4, 1 - 1e-4)]
        except Exception:
            return params

    return solve_location_shift(params, dist_name, line, fair_over_prob)


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
                 target_line, dist_type, mae_threshold, show_individual_plots,
                 consensus_excluded=None):
    """
    Core analysis pipeline — v2.4:
      1. Load books; compute per-book power-devig k from two-sided market.
      2. Over-only books use the highest k observed across all other books.
      3. Fit each multi-line book's distributions once (unshifted).
      4. Compute consensus anchor from the fitted distributions (or manual anchor).
         Books in `consensus_excluded` are dropped from the consensus only — they
         are still fit, shifted and displayed, just no longer influence the anchor.
      5. For each book: shift fit to anchor → report P(X > target_line).
    """
    consensus_excluded = set(consensus_excluded or [])
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

    # ── 3. Distributions to test ───────────────────────────────────────────────
    if dist_type == 'Discrete':
        models_to_test = ['poisson', 'nbinom']
    else:
        models_to_test = ['norm', 'lognorm', 'gamma', 'skewnorm']
        if SKEWT_AVAILABLE and max(book_alt_line_counts.values(), default=0) >= 5:
            models_to_test.append('skewt')

    # ── 4. Pre-fit each multi-line book's distributions (unshifted) ────────────
    #    Fit once here so the consensus can use real distribution shapes and the
    #    evaluation loop below can reuse the same fits without refitting.
    book_working  = {}   # book -> power-devigged working df
    book_all_fits = {}   # book -> {dist_name: params}
    book_best_fit = {}   # book -> {'dist_name', 'params', 'mae'} (best unshifted)
    for book in books:
        if book_alt_line_counts[book] < MIN_ALT_LINES:
            continue
        k   = book_ks.get(book, fallback_k)
        wdf = book_dataframes[book].copy()
        wdf['prob']      = wdf['odds'].apply(american_to_prob)
        wdf['fair_prob'] = wdf['prob'].apply(lambda p: p ** k)
        book_working[book] = wdf
        over_df = wdf[wdf['type'] == 'over']

        fits, best = {}, None
        for dist_name in models_to_test:
            try:
                params = fit_model(wdf, dist_name, target_col='fair_prob')
            except Exception:
                params = None
            if params is None:
                continue
            fits[dist_name] = params
            mp  = np.array([get_prob_from_model(params, x, dist_name)
                            for x in over_df['line'].values])
            mae = float(np.mean(np.abs(over_df['fair_prob'].values - mp)))
            if best is None or mae < best['mae']:
                best = {'dist_name': dist_name, 'params': params, 'mae': mae}
        book_all_fits[book] = fits
        if best:
            book_best_fit[book] = best

    # ── 5. Determine fair-value anchor ─────────────────────────────────────────
    if use_anchor:
        fair_anchor_line = anchor_line
        fair_anchor_prob = american_to_prob(anchor_odds)
        st.success(
            f"Manual anchor: Line {fair_anchor_line} @ {anchor_odds:+.0f} "
            f"({fair_anchor_prob:.2%})"
        )
    else:
        # Consensus uses every book except those the user has de-selected.
        cons_bdf  = {b: m for b, m in book_dataframes.items()
                     if b not in consensus_excluded}
        cons_fits = {b: f for b, f in book_best_fit.items()
                     if b not in consensus_excluded}
        consensus_line, consensus_prob, contributions = \
            calculate_consensus_fair_value(
                cons_bdf, book_ks, fallback_k,
                discrete=(dist_type == 'Discrete'),
                book_fits=cons_fits,
                line_bounds=(float(line_col.min()), float(line_col.max())))
        if consensus_excluded:
            st.caption(f"Consensus excludes de-selected book(s): "
                       f"{', '.join(sorted(consensus_excluded))}")
        if consensus_line is not None:
            fair_anchor_line = consensus_line
            fair_anchor_prob = consensus_prob
            st.success(
                f"Consensus anchor: Line {fair_anchor_line:.3f} "
                f"@ {prob_to_american(fair_anchor_prob):+.0f} "
                f"({fair_anchor_prob:.2%})"
            )
            with st.expander("Consensus Calculation Details"):
                st.caption(
                    "Only **two-way books** set the consensus location (`role = mean …`). "
                    "Each book's fair line (`fair_line_50%`) is the **median of a fitted "
                    "distribution** (`model`): a two-way book with ≥2 lines uses its own "
                    "fit; a single-line two-way book borrows the best multi-line fit "
                    "(`role = shape donor`), relocated to its posted point. The consensus "
                    "is the **equal-weighted** average of those medians — every two-way "
                    "book is one vote, regardless of how balanced its line is or how many "
                    "alt lines it has. For **Discrete** models the anchor is placed on the "
                    "nearest **posted two-way line** (preferring X.5 lines; whole-number "
                    "lines are treated as a push and kept symmetric), and the anchor "
                    "probability (`prob_at_anchor`) is the equal-weighted P(X > line) "
                    "there — generally not 50%."
                )
                contrib_df = pd.DataFrame(contributions)
                st.dataframe(contrib_df)
        else:
            st.warning("Could not calculate consensus. Proceeding without anchor.")
            fair_anchor_line = None
            fair_anchor_prob = None

    # ── 6. Evaluation: shift each pre-fit to the anchor, pick best, score ──────
    all_results      = []
    single_line_books = []

    progress_bar     = st.progress(0)
    total_iterations = len(books)

    for i, book in enumerate(books):
        progress_bar.progress(min((i + 1) / total_iterations, 1.0))

        num_lines = book_alt_line_counts[book]

        if num_lines < MIN_ALT_LINES:
            single_line_books.append(book)
            if fair_anchor_line is not None and fair_anchor_prob is not None:
                all_results.append({
                    'book': book, 'method': 'Power', 'model': 'single_line',
                    'params': None, 'mae': 0,
                    'target_prob_over': fair_anchor_prob, 'is_single_line': True
                })
            continue

        working_df = book_working[book]
        over_df    = working_df[working_df['type'] == 'over']

        best_model = None
        min_mae    = float('inf')

        for dist_name, params in book_all_fits.get(book, {}).items():
            try:
                # Shift the (already-fit) distribution to consensus / manual anchor
                final_params = params
                if fair_anchor_prob is not None:
                    final_params = solve_location_shift(
                        params, dist_name, fair_anchor_line, fair_anchor_prob
                    )

                # MAE against power-devigged over probs
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
    # NB: selected_books is managed by the UI so a consensus re-run does not reset it.
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
st.title("🎯 Advanced Prop Line Calculator v2.4")

with st.expander("ℹ️ v2.4 — Distribution-Based Consensus Anchor"):
    st.markdown("""
    **What changed in v2.4** (automatic anchor, used when *Manual Anchor Point* is off):

    The consensus separates two kinds of information and recombines them, using the
    **actual fitted distributions** rather than a logit-linear approximation:

    - **Location comes only from two-way books.** Books that post a genuine
      two-sided market have reliable power-devig (both sides present), so only they
      move the consensus mean. Over-only books no longer pull the location — their
      one-sided devig has a biased absolute level.
    - **Shape comes from the fitted distributions.** Every multi-line book
      (including over-only ladders) is fit upstream; those fits carry curvature,
      skew, and — for discrete families — integer mass that a straight log-odds line
      cannot. The best-fitting multi-line book is the reference *shape donor*.
    - **Each two-way book's fair line is a fitted median.** A two-way book with ≥2
      lines uses its own fitted distribution's median (where `P(X > L) = 0.5`); a
      single-line two-way book borrows the donor shape, relocated to its posted
      point, and reads that median. The consensus is the **equal-weighted** average
      of those medians — every two-way book is one vote, regardless of how balanced
      its posted line is or how many alt lines it has. Deriving the median from a
      real fitted distribution (rather than a linear slope) is materially more
      accurate when projecting offset two-way lines over many points, because real
      stat distributions are skewed.
    - **Discrete models anchor on a real posted two-way line.** When *Distribution
      Type* is **Discrete**, the anchor is the nearest posted two-way line to the
      consensus centre — preferring half (X.5) lines and only using whole-number
      two-way lines when no X.5 two-way market exists, so the anchor is never an
      invented value. Whole-number lines carry a push at X = N, so a book balanced
      there is kept symmetric about N (Over N−0.5 == Under N+0.5). The anchor
      probability is the consensus P(X > line) at that line (generally not 50%).

    **Carried over from v2.3 — Power-method devigging:**
    - Exponent *k* is solved from the most balanced two-sided market at each book
      (`p_over^k + p_under^k = 1`), then applied to every alt line for that book,
      concentrating more vig on longshots and less on heavy favourites.
    - **Over-only books** use the *highest k* seen across other books — the most
      conservative assumption about their vig level.
    - All books go through: power devig → fit distribution → location shift to anchor.
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
        st.session_state.last_excluded   = set()
        st.session_state.pop("book_selector", None)  # reset multiselect widget

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=',', engine='python',
                         on_bad_lines='skip', encoding='utf-8-sig')
        df.replace(['-', ''], np.nan, inplace=True)

        st.header("Data Preview")
        st.dataframe(df.head(), width=1500)

        if st.session_state.analysis_run:
            if st.session_state.results_df is None:
                # Fresh run from the button: consensus uses every book.
                run_analysis(df, use_anchor, anchor_line, anchor_odds,
                             target_line, dist_type, mae_threshold, show_individual,
                             consensus_excluded=set())
                st.session_state.selected_books = list(st.session_state.available_books)
                st.session_state.last_excluded  = set()

            if (st.session_state.results_df is not None
                    and len(st.session_state.available_books) > 0):

                st.header("Results")
                st.subheader("Select Books for Average")
                st.caption("De-selecting a book also removes it from the consensus "
                           "anchor (the analysis re-runs automatically).")
                selected = st.multiselect(
                    "Books to include in average & consensus:",
                    options=st.session_state.available_books,
                    default=st.session_state.selected_books,
                    key="book_selector"
                )
                st.session_state.selected_books = selected
                if not selected:
                    st.warning("⚠️ No books selected.")

                # If the selection changed, re-run so the consensus anchor (and the
                # location shift it drives for every book) reflects only the kept books.
                excluded = set(st.session_state.available_books) - set(selected)
                if not use_anchor and excluded != st.session_state.last_excluded:
                    run_analysis(df, use_anchor, anchor_line, anchor_odds,
                                 target_line, dist_type, mae_threshold, show_individual,
                                 consensus_excluded=excluded)
                    st.session_state.last_excluded  = excluded
                    st.session_state.selected_books = selected

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
