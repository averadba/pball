import time
import random
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


# ----------------------------- Constants --------------------------------- #

WHITE_BALLS = list(range(1, 70))   # 1–69
POWER_BALLS = list(range(1, 27))   # 1–26

THEORETICAL_WHITE_INCLUSION = 5.0 / 69.0   # P(a specific white ball appears among the 5 whites)
THEORETICAL_POWERBALL = 1.0 / 26.0         # P(a specific powerball is drawn)


# ----------------------- Main Simulation Functions ------------------------ #

def simulate_powerball_draws(
    n_draws: int,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate n_draws Powerball drawings.

    Returns:
        white_counts: length-69 array with counts per white ball (index 0 -> ball 1)
        power_counts: length-26 array with counts per Powerball (index 0 -> powerball 1)
    """
    if seed is not None:
        random.seed(seed)

    white_counts = np.zeros(69, dtype=int)
    power_counts = np.zeros(26, dtype=int)

    for _ in range(n_draws):
        white_draw = random.sample(WHITE_BALLS, 5)
        for w in white_draw:
            white_counts[w - 1] += 1

        pb = random.choice(POWER_BALLS)
        power_counts[pb - 1] += 1

    return white_counts, power_counts


def generate_random_tickets(n_tickets: int) -> List[Tuple[List[int], int]]:
    """Mode A: Pure random ticket generation."""
    tickets: List[Tuple[List[int], int]] = []
    for _ in range(n_tickets):
        white_draw = sorted(random.sample(WHITE_BALLS, 5))
        pb = random.choice(POWER_BALLS)
        tickets.append((white_draw, pb))
    return tickets


def generate_high_prob_tickets(
    white_counts: np.ndarray,
    power_counts: np.ndarray,
    n_tickets: int,
    top_white_pool: int = 20,
    top_power_pool: int = 10,
) -> List[Tuple[List[int], int]]:
    """Mode B: Ticket generation biased to the most frequent numbers in the simulation."""
    white_indices_sorted = sorted(range(len(white_counts)), key=lambda i: white_counts[i], reverse=True)
    power_indices_sorted = sorted(range(len(power_counts)), key=lambda i: power_counts[i], reverse=True)

    top_white_indices = white_indices_sorted[:top_white_pool]
    top_power_indices = power_indices_sorted[:top_power_pool]

    top_white_numbers = [i + 1 for i in top_white_indices]
    top_power_numbers = [i + 1 for i in top_power_indices]

    white_weights = [white_counts[i] for i in top_white_indices]
    power_weights = [power_counts[i] for i in top_power_indices]

    tickets: List[Tuple[List[int], int]] = []
    for _ in range(n_tickets):
        selected_white = set()
        while len(selected_white) < 5:
            chosen = random.choices(top_white_numbers, weights=white_weights, k=1)[0]
            selected_white.add(chosen)

        chosen_power = random.choices(top_power_numbers, weights=power_weights, k=1)[0]
        tickets.append((sorted(selected_white), chosen_power))

    return tickets


# -------------------- Probability Tables (Main UI) ------------------------- #

def build_probability_tables(
    white_counts: np.ndarray,
    power_counts: np.ndarray,
    n_draws: int,
    include_theoretical: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    white_df = pd.DataFrame({
        "Number": np.arange(1, 70),
        "Count": white_counts.astype(int),
    })
    white_df["Empirical per-draw inclusion %"] = (white_df["Count"] / float(n_draws)) * 100.0
    white_df["Empirical share of white selections %"] = (white_df["Count"] / float(n_draws * 5)) * 100.0

    if include_theoretical:
        white_df["Theoretical per-draw inclusion %"] = THEORETICAL_WHITE_INCLUSION * 100.0
        white_df["Theoretical share of selections %"] = (1.0 / 69.0) * 100.0

    power_df = pd.DataFrame({
        "Powerball": np.arange(1, 27),
        "Count": power_counts.astype(int),
    })
    power_df["Empirical per-draw probability %"] = (power_df["Count"] / float(n_draws)) * 100.0

    if include_theoretical:
        power_df["Theoretical per-draw probability %"] = THEORETICAL_POWERBALL * 100.0

    return white_df, power_df


def format_percent_cols(df: pd.DataFrame, cols: List[str], decimals: int = 3) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{float(x):.{decimals}f}%")
    return out


# --------------------- Sampling Distribution Module ------------------------ #

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.
    Returns (lower, upper) in [0,1].
    """
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1.0 + (z**2) / n
    center = (phat + (z**2) / (2*n)) / denom
    half = (z * np.sqrt((phat*(1-phat) / n) + (z**2) / (4*n*n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def simulate_one_rep_counts_numpy(rng: np.random.Generator, draws: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    One repetition using numpy RNG.
    Loop per draw (still) but faster than Python's random for repeated reps.
    """
    white_counts = np.zeros(69, dtype=int)
    power_counts = np.zeros(26, dtype=int)

    for _ in range(draws):
        whites = rng.choice(69, size=5, replace=False)  # 0..68
        white_counts[whites] += 1

        pb = rng.integers(0, 26)  # 0..25
        power_counts[pb] += 1

    return white_counts, power_counts


def run_sampling_distribution(
    reps: int,
    draws_per_rep: int,
    selected_white: int,
    selected_power: int,
    seed: Optional[int]
) -> Dict[str, object]:
    """
    Repeat simulations N times and return distributions for selected ball probabilities,
    plus Top10 sequences for animation.
    """
    base_seed = seed if seed is not None else int(time.time())
    rng = np.random.default_rng(base_seed)

    white_probs = np.zeros(reps, dtype=float)  # per-draw inclusion estimate for selected white
    power_probs = np.zeros(reps, dtype=float)  # per-draw probability estimate for selected powerball

    top10_whites_per_rep: List[List[int]] = []
    top10_powers_per_rep: List[List[int]] = []

    for r in range(reps):
        # Derive a new RNG for each rep (reproducible across reps)
        rep_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        wc, pc = simulate_one_rep_counts_numpy(rep_rng, draws_per_rep)

        w_count = int(wc[selected_white - 1])
        p_count = int(pc[selected_power - 1])

        white_probs[r] = w_count / float(draws_per_rep)
        power_probs[r] = p_count / float(draws_per_rep)

        top10_w = list((np.argsort(-wc)[:10] + 1).astype(int))
        top10_p = list((np.argsort(-pc)[:10] + 1).astype(int))
        top10_whites_per_rep.append(top10_w)
        top10_powers_per_rep.append(top10_p)

    # CI for the *current* run would be per rep; but we’ll show:
    # - CI around the mean probability using normal approx on the rep distribution (informal), AND
    # - Wilson CI for one rep is most interpretable; we’ll show Wilson for the last rep + also show expected.
    last_wc = None
    last_pc = None

    return {
        "white_probs": white_probs,
        "power_probs": power_probs,
        "top10_whites_per_rep": top10_whites_per_rep,
        "top10_powers_per_rep": top10_powers_per_rep,
        "seed_used": base_seed,
    }


def convergence_series(
    total_draws: int,
    selected_white: int,
    selected_power: int,
    seed: Optional[int],
    points: int = 60
) -> pd.DataFrame:
    """
    Build convergence data for empirical vs theoretical probabilities over cumulative draws.
    We record values at `points` checkpoints.
    """
    rng = np.random.default_rng(seed if seed is not None else int(time.time()))
    checkpoints = np.unique(np.linspace(1, total_draws, num=points, dtype=int))

    w_hits = 0
    p_hits = 0
    rows = []

    cp_idx = 0
    next_cp = checkpoints[cp_idx]

    for d in range(1, total_draws + 1):
        whites = rng.choice(69, size=5, replace=False) + 1
        pb = int(rng.integers(1, 27))

        if selected_white in whites:
            w_hits += 1
        if pb == selected_power:
            p_hits += 1

        if d == next_cp:
            rows.append({
                "Draws": d,
                "Empirical (White inclusion)": w_hits / float(d),
                "Theoretical (White inclusion)": THEORETICAL_WHITE_INCLUSION,
                "Empirical (Powerball)": p_hits / float(d),
                "Theoretical (Powerball)": THEORETICAL_POWERBALL,
            })
            cp_idx += 1
            if cp_idx >= len(checkpoints):
                break
            next_cp = checkpoints[cp_idx]

    return pd.DataFrame(rows)


# ------------------------------- App -------------------------------------- #

def main():
    st.set_page_config(
        page_title="Powerball Simulator & Ticket Generator",
        page_icon="🎱",
        layout="wide",
    )

    st.title("🎱 Powerball Simulator & Ticket Generator")

    st.markdown(
        """
        **What this app does**
        - Simulates Powerball drawings and shows **distributions**
        - Generates up to **5 tickets**:
          - **Mode A:** Pure random tickets (true to the lottery mechanics)
          - **Mode B:** “High-probability” tickets **based on this simulation’s** most frequent numbers (for fun/experiments)

        **Reality check:** In the real lottery, every valid combination is equally likely.
        """
    )

    # ---------------- Sidebar ---------------- #
    st.sidebar.header("Core Settings")

    mode = st.sidebar.radio(
        "Ticket generation mode",
        options=["Mode A – Pure Random", "Mode B – High-Probability (Simulation-based)"],
        index=0,
    )

    n_draws = st.sidebar.slider(
        "Number of simulated drawings (main)",
        min_value=1_000,
        max_value=200_000,
        value=50_000,
        step=1_000,
    )

    n_tickets = st.sidebar.slider(
        "Number of tickets to generate",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
    )

    seed_input = st.sidebar.text_input(
        "Random seed (optional)",
        value="",
        help="Use an integer for reproducible results."
    )

    st.sidebar.subheader("Probability tables")
    sort_mode = st.sidebar.radio(
        "Sort numbers by",
        options=["Hot (most frequent)", "Cold (least frequent)"],
        index=0,
    )
    include_theoretical = st.sidebar.checkbox("Show theoretical probabilities", value=True)
    top_n_table = st.sidebar.selectbox("Show rows", options=[10, 15, 20, 30, "All"], index=1)

    run_main = st.sidebar.button("Run Main Simulation")

    # Parse seed
    seed: Optional[int] = None
    if seed_input.strip():
        try:
            seed = int(seed_input.strip())
        except ValueError:
            st.warning("Seed must be an integer. Ignoring seed.")
            seed = None

    if not run_main:
        st.info("Click **Run Main Simulation** in the sidebar to generate charts, tables, and tickets.")
        st.markdown("---")
        st.subheader("🧪 Sampling Distribution Module")
        st.caption("Enable and run it from the sidebar after you run the main simulation (or independently).")
        # still show module UI below (it has its own run button)
    else:
        # ---------------- Main simulation ---------------- #
        with st.spinner("Simulating main drawings..."):
            white_counts, power_counts = simulate_powerball_draws(n_draws, seed)

        st.success(f"Main simulation completed: {n_draws:,} draws")

        # Charts
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.subheader("White Balls Distribution (1–69)")
            white_df_chart = pd.DataFrame({"Number": np.arange(1, 70), "Count": white_counts})
            chart_white = (
                alt.Chart(white_df_chart)
                .mark_bar()
                .encode(
                    x=alt.X("Number:O", sort="ascending"),
                    y=alt.Y("Count:Q"),
                    tooltip=["Number", "Count"],
                )
                .properties(height=320)
            )
            st.altair_chart(chart_white, use_container_width=True)

        with col2:
            st.subheader("Powerball Distribution (1–26)")
            power_df_chart = pd.DataFrame({"Powerball": np.arange(1, 27), "Count": power_counts})
            chart_power = (
                alt.Chart(power_df_chart)
                .mark_bar()
                .encode(
                    x=alt.X("Powerball:O", sort="ascending"),
                    y=alt.Y("Count:Q"),
                    tooltip=["Powerball", "Count"],
                )
                .properties(height=320)
            )
            st.altair_chart(chart_power, use_container_width=True)

        # Probability Tables
        st.markdown("---")
        st.subheader("📊 Probability Distribution Tables (from main simulation)")

        white_prob_df, power_prob_df = build_probability_tables(
            white_counts=white_counts,
            power_counts=power_counts,
            n_draws=n_draws,
            include_theoretical=include_theoretical
        )

        ascending = sort_mode.startswith("Cold")
        white_prob_df = white_prob_df.sort_values("Count", ascending=ascending).reset_index(drop=True)
        power_prob_df = power_prob_df.sort_values("Count", ascending=ascending).reset_index(drop=True)

        if top_n_table != "All":
            white_show = white_prob_df.head(int(top_n_table)).copy()
            power_show = power_prob_df.head(int(top_n_table)).copy()
        else:
            white_show = white_prob_df.copy()
            power_show = power_prob_df.copy()

        white_percent_cols = [
            "Empirical per-draw inclusion %",
            "Empirical share of white selections %",
            "Theoretical per-draw inclusion %",
            "Theoretical share of selections %",
        ]
        power_percent_cols = [
            "Empirical per-draw probability %",
            "Theoretical per-draw probability %",
        ]

        white_show = format_percent_cols(white_show, white_percent_cols, decimals=3)
        power_show = format_percent_cols(power_show, power_percent_cols, decimals=3)

        tcol1, tcol2 = st.columns(2, gap="large")
        with tcol1:
            st.markdown("**White balls**")
            st.dataframe(white_show, use_container_width=True, hide_index=True)
        with tcol2:
            st.markdown("**Powerball**")
            st.dataframe(power_show, use_container_width=True, hide_index=True)

        # Ticket generation
        st.markdown("---")
        if mode.startswith("Mode A"):
            st.subheader("🎟 Tickets – Mode A: Pure Random")
            tickets = generate_random_tickets(n_tickets)
        else:
            st.subheader("🎟 Tickets – Mode B: Simulation-based High-Frequency")
            tickets = generate_high_prob_tickets(
                white_counts=white_counts,
                power_counts=power_counts,
                n_tickets=n_tickets,
                top_white_pool=20,
                top_power_pool=10,
            )

        tickets_df = pd.DataFrame([{
            "Ticket #": i + 1,
            "White Balls": " - ".join(f"{n:02d}" for n in t[0]),
            "Powerball": f"{t[1]:02d}",
        } for i, t in enumerate(tickets)])

        st.table(tickets_df)

        if mode.startswith("Mode B"):
            st.info("Mode B is based on the simulation’s frequency patterns and does not improve real-world lottery odds.")

    # ---------------- Sampling Distribution Module ---------------- #
    st.markdown("---")
    st.subheader("🧪 Sampling Distribution Module")

    st.sidebar.header("Sampling Distribution Module")
    enable_sampling = st.sidebar.checkbox("Enable sampling module", value=True)

    if not enable_sampling:
        st.caption("Enable the sampling module from the sidebar to run repeated simulations, CIs, animation, and convergence plots.")
        return

    # Controls
    reps = st.sidebar.slider("🔁 Repetitions (N runs)", min_value=10, max_value=200, value=50, step=10)
    draws_per_rep = st.sidebar.slider("Draws per repetition", min_value=500, max_value=50_000, value=5_000, step=500)

    selected_white = st.sidebar.number_input("Track white ball # (1–69)", min_value=1, max_value=69, value=7, step=1)
    selected_power = st.sidebar.number_input("Track powerball # (1–26)", min_value=1, max_value=26, value=13, step=1)

    animate_top10 = st.sidebar.checkbox("🔀 Animate Top 10 changes", value=True)
    anim_delay = st.sidebar.slider("Animation delay (seconds)", 0.0, 0.5, 0.05, 0.01)

    conv_total_draws = st.sidebar.slider("📊 Convergence plot draws", min_value=2_000, max_value=200_000, value=30_000, step=1_000)
    conv_points = st.sidebar.slider("Convergence checkpoints", min_value=20, max_value=120, value=60, step=10)

    run_sampling = st.sidebar.button("Run Sampling Module")

    st.caption(
        "This module repeats the simulation **N times** and studies the **sampling variability** "
        "of empirical probabilities. Great for statistics education and simulator validation."
    )

    if not run_sampling:
        st.info("Click **Run Sampling Module** in the sidebar.")
        return

    # 🔁 Repeat simulation N times
    with st.spinner("Running repeated simulations..."):
        results = run_sampling_distribution(
            reps=reps,
            draws_per_rep=draws_per_rep,
            selected_white=int(selected_white),
            selected_power=int(selected_power),
            seed=seed
        )

    white_probs = results["white_probs"]
    power_probs = results["power_probs"]
    top10_whites_per_rep = results["top10_whites_per_rep"]
    top10_powers_per_rep = results["top10_powers_per_rep"]
    seed_used = results["seed_used"]

    st.success(f"Sampling module finished (seed used: {seed_used})")

    # 📉 Confidence intervals around empirical probabilities (Wilson)
    st.markdown("### 📉 Confidence intervals (Wilson 95%) for one repetition (binomial model)")
    st.caption(
        "For the tracked white ball, success = “ball appears in the draw among the 5 whites”. "
        "Trials = number of draws in that repetition."
    )

    # Use the last repetition’s probability to show a representative CI
    last_white_k = int(round(float(white_probs[-1]) * draws_per_rep))
    last_power_k = int(round(float(power_probs[-1]) * draws_per_rep))

    w_lo, w_hi = wilson_ci(last_white_k, draws_per_rep, z=1.96)
    p_lo, p_hi = wilson_ci(last_power_k, draws_per_rep, z=1.96)

    ci_df = pd.DataFrame([
        {
            "Tracked": f"White ball {int(selected_white)}",
            "Empirical p̂ (last run)": f"{white_probs[-1]:.5f}",
            "95% CI": f"[{w_lo:.5f}, {w_hi:.5f}]",
            "Theoretical p": f"{THEORETICAL_WHITE_INCLUSION:.5f}",
        },
        {
            "Tracked": f"Powerball {int(selected_power)}",
            "Empirical p̂ (last run)": f"{power_probs[-1]:.5f}",
            "95% CI": f"[{p_lo:.5f}, {p_hi:.5f}]",
            "Theoretical p": f"{THEORETICAL_POWERBALL:.5f}",
        }
    ])
    st.table(ci_df)

    # 🔁 Sampling distribution plots (histograms)
    st.markdown("### 🔁 Sampling distributions of empirical probabilities (across repetitions)")

    dist_col1, dist_col2 = st.columns(2, gap="large")

    with dist_col1:
        w_df = pd.DataFrame({"p_hat": white_probs})
        w_hist = (
            alt.Chart(w_df)
            .mark_bar()
            .encode(
                x=alt.X("p_hat:Q", bin=alt.Bin(maxbins=30), title="Empirical p̂ (white inclusion)"),
                y=alt.Y("count():Q", title="Repetitions"),
                tooltip=["count():Q"],
            )
            .properties(height=260)
        )
        st.altair_chart(w_hist, use_container_width=True)
        st.caption(f"Theoretical ≈ {THEORETICAL_WHITE_INCLUSION:.5f}")

    with dist_col2:
        p_df = pd.DataFrame({"p_hat": power_probs})
        p_hist = (
            alt.Chart(p_df)
            .mark_bar()
            .encode(
                x=alt.X("p_hat:Q", bin=alt.Bin(maxbins=30), title="Empirical p̂ (Powerball)"),
                y=alt.Y("count():Q", title="Repetitions"),
                tooltip=["count():Q"],
            )
            .properties(height=260)
        )
        st.altair_chart(p_hist, use_container_width=True)
        st.caption(f"Theoretical ≈ {THEORETICAL_POWERBALL:.5f}")

    # Summary stats
    st.markdown("### Summary stats")
    stats_df = pd.DataFrame([
        {"Metric": "White inclusion p̂ (mean)", "Value": float(np.mean(white_probs))},
        {"Metric": "White inclusion p̂ (std)", "Value": float(np.std(white_probs, ddof=1))},
        {"Metric": "Powerball p̂ (mean)", "Value": float(np.mean(power_probs))},
        {"Metric": "Powerball p̂ (std)", "Value": float(np.std(power_probs, ddof=1))},
    ])
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # 🔀 Animation showing how “top 10 numbers” change run-to-run
    st.markdown("### 🔀 Top 10 numbers changing run-to-run")

    if animate_top10:
        st.caption("Animating the Top 10 lists across repetitions...")
        placeholder = st.empty()

        for i in range(reps):
            frame_left = pd.DataFrame({
                "Rank": list(range(1, 11)),
                "Top 10 Whites": top10_whites_per_rep[i]
            })
            frame_right = pd.DataFrame({
                "Rank": list(range(1, 11)),
                "Top 10 Powerballs": top10_powers_per_rep[i]
            })

            with placeholder.container():
                a, b = st.columns(2, gap="large")
                with a:
                    st.markdown(f"**Repetition {i+1}/{reps} — White balls Top 10**")
                    st.table(frame_left)
                with b:
                    st.markdown(f"**Repetition {i+1}/{reps} — Powerballs Top 10**")
                    st.table(frame_right)

            time.sleep(anim_delay)
    else:
        st.caption("Animation disabled. Showing final repetition Top 10.")
        a, b = st.columns(2, gap="large")
        with a:
            st.table(pd.DataFrame({"Rank": range(1, 11), "Top 10 Whites": top10_whites_per_rep[-1]}))
        with b:
            st.table(pd.DataFrame({"Rank": range(1, 11), "Top 10 Powerballs": top10_powers_per_rep[-1]}))

    # 📊 Empirical vs theoretical convergence plots
    st.markdown("### 📊 Empirical vs theoretical convergence")

    with st.spinner("Building convergence series..."):
        conv_df = convergence_series(
            total_draws=conv_total_draws,
            selected_white=int(selected_white),
            selected_power=int(selected_power),
            seed=seed,
            points=int(conv_points)
        )

    # White convergence chart
    white_long = conv_df.melt(
        id_vars=["Draws"],
        value_vars=["Empirical (White inclusion)", "Theoretical (White inclusion)"],
        var_name="Series",
        value_name="Probability"
    )
    chart_conv_white = (
        alt.Chart(white_long)
        .mark_line()
        .encode(
            x=alt.X("Draws:Q", title="Cumulative draws"),
            y=alt.Y("Probability:Q", title="Probability"),
            tooltip=["Draws", "Series", "Probability"],
            color="Series:N",
        )
        .properties(height=260, title=f"White ball {int(selected_white)}: empirical vs theoretical")
    )
    st.altair_chart(chart_conv_white, use_container_width=True)

    # Powerball convergence chart
    power_long = conv_df.melt(
        id_vars=["Draws"],
        value_vars=["Empirical (Powerball)", "Theoretical (Powerball)"],
        var_name="Series",
        value_name="Probability"
    )
    chart_conv_power = (
        alt.Chart(power_long)
        .mark_line()
        .encode(
            x=alt.X("Draws:Q", title="Cumulative draws"),
            y=alt.Y("Probability:Q", title="Probability"),
            tooltip=["Draws", "Series", "Probability"],
            color="Series:N",
        )
        .properties(height=260, title=f"Powerball {int(selected_power)}: empirical vs theoretical")
    )
    st.altair_chart(chart_conv_power, use_container_width=True)

    st.info(
        "Interpretation: if the system is fair, the empirical curves wander early (randomness), "
        "then gradually converge toward the theoretical lines as draws increase."
    )


if __name__ == "__main__":
    main()
