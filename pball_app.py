import random
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


# Core Powerball logic

WHITE_BALLS = list(range(1, 70))   # 1–69
POWER_BALLS = list(range(1, 27))   # 1–26


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

    white_counts = np.zeros(len(WHITE_BALLS), dtype=int)
    power_counts = np.zeros(len(POWER_BALLS), dtype=int)

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


def build_probability_tables(
    white_counts: np.ndarray,
    power_counts: np.ndarray,
    n_draws: int,
    include_theoretical: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create probability distribution tables based on simulation results.

    Empirical (from simulation):
      White balls:
        - Count: times the ball appeared across all draws (5 whites per draw)
        - Empirical per-draw inclusion: Count / n_draws
        - Empirical share of white selections: Count / (n_draws * 5)

      Powerball:
        - Count: times PB appeared
        - Empirical per-draw probability: Count / n_draws

    Theoretical (true lottery):
      - White per-draw inclusion = 5/69 (chance a specific white ball appears in the 5 whites)
      - White share of selections = 1/69
      - Powerball per-draw = 1/26
    """
    white_df = pd.DataFrame({
        "Number": np.arange(1, 70),
        "Count": white_counts.astype(int),
    })

    white_df["Empirical per-draw inclusion %"] = (white_df["Count"] / float(n_draws)) * 100.0
    white_df["Empirical share of white selections %"] = (white_df["Count"] / float(n_draws * 5)) * 100.0

    if include_theoretical:
        white_df["Theoretical per-draw inclusion %"] = (5.0 / 69.0) * 100.0
        white_df["Theoretical share of selections %"] = (1.0 / 69.0) * 100.0

    # Sort will be applied in the UI based on user selection
    power_df = pd.DataFrame({
        "Powerball": np.arange(1, 27),
        "Count": power_counts.astype(int),
    })
    power_df["Empirical per-draw probability %"] = (power_df["Count"] / float(n_draws)) * 100.0

    if include_theoretical:
        power_df["Theoretical per-draw probability %"] = (1.0 / 26.0) * 100.0

    return white_df, power_df


def format_percent_cols(df: pd.DataFrame, cols: List[str], decimals: int = 3) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{float(x):.{decimals}f}%")
    return out


# Streamlit App

def main():
    st.set_page_config(
        page_title="Powerball Simulator & Ticket Generator",
        page_icon="🎱",
        layout="wide",
    )

    st.title("🎱 Powerball Simulator & Ticket Generator")

    st.markdown(
        """
        Simulate Powerball drawings, visualize distributions, and generate up to **five tickets** in two modes:

        - **Mode A:** Pure random tickets (matches real Powerball mechanics).
        - **Mode B:** “High-probability” tickets **based on this simulation’s most frequent numbers** (for fun/experiments).

        **Reality check:** in the real lottery, every valid combination is equally likely.
        """
    )

    # Sidebar controls
    st.sidebar.header("Settings")

    mode = st.sidebar.radio(
        "Ticket generation mode",
        options=["Mode A – Pure Random", "Mode B – High-Probability (Simulation-based)"],
        index=0,
    )

    n_draws = st.sidebar.slider(
        "Number of simulated drawings",
        min_value=1_000,
        max_value=200_000,
        value=50_000,
        step=1_000,
        help="More draws → smoother distributions, but slower to compute."
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
        help="Use a number for reproducible simulations. Leave blank for random."
    )

    st.sidebar.subheader("Probability tables")

    sort_mode = st.sidebar.radio(
        "Sort numbers by",
        options=["Hot (most frequent)", "Cold (least frequent)"],
        index=0,
        help="Hot = highest empirical counts; Cold = lowest empirical counts."
    )

    include_theoretical = st.sidebar.checkbox(
        "Show theoretical probabilities (true lottery)",
        value=True
    )

    top_n_table = st.sidebar.selectbox(
        "Show how many rows?",
        options=[10, 15, 20, 30, "All"],
        index=1
    )

    simulate_button = st.sidebar.button("Run Simulation & Generate Tickets")

    if not simulate_button:
        st.info("Choose a mode, adjust settings, then click **Run Simulation & Generate Tickets**.")
        return

    # Seed handling
    seed: Optional[int] = None
    if seed_input.strip():
        try:
            seed = int(seed_input.strip())
        except ValueError:
            st.warning("Seed must be an integer. Ignoring seed value.")
            seed = None

    # Run simulation
    with st.spinner("Simulating drawings..."):
        white_counts, power_counts = simulate_powerball_draws(n_draws, seed)

    st.success(f"Simulation completed with {n_draws:,} drawings.")

    # Distributions (charts)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("White Balls Distribution (1–69)")
        white_df_chart = pd.DataFrame({"Number": np.arange(1, 70), "Count": white_counts})

        chart_white = (
            alt.Chart(white_df_chart)
            .mark_bar()
            .encode(
                x=alt.X("Number:O", sort="ascending", title="White Ball"),
                y=alt.Y("Count:Q", title="Frequency in Simulation"),
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
                x=alt.X("Powerball:O", sort="ascending", title="Powerball"),
                y=alt.Y("Count:Q", title="Frequency in Simulation"),
                tooltip=["Powerball", "Count"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart_power, use_container_width=True)

    # Probability Distribution Tables
    st.markdown("---")
    st.subheader("📊 Probability Distribution Tables")

    white_prob_df, power_prob_df = build_probability_tables(
        white_counts=white_counts,
        power_counts=power_counts,
        n_draws=n_draws,
        include_theoretical=include_theoretical
    )

    ascending = (sort_mode.startswith("Cold"))
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
        if include_theoretical:
            st.caption("Theoretical white-ball inclusion per draw is **5/69 ≈ 7.246%**; share of selections is **1/69 ≈ 1.449%**.")

    with tcol2:
        st.markdown("**Powerball**")
        st.dataframe(power_show, use_container_width=True, hide_index=True)
        if include_theoretical:
            st.caption("Theoretical Powerball probability per draw is **1/26 ≈ 3.846%**.")

    st.caption(
        "Empirical probabilities are estimates from the simulation. "
        "Theoretical probabilities reflect the actual lottery rules."
    )

    # Ticket generation
    st.markdown("---")
    if mode.startswith("Mode A"):
        st.subheader("🎟 Generated Tickets – Mode A: Pure Random")
        st.caption("Uniform random tickets (matches real Powerball mechanics).")
        tickets = generate_random_tickets(n_tickets)
    else:
        st.subheader("🎟 Generated Tickets – Mode B: High-Probability (Simulation-based)")
        st.caption(
            "Tickets use only the most frequent numbers in this simulation "
            "(top 20 white balls + top 10 Powerballs), weighted by frequency."
        )
        tickets = generate_high_prob_tickets(
            white_counts=white_counts,
            power_counts=power_counts,
            n_tickets=n_tickets,
            top_white_pool=20,
            top_power_pool=10,
        )

        # Show the pools used
        white_sorted_indices = np.argsort(-white_counts)
        power_sorted_indices = np.argsort(-power_counts)
        top_white_numbers = [int(i + 1) for i in white_sorted_indices[:20]]
        top_power_numbers = [int(i + 1) for i in power_sorted_indices[:10]]

        pool_col1, pool_col2 = st.columns(2)
        with pool_col1:
            st.caption("High-probability white balls (top 20 in this simulation):")
            st.write(", ".join(str(x) for x in top_white_numbers))
        with pool_col2:
            st.caption("High-probability Powerballs (top 10 in this simulation):")
            st.write(", ".join(str(x) for x in top_power_numbers))

    # Display tickets
    tickets_data = []
    for idx, (white_nums, pb) in enumerate(tickets, start=1):
        tickets_data.append({
            "Ticket #": idx,
            "White Balls": " - ".join(f"{n:02d}" for n in white_nums),
            "Powerball": f"{pb:02d}",
        })
    st.table(pd.DataFrame(tickets_data))

    if mode.startswith("Mode B"):
        st.info(
            "Mode B biases toward numbers frequent in *this simulation*; "
            "it does not improve real lottery odds."
        )


if __name__ == "__main__":
    main()
