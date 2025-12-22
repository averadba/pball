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
        # 5 unique white balls from 1–69
        white_draw = random.sample(WHITE_BALLS, 5)
        for w in white_draw:
            white_counts[w - 1] += 1

        # 1 Powerball from 1–26
        pb = random.choice(POWER_BALLS)
        power_counts[pb - 1] += 1

    return white_counts, power_counts


def generate_random_tickets(
    n_tickets: int
) -> List[Tuple[List[int], int]]:
    """
    Generate tickets using pure random sampling
    (true to the real Powerball mechanics).
    """
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
    """
    Generate tickets using only the most frequently drawn numbers from the simulation.

    Args:
        white_counts: array of length 69 with frequencies per white ball.
        power_counts: array of length 26 with frequencies per powerball.
        n_tickets: number of tickets to generate (1–5).
        top_white_pool: how many top white numbers to use as the "high probability" pool.
        top_power_pool: same for Powerball numbers.

    Returns:
        List of tickets, each as (white_numbers_list, powerball_number).
    """
    # Sort indices by frequency descending
    white_indices_sorted = sorted(
        range(len(white_counts)),
        key=lambda i: white_counts[i],
        reverse=True
    )
    power_indices_sorted = sorted(
        range(len(power_counts)),
        key=lambda i: power_counts[i],
        reverse=True
    )

    # Restrict to "high probability" pools
    top_white_indices = white_indices_sorted[:top_white_pool]
    top_power_indices = power_indices_sorted[:top_power_pool]

    top_white_numbers = [i + 1 for i in top_white_indices]
    top_power_numbers = [i + 1 for i in top_power_indices]

    # Use frequencies as weights inside the top pool
    white_weights = [white_counts[i] for i in top_white_indices]
    power_weights = [power_counts[i] for i in top_power_indices]

    tickets: List[Tuple[List[int], int]] = []

    for _ in range(n_tickets):
        # 5 unique white balls from the high-probability pool (weighted by frequency)
        selected_white = set()
        while len(selected_white) < 5:
            chosen = random.choices(top_white_numbers, weights=white_weights, k=1)[0]
            selected_white.add(chosen)

        # 1 Powerball from its high-probability pool
        chosen_power = random.choices(top_power_numbers, weights=power_weights, k=1)[0]

        tickets.append((sorted(selected_white), chosen_power))

    return tickets


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
        This app:
        - Simulates many Powerball drawings.  
        - Shows the **distribution** of drawn numbers.  
        - Generates up to **five tickets** in two modes:
          - **Mode A:** Pure random tickets (true to real Powerball odds).  
          - **Mode B:** Tickets from **high-frequency numbers** in the simulation.
        
        > In the **real** game, every valid combination is equally likely.  
        > Mode B is for **fun/experimentation**, not a real-world advantage.
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

    simulate_button = st.sidebar.button("Run Simulation & Generate Tickets")

    if simulate_button:
        seed: Optional[int] = None
        if seed_input.strip():
            try:
                seed = int(seed_input.strip())
            except ValueError:
                st.warning("Seed must be an integer. Ignoring seed value.")
                seed = None

        with st.spinner("Simulating drawings..."):
            white_counts, power_counts = simulate_powerball_draws(n_draws, seed)

        st.success(f"Simulation completed with {n_draws:,} drawings.")

        # Distributions
        col1, col2 = st.columns(2, gap="large")

        # White balls distribution
        with col1:
            st.subheader("White Balls Distribution (1–69)")

            white_df = pd.DataFrame({
                "Number": np.arange(1, 70),
                "Count": white_counts
            })

            chart_white = (
                alt.Chart(white_df)
                .mark_bar()
                .encode(
                    x=alt.X("Number:O", sort="ascending", title="White Ball"),
                    y=alt.Y("Count:Q", title="Frequency in Simulation"),
                    tooltip=["Number", "Count"],
                )
                .properties(height=350)
            )
            st.altair_chart(chart_white, use_container_width=True)

        # Powerball distribution
        with col2:
            st.subheader("Powerball Distribution (1–26)")

            power_df = pd.DataFrame({
                "Powerball": np.arange(1, 27),
                "Count": power_counts
            })

            chart_power = (
                alt.Chart(power_df)
                .mark_bar()
                .encode(
                    x=alt.X("Powerball:O", sort="ascending", title="Powerball"),
                    y=alt.Y("Count:Q", title="Frequency in Simulation"),
                    tooltip=["Powerball", "Count"],
                )
                .properties(height=350)
            )
            st.altair_chart(chart_power, use_container_width=True)

        # Ticket generation
        st.markdown("---")

        if mode.startswith("Mode A"):
            st.subheader("🎟 Generated Tickets – Mode A: Pure Random")
            st.caption(
                "Tickets are generated with uniform randomness across all valid numbers, "
                "matching the real Powerball draw mechanics."
            )

            tickets = generate_random_tickets(n_tickets)

            # Tickets table
            tickets_data = []
            for idx, (white_nums, pb) in enumerate(tickets, start=1):
                tickets_data.append({
                    "Ticket #": idx,
                    "White Balls": " - ".join(f"{n:02d}" for n in white_nums),
                    "Powerball": f"{pb:02d}",
                })

            tickets_df = pd.DataFrame(tickets_data)
            st.table(tickets_df)

            st.info(
                "Mode A reflects the real lottery: every valid combination has the same probability. "
                "Simulated distributions are for visualization only."
            )

        else:
            st.subheader("🎟 Generated Tickets – Mode B: High-Probability (Simulation-based)")
            st.caption(
                "Tickets are generated only from the most frequent numbers in this simulation "
                "(top 20 white balls, top 10 Powerballs). This is for experimentation/fun; "
                "it does **not** improve real-world odds."
            )

            tickets = generate_high_prob_tickets(
                white_counts=white_counts,
                power_counts=power_counts,
                n_tickets=n_tickets,
                top_white_pool=20,   # Only the 20 most frequent white balls
                top_power_pool=10,   # Only the 10 most frequent Powerballs
            )

            # Show which numbers are considered "high probability" in this run
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

            # Tickets table
            tickets_data = []
            for idx, (white_nums, pb) in enumerate(tickets, start=1):
                tickets_data.append({
                    "Ticket #": idx,
                    "White Balls": " - ".join(f"{n:02d}" for n in white_nums),
                    "Powerball": f"{pb:02d}",
                })

            tickets_df = pd.DataFrame(tickets_data)
            st.table(tickets_df)

            st.info(
                "Reminder: Mode B biases tickets toward numbers that appeared most often "
                "in this simulation. Real Powerball odds remain the same for all combinations."
            )

    else:
        st.info(
            "Use the controls in the sidebar, choose **Mode A** or **Mode B**, "
            "and click **Run Simulation & Generate Tickets** to get started."
        )


if __name__ == "__main__":
    main()
