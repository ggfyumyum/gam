import math
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bonus Parlay EV + Breakeven", layout="centered")


def calc_ev(
    wager,
    bonus_val,
    offer1,
    offer2,
    offer3,
    house_edge=0.06,
    true_prob1=None,
    true_prob2=None,
    true_prob3=None,
):
    # infer missing probabilities
    ps = [true_prob1, true_prob2, true_prob3]
    offers = [offer1, offer2, offer3]
    for i, p in enumerate(ps):
        if p is None:
            # Decimal odds implied probability is 1/odds.
            # Use house_edge as a simple discount on implied probability.
            implied = 1.0 / offers[i]
            inferred = implied * (1.0 - house_edge)
            ps[i] = max(0.0, min(1.0, inferred))
    true_prob1, true_prob2, true_prob3 = ps

    win_all = offer1 * offer2 * offer3 * wager
    lose_1 = bonus_val * wager

    p_wa = true_prob1 * true_prob2 * true_prob3

    p_l1 = (
        true_prob1 * (1 - true_prob2) * true_prob3
        + (1 - true_prob1) * true_prob2 * true_prob3
        + true_prob1 * true_prob2 * (1 - true_prob3)
    )

    # Exactly two lose (i.e., exactly one wins)
    p_l2 = (
        true_prob1 * (1 - true_prob2) * (1 - true_prob3)
        + (1 - true_prob1) * true_prob2 * (1 - true_prob3)
        + (1 - true_prob1) * (1 - true_prob2) * true_prob3
    )
    p_la = (1 - true_prob1) * (1 - true_prob2) * (1 - true_prob3)

    ev_payout = p_wa * win_all + p_l1 * lose_1
    ev_net = ev_payout - wager

    out = {
        "true_prob1": true_prob1,
        "true_prob2": true_prob2,
        "true_prob3": true_prob3,
        "p_win_all": p_wa,
        "p_lose_1_get_bonus": p_l1,
        "p_lose_2": p_l2,
        "p_lose_all": p_la,
        "win_all_payout": win_all,
        "lose_1_payout": lose_1,
        "ev_payout": ev_payout,
        "ev_net": ev_net,
    }
    return out


def breakeven_true_prob_bet2_equal_bet3(
    wager,
    bonus_val,
    offer1,
    offer23,
    house_edge=0.06,
    true_prob1=None,
):
    """
    Breakeven true probability p for bet2 and bet3 (assumed equal),
    such that EV payout == wager (same EV structure as calc_ev()).
    bet1 fixed. If true_prob1 is None, inferred as 1/(offer1+house_edge).
    """
    if true_prob1 is not None:
        q = true_prob1
    else:
        implied1 = 1.0 / offer1
        q = max(0.0, min(1.0, implied1 * (1.0 - house_edge)))
    O = offer1 * (offer23 ** 2)
    B = bonus_val

    # Breakeven condition: q p^2 O + [2 q p(1-p) + (1-q)p^2] B = 1
    # => A p^2 + C p - 1 = 0
    A = q * O + (1 - 3 * q) * B
    C = 2 * q * B

    eps = 1e-12
    if abs(A) < eps:
        if abs(C) < eps:
            return None
        p = 1.0 / C
        return p if 0 <= p <= 1 else None

    disc = C * C + 4 * A
    if disc < 0:
        return None

    sqrt_disc = math.sqrt(disc)
    p1 = (-C + sqrt_disc) / (2 * A)
    p2 = (-C - sqrt_disc) / (2 * A)

    candidates = [p for p in (p1, p2) if 0 <= p <= 1]
    if not candidates:
        return None

    def ev_ratio(p):
        p_wa = q * p * p
        p_l1 = 2 * q * p * (1 - p) + (1 - q) * p * p
        return p_wa * O + p_l1 * B  # should equal 1 at breakeven

    return min(candidates, key=lambda p: abs(ev_ratio(p) - 1.0))


st.title("Bonus Parlay EV + Breakeven (bet2 = bet3)")

with st.sidebar:
    st.header("Inputs")

    wager = st.number_input("Wager ($)", min_value=0.0, value=50.0, step=1.0)
    bonus_val = st.number_input(
        "Bonus cash value (as $ per $1 bonus)", min_value=0.0, value=1.0, step=0.05
    )
    house_edge = st.number_input("House edge (used to infer probs)", min_value=0.0, value=0.06, step=0.01)

    st.subheader("Offers (decimal odds)")
    offer1 = st.number_input("Offer 1", min_value=1.0, value=2.0, step=0.01)
    offer2 = st.number_input("Offer 2", min_value=1.0, value=1.01, step=0.01)
    offer3 = st.number_input("Offer 3", min_value=1.0, value=1.01, step=0.01)

    st.subheader("Optional true probs (leave blank to infer)")
    use_p1 = st.checkbox("Override true_prob1?", value=False)
    true_prob1 = st.number_input("true_prob1", min_value=0.0, max_value=1.0, value=0.5, step=0.01) if use_p1 else None

    use_p2 = st.checkbox("Override true_prob2?", value=False)
    true_prob2 = st.number_input("true_prob2", min_value=0.0, max_value=1.0, value=0.5, step=0.01) if use_p2 else None

    use_p3 = st.checkbox("Override true_prob3?", value=False)
    true_prob3 = st.number_input("true_prob3", min_value=0.0, max_value=1.0, value=0.5, step=0.01) if use_p3 else None


st.subheader("EV for your 3-leg bet")

res = calc_ev(
    wager=wager,
    bonus_val=bonus_val,
    offer1=offer1,
    offer2=offer2,
    offer3=offer3,
    house_edge=house_edge,
    true_prob1=true_prob1,
    true_prob2=true_prob2,
    true_prob3=true_prob3,
)

col1, col2, col3 = st.columns(3)
col1.metric("EV payout ($)", f"{res['ev_payout']:.4f}")
col2.metric("EV net ($)", f"{res['ev_net']:.4f}")
col3.metric("EV net (%)", f"{(res['ev_net']/wager*100 if wager else 0):.4f}%")

df_probs = pd.DataFrame(
    {
        "Metric": ["p(win all)", "p(lose exactly 1 & bonus)", "p(lose 2)", "p(lose all)"],
        "Value": [res["p_win_all"], res["p_lose_1_get_bonus"], res["p_lose_2"], res["p_lose_all"]],
    }
)
df_true = pd.DataFrame(
    {
        "Leg": ["1", "2", "3"],
        "Offer": [offer1, offer2, offer3],
        "True prob": [res["true_prob1"], res["true_prob2"], res["true_prob3"]],
    }
)

st.write("### Inferred / used true probabilities")
st.dataframe(df_true, use_container_width=True)

st.write("### Outcome probabilities (per your model)")
st.dataframe(df_probs, use_container_width=True)

st.divider()
st.subheader("Breakeven true probability for bet2 = bet3 (bet1 fixed)")

offer23 = st.number_input("Offer for bet2 and bet3 (same)", min_value=1.0, value=float(offer2), step=0.01)

p_be = breakeven_true_prob_bet2_equal_bet3(
    wager=wager,
    bonus_val=bonus_val,
    offer1=offer1,
    offer23=offer23,
    house_edge=house_edge,
    true_prob1=true_prob1,  # if None, inferred from offer1 + house_edge
)

if p_be is None:
    st.error("No valid breakeven probability found in [0,1] for these parameters.")
else:
    st.success(f"Breakeven true probability for bet2 = bet3 is: **{p_be:.6f}**")

    if st.button("Sanity check: run calc_ev using breakeven p for legs 2 & 3"):
        check = calc_ev(
            wager=wager,
            bonus_val=bonus_val,
            offer1=offer1,
            offer2=offer23,
            offer3=offer23,
            house_edge=house_edge,
            true_prob1=true_prob1,  # fixed / inferred
            true_prob2=p_be,
            true_prob3=p_be,
        )
        st.write("### Sanity check results (using p_be for leg2 & leg3)")
        st.write(f"EV payout: {check['ev_payout']:.6f} (target = wager {wager:.6f})")
        st.write(f"EV net: {check['ev_net']:.6f} (target = 0)")
