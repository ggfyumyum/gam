import math
import os
import re
import shutil
import sys
from typing import Any

import pandas as pd
try:
    import streamlit as st
except ModuleNotFoundError:
    print("Missing dependency: streamlit. Install requirements and run with: streamlit run horse_only.py")
    sys.exit(1)

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is None:
        print("This app must be run with: streamlit run horse_only.py")
        sys.exit(0)
except Exception:
    # If Streamlit internals change, don't block execution.
    pass

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore

try:
    from streamlit_paste_button import paste_image_button
except Exception:  # pragma: no cover
    paste_image_button = None  # type: ignore

st.set_page_config(page_title="Multiway Vig + Bet-Back EV", layout="centered")


def _fraction_to_decimal_odds(token: str) -> float | None:
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", token)
    if not m:
        return None
    num = float(m.group(1))
    den = float(m.group(2))
    if den <= 0:
        return None
    return 1.0 + (num / den)


def _parse_decimal_odds_token(token: str) -> float | None:
    if token is None:
        return None
    s = str(token).strip()
    if not s:
        return None

    # common OCR normalizations
    s = s.replace(",", ".")
    s = s.replace("$", "")
    s_up = s.upper()

    # Common sportsbook strings
    if s_up in {"EVEN", "EVS", "EV"}:
        return 2.0

    # American odds: +150 / -110
    m_amer = re.fullmatch(r"\s*([+-])\s*(\d{2,5})\s*", s)
    if m_amer:
        sign = m_amer.group(1)
        amt = float(m_amer.group(2))
        if amt <= 0:
            return None
        if sign == "+":
            return 1.0 + (amt / 100.0)
        # negative
        return 1.0 + (100.0 / amt)

    frac = _fraction_to_decimal_odds(s)
    if frac is not None:
        return frac

    # keep only a conservative set of characters
    s = re.sub(r"[^0-9.+-]", "", s)
    if s in ("", ".", "+", "-"):
        return None

    try:
        val = float(s)
    except ValueError:
        return None
    return val


def parse_win_place_rows_from_text(text: str, max_rows: int = 10) -> list[dict[str, Any]]:
    """Heuristic parser for OCR text of a Win/Place odds table.

    Expected to find at least 2 odds per runner (win + place). Returns rows:
    {horse, win_odds, place_odds}.
    """
    if not text:
        return []

    # Normalize whitespace but keep line breaks for row detection.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    rows: list[dict[str, Any]] = []
    for line in lines:
        low = line.lower()
        if "win" in low and "place" in low and len(low) < 40:
            continue

        # Split on whitespace but preserve things like 5/2 tokens.
        tokens = re.split(r"\s+", line)
        odds: list[float] = []
        first_odds_idx: int | None = None

        for i, tok in enumerate(tokens):
            val = _parse_decimal_odds_token(tok)
            if val is None:
                continue
            # Filter out runner numbers and nonsense.
            if val < 1.01 or val > 1000:
                continue
            if first_odds_idx is None:
                first_odds_idx = i
            odds.append(val)

        if len(odds) < 2:
            continue

        # Name is everything before the first odds token.
        name_part = " ".join(tokens[: first_odds_idx or 0]).strip()
        # Strip a leading runner number (e.g., "1" or "(1)")
        name_part = re.sub(r"^\(?\d+\)?\s+", "", name_part).strip()
        horse = name_part if name_part else f"Runner {len(rows)+1}"

        # Books vary: some show WIN/PLC, some show multiple columns.
        # Heuristic: among the first few odds-like tokens, WIN is usually the largest, PLACE the smallest.
        candidates = odds[:3] if len(odds) >= 3 else odds
        win_odds = max(candidates)
        place_odds = min(candidates)

        # If there were exactly 2 and they look reversed, swap.
        if len(odds) == 2 and odds[0] < odds[1]:
            win_odds, place_odds = odds[1], odds[0]

        rows.append({"horse": horse, "win_odds": win_odds, "place_odds": place_odds})
        if len(rows) >= max_rows:
            break

    return rows


def ocr_image_to_text(image, tesseract_cmd: str | None = None) -> str:
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed.")
    if Image is None:
        raise RuntimeError("Pillow is not installed.")

    # Best-effort Tesseract detection (Windows-friendly).
    resolved_cmd = (tesseract_cmd or "").strip() or None
    if resolved_cmd is None:
        env_cmd = (os.environ.get("TESSERACT_CMD") or "").strip()
        if env_cmd:
            resolved_cmd = env_cmd
    if resolved_cmd is None:
        which_cmd = shutil.which("tesseract")
        if which_cmd:
            resolved_cmd = which_cmd
    if resolved_cmd is None and os.name == "nt":
        candidates = [
            r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
            r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
        ]
        for c in candidates:
            if os.path.exists(c):
                resolved_cmd = c
                break
    if resolved_cmd:
        pytesseract.pytesseract.tesseract_cmd = resolved_cmd

    # Light preprocessing tends to help with screenshots.
    im = image.convert("L")
    # Simple contrast stretch via point() is cheap and helps some OCR.
    im = im.point(lambda p: 0 if p < 25 else 255 if p > 230 else p)

    try:
        # PSM 6: Assume a uniform block of text.
        return pytesseract.image_to_string(im, config="--psm 6")
    except pytesseract.pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract OCR executable not found. Install Tesseract (Windows) or set the 'Tesseract executable path' "
            "in OCR settings (or set env var TESSERACT_CMD)."
        )


def bet_back_ev_unconditional_place(
    wager,
    win_odds,
    place_odds,
    bonus_val=1.0,
    true_prob_win=None,
    true_prob_place_uncond=None,
):
    """
    Single runner EV with "bet back" where PLACE means top-3 INCLUDING win.

    Outcomes:
      - WIN: payout = win_odds * wager  (stake returned on win)
      - PLACE but NOT WIN: payout = bonus_val * wager  (bet back as bonus cash)
      - LOSE and NOT PLACE: payout = 0

    Probabilities:
      - P(win) = p_win
      - P(place) = p_place (includes win)
      - P(place but not win) = max(p_place - p_win, 0)
      - P(lose all) = max(1 - p_place, 0)

    Returns EV payout plus net vs cash stake.
    """
    if true_prob_win is None or true_prob_place_uncond is None:
        raise ValueError("You must pass true_prob_win and true_prob_place_uncond (from de-vigged markets).")

    p_win = float(true_prob_win)
    p_place = float(true_prob_place_uncond)

    p_place_not_win = max(p_place - p_win, 0.0)
    p_lose_all = max(1.0 - p_place, 0.0)

    payout_win = win_odds * wager
    payout_betback = bonus_val * wager

    ev_payout = p_win * payout_win + p_place_not_win * payout_betback

    return {
        "true_prob_win": p_win,
        "true_prob_place_unconditional": p_place,
        "p_win": p_win,
        "p_place_not_win": p_place_not_win,
        "p_lose_all": p_lose_all,
        "payout_win": payout_win,
        "payout_betback": payout_betback,
        "ev_payout": ev_payout,
        "ev_net_cash": ev_payout - wager,
        "ev_net_if_bonus_stake": ev_payout,
    }


def vig_from_win_place_markets(win_odds_list, place_odds_list, places=3):
    """
    WIN market:
      implied_win_i = 1 / win_odds_i
      sum_implied_win = sum(implied_win_i)
      overround_win = sum_implied_win - 1
      de-vig P(win)_i = implied_win_i / sum_implied_win  (sums to 1)

    PLACE market (Top 'places', e.g. 3):
      implied_place_i = 1 / place_odds_i
      sum_implied_place = sum(implied_place_i)
      overround_place_abs = sum_implied_place - places
      overround_place_rel = sum_implied_place/places - 1
      de-vig P(place)_i = implied_place_i / sum_implied_place * places  (sums to places)

    Returns dict with edges and de-vigged probs.
    """
    # WIN
    implied_win = [1.0 / o for o in win_odds_list]
    sum_iw = sum(implied_win)
    overround_win = sum_iw - 1.0
    p_true_win = [p / sum_iw for p in implied_win] if sum_iw > 0 else [None] * len(implied_win)

    # PLACE
    implied_place = [1.0 / o for o in place_odds_list]
    sum_ip = sum(implied_place)
    overround_place_abs = sum_ip - float(places)
    overround_place_rel = (sum_ip / float(places) - 1.0) if places > 0 else None
    p_true_place = [p / sum_ip * float(places) for p in implied_place] if sum_ip > 0 else [None] * len(implied_place)

    return {
        "sum_implied_win": sum_iw,
        "overround_win": overround_win,
        "sum_implied_place": sum_ip,
        "overround_place_abs": overround_place_abs,
        "overround_place_rel": overround_place_rel,
        "p_true_win": p_true_win,
        "p_true_place": p_true_place,
    }


st.title("Multiway Vig (WIN + PLACE) + Bet-Back EV")

st.caption(
    "Enter **win odds** and **place odds** for up to 10 horses.\n\n"
    "**Assumptions**\n"
    "- Exactly **one** horse wins.\n"
    "- **Place** means **Top-3 (1st/2nd/3rd), includes the winner**.\n"
    "- WIN implied probabilities should sum to 1 + vig.\n"
    "- PLACE implied probabilities should sum to 3 + vig.\n"
    "- We de-vig by normalizing implied probabilities."
)

st.divider()

colA, colB = st.columns(2)
with colA:
    wager = st.number_input("Wager ($)", min_value=0.0, value=50.0, step=1.0)
with colB:
    bonus_val = st.number_input("Bonus value ($ per $1 bonus)", min_value=0.0, value=1.0, step=0.05)

st.divider()
st.subheader("Odds input (up to 10 horses)")

st.subheader("Paste odds screenshot (OCR)")
st.caption(
    "Paste a screenshot of a Win/Place odds table from your clipboard. We'll OCR it and try to auto-fill the inputs below. "
    "If parsing is off, edit the extracted text and re-parse."
)

if "ocr_text" not in st.session_state:
    st.session_state["ocr_text"] = ""
if "ocr_parsed_rows" not in st.session_state:
    st.session_state["ocr_parsed_rows"] = []

if "ocr_image" not in st.session_state:
    st.session_state["ocr_image"] = None

if paste_image_button is None:
    st.error("Clipboard paste requires `streamlit-paste-button`. Install it to enable paste.")
else:
    pasted = paste_image_button("Paste screenshot from clipboard", key="paste_odds_screenshot")
    if pasted and getattr(pasted, "image_data", None) is not None:
        st.session_state["ocr_image"] = pasted.image_data

with st.expander("OCR settings", expanded=False):
    tesseract_cmd = st.text_input(
        "Tesseract executable path (optional)",
        value=st.session_state.get("tesseract_cmd", ""),
        help="On Windows you may need to install Tesseract OCR and either add it to PATH or point this to tesseract.exe",
    )
    st.session_state["tesseract_cmd"] = tesseract_cmd

    detected = shutil.which("tesseract")
    st.caption(
        f"Python: {sys.executable} | pytesseract: {'ok' if pytesseract is not None else 'missing'} | "
        f"tesseract (PATH): {detected or 'not found'}"
    )

img = st.session_state.get("ocr_image")
if img is not None:
    if Image is None:
        st.error("Pillow is required for image OCR. Add `pillow` to requirements and install it.")
    else:
        st.image(img, caption="Pasted screenshot", use_container_width=True)

        c1, c2 = st.columns([1, 1])
        with c1:
            run_ocr = st.button("Run OCR", key="run_ocr")
        with c2:
            parse_only = st.button("Parse text", key="parse_ocr")

        if run_ocr:
            if pytesseract is None:
                st.error(
                    "OCR requires `pytesseract` and the Tesseract binary. "
                    "Install `pytesseract` and install Tesseract OCR on Windows (then set PATH or the executable path)."
                )
            else:
                try:
                    st.session_state["ocr_text"] = ocr_image_to_text(
                        img,
                        tesseract_cmd=st.session_state.get("tesseract_cmd") or None,
                    )
                except Exception as e:
                    st.error(f"OCR failed: {e}")

        st.session_state["ocr_text"] = st.text_area(
            "Extracted / editable text",
            value=st.session_state.get("ocr_text", ""),
            height=200,
            key="ocr_text_area",
        )

        if parse_only or run_ocr:
            st.session_state["ocr_parsed_rows"] = parse_win_place_rows_from_text(
                st.session_state.get("ocr_text", ""),
                max_rows=10,
            )

        parsed_rows = st.session_state.get("ocr_parsed_rows", [])
        if parsed_rows:
            st.write("Parsed rows (preview):")
            st.dataframe(pd.DataFrame(parsed_rows), use_container_width=True)

            if st.button("Fill odds inputs from parsed rows", key="fill_from_ocr"):
                for i, r in enumerate(parsed_rows[:10]):
                    st.session_state[f"name_{i}"] = str(r.get("horse", "") or "")
                    st.session_state[f"win_{i}"] = str(r.get("win_odds", "") or "")
                    st.session_state[f"place_{i}"] = str(r.get("place_odds", "") or "")
                st.rerun()
        else:
            st.info("No rows parsed yet. Try OCR, then Parse text.")

st.divider()

N_MAX = 10
rows = []

for i in range(N_MAX):
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        name = st.text_input(f"Horse {i+1} name", value="", key=f"name_{i}")
    with c2:
        w = st.text_input(f"Win odds", value="", key=f"win_{i}")
    with c3:
        p = st.text_input(f"Place odds", value="", key=f"place_{i}")

    # parse odds
    try:
        win_odds = float(w) if str(w).strip() != "" else None
    except ValueError:
        win_odds = None

    try:
        place_odds = float(p) if str(p).strip() != "" else None
    except ValueError:
        place_odds = None

    if name.strip() == "" and win_odds is None and place_odds is None:
        continue

    rows.append(
        {
            "horse": name.strip() if name.strip() else f"Horse {i+1}",
            "win_odds": win_odds,
            "place_odds": place_odds,
        }
    )

valid = [
    r for r in rows
    if (r["win_odds"] is not None and r["place_odds"] is not None and r["win_odds"] >= 1.0 and r["place_odds"] >= 1.0)
]

if len(rows) == 0:
    st.info("Fill at least one row (horse name optional).")
elif len(valid) == 0:
    st.warning("You have entries, but none have BOTH valid win odds and place odds (>= 1).")
else:
    win_odds_list = [r["win_odds"] for r in valid]
    place_odds_list = [r["place_odds"] for r in valid]

    vig = vig_from_win_place_markets(win_odds_list, place_odds_list, places=3)

    st.subheader("Market vig (overround)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("WIN sum implied", f"{vig['sum_implied_win']:.6f}")
    c2.metric("WIN overround", f"{vig['overround_win']:.6f}")
    c3.metric("PLACE sum implied", f"{vig['sum_implied_place']:.6f}")
    c4.metric("PLACE overround vs 3", f"{vig['overround_place_abs']:.6f}")

    if vig["overround_place_rel"] is not None:
        st.write(f"PLACE overround (relative): **{vig['overround_place_rel']*100:.3f}%**")

    st.subheader("De-vigged probabilities (used for EV)")
    table = []
    for r, pwin, pplace in zip(valid, vig["p_true_win"], vig["p_true_place"]):
        table.append(
            {
                "Horse": r["horse"],
                "Win odds": r["win_odds"],
                "Place odds": r["place_odds"],
                "Implied P(win)": 1.0 / r["win_odds"],
                "Implied P(place)": 1.0 / r["place_odds"],
                "De-vig P(win)": pwin,
                "De-vig P(place top3)": pplace,  # sums to 3 across horses
            }
        )

    df = pd.DataFrame(table)
    st.dataframe(df, use_container_width=True)

    st.divider()
    st.subheader("Bet-Back EV on a selected horse")

    horse_names = [r["horse"] for r in valid]
    pick = st.selectbox("Select horse", horse_names, index=0)
    idx = horse_names.index(pick)

    sel = valid[idx]
    p_win_true = vig["p_true_win"][idx]
    p_place_true = vig["p_true_place"][idx]

    # show quick probability sanity
    if p_place_true + 1e-9 < p_win_true:
        st.warning(
            "De-vigged P(place) < P(win) for this horse (market inconsistency). "
            "Place-but-not-win will be clamped to 0."
        )

    if st.button("Calculate Bet-Back EV", key="calc_bb_from_market"):
        ev = bet_back_ev_unconditional_place(
            wager=wager,
            win_odds=sel["win_odds"],
            place_odds=sel["place_odds"],
            bonus_val=bonus_val,
            true_prob_win=p_win_true,
            true_prob_place_uncond=p_place_true,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("De-vig P(win)", f"{ev['true_prob_win']:.6f}")
        c2.metric("De-vig P(place)", f"{ev['true_prob_place_unconditional']:.6f}")
        c3.metric("EV payout ($)", f"{ev['ev_payout']:.4f}")

        st.dataframe(
            pd.DataFrame(
                {
                    "Outcome": ["Win", "Place but not win", "Lose all"],
                    "Probability": [ev["p_win"], ev["p_place_not_win"], ev["p_lose_all"]],
                    "Payout ($)": [ev["payout_win"], ev["payout_betback"], 0.0],
                }
            ),
            use_container_width=True,
        )

        st.write(f"EV net vs cash stake (EV - wager): **{ev['ev_net_cash']:.4f}**")
        st.write(f"EV if stake is bonus (treat wager as free): **{ev['ev_net_if_bonus_stake']:.4f}**")
