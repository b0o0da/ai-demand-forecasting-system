import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROC, REP = ROOT / "data" / "processed", ROOT / "reports"
BLUE, ORANGE, GREY, DARK = "#2F5D8C", "#D9822B", "#9AA5B1", "#1B2430"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
Z = {"90%": 1.2816, "95%": 1.645, "98%": 2.054}

st.set_page_config(page_title="Demand Forecasting Dashboard", page_icon=":bar_chart:", layout="wide")


@st.cache_data
def load():
    hist = pd.read_parquet(PROC / "daily_sales.parquet")
    preds = pd.read_parquet(PROC / "predictions.parquet")
    fut = pd.read_parquet(PROC / "future_forecast.parquet")
    metrics = json.loads((REP / "metrics.json").read_text())
    ets = json.loads((REP / "ets_comparison.json").read_text())
    insights = json.loads((REP / "insights_summary.json").read_text())
    item_ins = pd.read_csv(REP / "item_insights.csv")
    weekly = pd.read_csv(REP / "weekly_seasonal_index.csv")
    return hist, preds, fut, metrics, ets, insights, item_ins, weekly


hist, preds, fut, metrics, ets, insights, item_ins, weekly = load()


def scope(frame, store, item):
    if store != "All stores":
        frame = frame[frame["store"] == int(store.split()[1])]
    if item != "All items":
        frame = frame[frame["item"] == int(item.split()[1])]
    return frame


def style(fig, title, xlab, ylab, height=380):
    fig.update_layout(title=title, xaxis_title=xlab, yaxis_title=ylab, height=height,
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2),
                      template="plotly_white")
    return fig


st.sidebar.header("Scope")
store_sel = st.sidebar.selectbox("Store", ["All stores"] + [f"Store {i}" for i in sorted(hist["store"].unique())])
item_sel = st.sidebar.selectbox("Item", ["All items"] + [f"Item {i}" for i in sorted(hist["item"].unique())])
st.sidebar.caption("Colour code: blue = actual demand, orange = forecast, grey = baseline.")

h_s, p_s, f_s = scope(hist, store_sel, item_sel), scope(preds, store_sel, item_sel), scope(fut, store_sel, item_sel)
daily = h_s.groupby("date")["sales"].sum()
fdaily = f_s.groupby("date")[["p10", "p50", "p90"]].sum()
f0 = fdaily.index.min()

st.title("Demand Forecasting Dashboard")
st.caption(f"{store_sel} | {item_sel} | history {daily.index.min():%d %b %Y} to {daily.index.max():%d %b %Y} | "
           f"forecast {f0:%d %b %Y} to {fdaily.index.max():%d %b %Y} (28 days)")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Demand trends", "Product analysis", "Forecast performance", "Business insights"])

with tab1:
    ly = daily[f0 - pd.Timedelta(days=364): fdaily.index.max() - pd.Timedelta(days=364)].sum()
    fc = fdaily["p50"].sum()
    growth = (fc / ly - 1) * 100
    c = st.columns(4)
    c[0].metric("Total historical sales", f"{daily.sum():,.0f} units")
    c[1].metric("Average demand", f"{daily.mean():,.1f} units per day")
    c[2].metric("Forecast, next 28 days", f"{fc:,.0f} units", f"{growth:+.1f}% vs same weeks last year")
    c[3].metric("Likely range (80%)", f"{f_s['p10'].sum():,.0f} to {f_s['p90'].sum():,.0f}")
    recent = daily[-180:]
    fig = go.Figure()
    fig.add_scatter(x=recent.index, y=recent.values, name="Actual demand", line=dict(color=BLUE))
    fig.add_scatter(x=fdaily.index, y=fdaily["p50"], name="Forecast", line=dict(color=ORANGE, width=3))
    fig.add_vline(x=f0, line_dash="dash", line_color=DARK)
    fig.add_annotation(x=f0, y=1, yref="paper", text="Forecast origin", showarrow=False, xanchor="left")
    style(fig, "Last 6 months of demand and the 28-day forecast", "Date", "Units sold per day")
    st.plotly_chart(fig, width="stretch")
    st.info("Demand is growing year on year and the next four weeks sit in the seasonal low of January. "
            "Expect a drop against December, but a rise against January of last year.")

with tab2:
    win = st.radio("Rolling average window", [7, 30, 60, 90], index=1, horizontal=True)
    fig = go.Figure()
    fig.add_scatter(x=daily.index, y=daily.values, name="Daily demand", line=dict(color=GREY, width=1))
    fig.add_scatter(x=daily.index, y=daily.rolling(win).mean(), name=f"{win}-day rolling mean", line=dict(color=BLUE, width=2.5))
    fig.add_scatter(x=fdaily.index, y=fdaily["p50"], name="Forecast", line=dict(color=ORANGE, width=3))
    style(fig, "Historical and forecast demand on one timeline", "Date", "Units sold per day", 420)
    st.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    dow = daily.groupby(daily.index.dayofweek).mean()
    mon = daily.groupby(daily.index.month).mean()
    f1 = go.Figure(go.Bar(x=DAYS, y=dow.values / dow.mean() * 100, marker_color=BLUE))
    style(f1, "Weekday pattern (100 = average day)", "Weekday", "Demand index", 320)
    f2 = go.Figure(go.Bar(x=[MONTHS[m - 1] for m in mon.index], y=mon.values / mon.mean() * 100, marker_color=BLUE))
    style(f2, "Monthly pattern (100 = average month)", "Month", "Demand index", 320)
    a.plotly_chart(f1, width="stretch")
    b.plotly_chart(f2, width="stretch")
    st.info("Sundays and Saturdays are the busiest days; July is the annual peak and January the low point.")

with tab3:
    st.subheader("Ranking of items by forecast demand")
    rank = scope(fut, store_sel, "All items").groupby("item")["p50"].sum().sort_values(ascending=False)
    top = rank.head(10)
    colors = [ORANGE if f"Item {i}" == item_sel else BLUE for i in top.index]
    fig = go.Figure(go.Bar(x=top.values, y=[f"Item {i}" for i in top.index], orientation="h", marker_color=colors))
    fig.update_yaxes(autorange="reversed")
    style(fig, "Top 10 items by forecast units, next 28 days", "Forecast units", "Item", 380)
    st.plotly_chart(fig, width="stretch")
    st.subheader("Forecast for the selected scope")
    hist_part = daily[-90:]
    test_part = p_s[p_s["split"] == "test"].groupby("date")[["lgbm", "p10", "p90"]].sum().iloc[-60:]
    fig = go.Figure()
    fig.add_scatter(x=hist_part.index, y=hist_part.values, name="Actual demand", line=dict(color=BLUE))
    fig.add_scatter(x=test_part.index, y=test_part["lgbm"], name="Past forecast", line=dict(color=ORANGE, width=1, dash="dot"))
    fig.add_scatter(x=fdaily.index, y=fdaily["p90"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=fdaily.index, y=fdaily["p10"], fill="tonexty", fillcolor="rgba(217,130,43,0.25)",
                    line=dict(width=0), name="80% range")
    fig.add_scatter(x=fdaily.index, y=fdaily["p50"], name="Forecast", line=dict(color=ORANGE, width=3))
    fig.add_vline(x=f0, line_dash="dash", line_color=DARK)
    style(fig, f"{store_sel}, {item_sel}: last 90 days and 28-day forecast", "Date", "Units sold per day", 420)
    st.plotly_chart(fig, width="stretch")
    if item_sel == "All items" or store_sel == "All stores":
        st.caption("Ranges for combined scopes add up item-level bounds, so they are wider than the true range.")

with tab4:
    split = st.radio("Evaluation period", ["test", "validation"], horizontal=True,
                     format_func=lambda s: "Test (Apr to Dec 2017)" if s == "test" else "Validation (Jul 2016 to Mar 2017)")
    names = ["Seasonal naive (last same weekday)", "Moving average (28 days)", "Exponential smoothing (Holt-Winters)",
             "Random Forest", "LightGBM"]
    rows = []
    for n in names:
        m = metrics[split].get(n)
        if m:
            rows.append({"Model": n + (" (baseline)" if n.startswith("Seasonal") else ""), "MAE (units)": m["MAE"],
                         "RMSE (units)": m["RMSE"], "sMAPE (%)": m["sMAPE"], "WAPE (%)": m["WAPE"]})
    table = pd.DataFrame(rows)
    base = table.iloc[0]["MAE (units)"]
    table["Error vs baseline"] = (table["MAE (units)"] / base - 1) * 100
    st.dataframe(table.style.format({"MAE (units)": "{:.2f}", "RMSE (units)": "{:.2f}", "sMAPE (%)": "{:.1f}",
                                     "WAPE (%)": "{:.1f}", "Error vs baseline": "{:+.0f}%"}), hide_index=True, width="stretch")
    best = metrics[split]["LightGBM"]["MAE"]
    st.success(f"LightGBM cuts the average error from {base:.1f} to {best:.1f} units per store-item per day ({(1 - best / base) * 100:.0f}% lower than the baseline).")
    colors = [GREY if i == 0 else (ORANGE if r["Model"] == "LightGBM" else BLUE) for i, r in table.iterrows()]
    fig = go.Figure(go.Bar(x=table["Model"], y=table["MAE (units)"], marker_color=colors))
    style(fig, "Average error by model (lower is better)", "Model", "MAE, units per store-item per day", 340)
    st.plotly_chart(fig, width="stretch")
    comp = p_s[p_s["split"] == split].groupby("date")[["actual", "lgbm", "naive_origin"]].sum()
    fig = go.Figure()
    fig.add_scatter(x=comp.index, y=comp["actual"], name="Actual demand", line=dict(color=BLUE))
    fig.add_scatter(x=comp.index, y=comp["lgbm"], name="LightGBM forecast", line=dict(color=ORANGE))
    fig.add_scatter(x=comp.index, y=comp["naive_origin"], name="Baseline", line=dict(color=GREY, dash="dot"))
    style(fig, "Actual versus forecast for the selected scope", "Date", "Units sold per day", 380)
    st.plotly_chart(fig, width="stretch")
    iv = metrics[split]["_interval_80"]
    st.caption(f"The 80% range contained the actual value {iv['coverage'] * 100:.0f}% of the time. "
               "Holt-Winters was only run on a sample of 60 store-item series, so it is compared like for like below.")
    sample = pd.DataFrame(ets["like_for_like"][split]).T[["MAE", "RMSE", "sMAPE", "WAPE"]]
    st.dataframe(sample.style.format("{:.2f}"), width="stretch")

with tab5:
    g = st.columns(3)
    g[0].metric("Forecast vs same weeks last year", f"{insights['yoy_change_pct']:+.1f}%")
    g[1].metric("Forecast vs last 28 days", f"{insights['vs_last_28_days_pct']:+.1f}%")
    g[2].metric("Items forecast to decline year on year", f"{insights['items_with_negative_yoy']} of 50")
    st.warning("The forecast growth is higher than the 3.9% growth seen in 2017. The model over-forecast "
               "January to March 2017 by about 2% during validation, so treat the upper part of the growth range with care.")
    a, b = st.columns(2)
    ins = item_ins.copy()
    ins["Item"] = "Item " + ins["item"].astype(str)
    grow = ins.sort_values("yoy_change_pct", ascending=False).head(8)
    fig = go.Figure(go.Bar(x=grow["Item"], y=grow["yoy_change_pct"], marker_color=BLUE))
    a.plotly_chart(style(fig, "Fastest growing items (forecast vs last year)", "Item", "Change (%)", 330), width="stretch")
    vol = ins.sort_values("relative_interval_width", ascending=False).head(8)
    fig = go.Figure(go.Bar(x=vol["Item"], y=vol["relative_interval_width"] * 100, marker_color=ORANGE))
    b.plotly_chart(style(fig, "Most uncertain items (need larger safety stock)", "Item", "Range width, % of forecast", 330), width="stretch")
    wk = weekly.rename(columns={weekly.columns[0]: "week", "demand_index": "index"})
    fig = go.Figure(go.Bar(x=wk["week"], y=wk["index"], marker_color=[ORANGE if v > 120 else BLUE for v in wk["index"]]))
    style(fig, "Seasonal demand by week of year (100 = average week); orange = peak weeks", "Week of year", "Demand index", 330)
    st.plotly_chart(fig, width="stretch")
    st.markdown("**Reorder planner** (illustrative: the dataset has no stock data, so you enter current stock)")
    c = st.columns(5)
    p_store = c[0].selectbox("Store", sorted(hist["store"].unique()), key="ps")
    p_item = c[1].selectbox("Item", sorted(hist["item"].unique()), key="pi")
    lead = c[2].slider("Supplier lead time (days)", 1, 28, 7)
    svc = c[3].selectbox("Service level", list(Z), index=1)
    stock = c[4].number_input("Current stock (units)", min_value=0, value=300, step=10)
    one = fut[(fut["store"] == p_store) & (fut["item"] == p_item)].sort_values("date")
    ltd = one["p50"].iloc[:lead].sum()
    sigma = ((one["p90"] - one["p10"]) / (2 * 1.2816)).mean()
    ss = Z[svc] * sigma * np.sqrt(lead)
    rop = ltd + ss
    cover = np.searchsorted(one["p50"].cumsum().values, stock) 
    r = st.columns(4)
    r[0].metric("Demand during lead time", f"{ltd:,.0f} units")
    r[1].metric("Safety stock", f"{ss:,.0f} units")
    r[2].metric("Reorder point", f"{rop:,.0f} units")
    r[3].metric("Stock cover", f"{cover} days" if cover < len(one) else "more than 28 days")
    if stock <= rop:
        st.error(f"Order now: stock is at or below the reorder point of {rop:,.0f} units.")
    else:
        st.success(f"Stock is above the reorder point. Re-check in {max(1, cover - lead)} days.")
