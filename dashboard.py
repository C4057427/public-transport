import os, pandas as pd, plotly.express as px

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def build_dashboard(agg, out_html):
    _ensure_dir(os.path.dirname(out_html))
    figs=[]

    if "daily_route" in agg and len(agg["daily_route"]):
        df=agg["daily_route"].copy()
        has_date = "service_date" in df.columns
        has_dir  = "direction" in df.columns

        if has_date:
            df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")

            # OTP Crease - Force SVG rendering, Remove markers
            f1 = px.line(
                df, x="service_date", y="otp", color="route",
                line_dash=("direction" if has_dir else None),
                markers=False, render_mode="svg",
                title="On-Time Performance (Trip-level, origin-first-ping)"
            )
            f1.update_traces(line_simplify=True)
            f1.update_layout(yaxis_tickformat=".0%", yaxis_title="OTP", xaxis_title="Date")
            figs.append(f1)

            # Average Delay Line - Also using SVG
            f2 = px.line(
                df, x="service_date", y="delay_avg", color="route",
                line_dash=("direction" if has_dir else None),
                markers=False, render_mode="svg",
                title="Average Departure Delay (min, at origin)"
            )
            f2.update_traces(line_simplify=True)
            figs.append(f2)

        else:
            f1 = px.bar(
                df, x="route", y="otp", color=("direction" if has_dir else None),
                barmode="group", title="On-Time Performance by Route (no date)"
            )
            f1.update_layout(yaxis_tickformat=".0%", yaxis_title="OTP", xaxis_title="Route")
            figs.append(f1)

            f2 = px.bar(
                df, x="route", y="delay_avg", color=("direction" if has_dir else None),
                barmode="group", title="Average Departure Delay by Route (min)"
            )
            figs.append(f2)

        # Cancellation Rate
        if "cancel_rate" in df.columns:
            if has_date:
                f3 = px.line(
                    df, x="service_date", y="cancel_rate", color="route",
                    line_dash=("direction" if has_dir else None),
                    markers=False, render_mode="svg",
                    title="Cancellation Rate"
                )
                f3.update_traces(line_simplify=True)
                f3.update_layout(yaxis_tickformat=".0%", yaxis_title="Cancel Rate", xaxis_title="Date")
            else:
                f3 = px.bar(
                    df, x="route", y="cancel_rate", color=("direction" if has_dir else None),
                    barmode="group", title="Cancellation Rate by Route (no date)"
                )
                f3.update_layout(yaxis_tickformat=".0%", yaxis_title="Cancel Rate", xaxis_title="Route")
            figs.append(f3)

    if "stop_rank" in agg and len(agg["stop_rank"]):
        sr = agg["stop_rank"].copy().sort_values("delay_med", ascending=False).head(20)
        f4 = px.bar(
            sr, x="delay_med", y="stop_id", color="route", orientation="h",
            title="Top-20 Origins by Median Departure Delay (Worst First)"
        )
        f4.update_layout(xaxis_title="Median Departure Delay (min)", yaxis_title="Origin Stop ID")
        figs.append(f4)

        f5 = px.histogram(agg["stop_rank"], x="otp", nbins=20, title="Origin-level OTP Distribution")
        figs.append(f5)

    html = ["<html><head><meta charset='utf-8'><title>Transit Reliability Dashboard</title></head><body>"]
    for fig in figs:
        html.append(fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True}))
    html.append("</body></html>")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    return out_html
