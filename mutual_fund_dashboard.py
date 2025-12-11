import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests
import io
import csv
import math
import os

st.markdown("""
    <style>
    /* Reduce metric label text size */
    div.css-1xarl3l span {
        font-size: 14px !important;
    }

    /* Reduce metric value text size */
    div.css-1xarl3l div {
        font-size: 18px !important;
    }
    </style>
""", unsafe_allow_html=True)

fund_logos = {
"125354" : "mutualfund/logo/axis.png",
"147946" : "mutualfund/logo/bandhan.png",
"150902" : "mutualfund/logo/edelweiss.jpg",
"151034" : "mutualfund/logo/hsbc.png",
"120594" : "mutualfund/logo/ICICI.jpg",
"119775" : "mutualfund/logo/Kotak.png",
"148928" : "mutualfund/logo/mirae.png",
"127042" : "mutualfund/logo/motilal.png",
"118632" : "mutualfund/logo/nippon.png",
"122639" : "mutualfund/logo/parag.jpg",
"120828" : "mutualfund/logo/quant.jpeg",
"125497" : "mutualfund/logo/sbi.png",
"118834" : "mutualfund/logo/mirae.png",
"143903" : "mutualfund/logo/ICICI.jpg",
"120841" : "mutualfund/logo/quant.jpeg",
"148490" : "mutualfund/logo/sbi.png",
"120834" : "mutualfund/logo/quant.jpeg",
"112090" : "mutualfund/logo/Kotak.png"
}



st.set_page_config(page_title="Mutual Fund Dashboard", layout="wide")
st.title("📈 Mutual Fund Dashboard")

# ---------------------------------------
# MUTUAL FUNDS LIST
# ---------------------------------------
mutual_funds = {
    "Bandhan Small Cap Fund": "147946",
    "Axis Small Cap Fund": "125354",
    "SBI Small Cap Fund": "125497",
    "quant Small Cap Fund": "120828",
    
    "Motilal Oswal Midcap Fund": "127042",
    "HSBC Midcap Fund": "151034",
    "Kotak Midcap Fund": "119775",
    "quant Mid Cap Fund": "120841",
    "Edelweiss Nifty Midcap150 Momentum 50 Index Fund": "150902",

    "Parag Parikh Flexi Cap Fund": "122639",
     "Kotak Flexicap Fund": "112090",
    
    "Nippon India Large Cap Fund": "118632",
    "Mirae Asset Large & Midcap Fund": "118834",
    "ICICI Pru BHARAT 22 FOF": "143903",
    "quant Focused Fund": "120834",
    
    
    "Mirae Asset FANG+": "148928",
    "ICICI Pru Technology Fund": "120594",
    "SBI Magnum Children's Benefit Fund": "148490"
   
}

st.sidebar.header("Your Mutual Funds")



# -----------------------
# Portfolio Tools (multiple CSVs)
# -----------------------
st.sidebar.markdown("---")
st.sidebar.header("Portfolio Tools")
st.sidebar.markdown("---")
overview_button = st.sidebar.button("Build Complete Overview")




# -----------------------
# Compare Funds
# -----------------------
st.sidebar.markdown("---")
st.sidebar.header("Compare Funds")
compare_selection = st.sidebar.multiselect(
    "Select up to 2 funds to compare (NAV lines)",
    list(mutual_funds.keys()),
    max_selections=2
)




compare_start = st.sidebar.date_input("Compare - start date", value=datetime(2020, 1, 1), key="cmp_start")
compare_end = st.sidebar.date_input("Compare - end date", value=datetime.today(), key="cmp_end")
if st.sidebar.button("Compare NAVs"):
    if len(compare_selection) < 1:
        st.sidebar.error("Select at least one fund to compare.")
    else:
        # fetch and plot comparison
        traces = []
        for fund in compare_selection:
            code = mutual_funds[fund]
            api_url = f"https://api.mfapi.in/mf/{code}?startDate={compare_start}&endDate={compare_end}"
            resp = requests.get(api_url)
            if resp.status_code != 200:
                st.error(f"Failed fetching {fund}")
                continue
            j = resp.json()
            if "data" not in j or not j["data"]:
                st.error(f"No NAV data for {fund}")
                continue
            dfj = pd.DataFrame(j["data"])
            dfj["date"] = pd.to_datetime(dfj["date"], format="%d-%m-%Y")
            dfj["nav"] = pd.to_numeric(dfj["nav"], errors="coerce")
            dfj = dfj.sort_values("date")
            traces.append((fund, dfj))
        if traces:
            fig = go.Figure()
            for name, dfj in traces:
                fig.add_trace(go.Scatter(x=dfj["date"], y=dfj["nav"], mode="lines+markers", name=name))
            fig.update_layout(title="Compare NAVs", xaxis_title="Date", yaxis_title="NAV", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------
# helper functions: delimiter detection & csv cleaner
# ---------------------------------------
def detect_delimiter(sample_bytes: bytes) -> str:
    sample = sample_bytes.decode("utf-8", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        return dialect.delimiter
    except Exception:
        return ";" if ";" in sample and sample.count(";") >= sample.count(",") else ","

def load_and_clean_csv_bytes(raw_bytes: bytes) -> pd.DataFrame:
    delim = detect_delimiter(raw_bytes)
    df = pd.read_csv(io.BytesIO(raw_bytes), sep=delim, engine="python", dtype=str)
    # strip headers and values
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # drop completely empty rows
    df.dropna(how="all", inplace=True)
    # normalize column names
    col_map = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc == "date":
            col_map[c] = "Date"
        elif lc == "units":
            col_map[c] = "Units"
        elif lc == "nav":
            col_map[c] = "NAV"
        elif lc == "amount":
            col_map[c] = "Amount"
        else:
            col_map[c] = c.strip()
    df = df.rename(columns=col_map)
    if "Date" not in df.columns:
        raise ValueError("CSV does not contain a 'Date' column. Please verify file.")
    # parse date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df[df["Date"].notna()].copy()
    # clean numeric columns
    if "Units" in df.columns:
        df["Units"] = df["Units"].astype(str).str.replace(",", "").str.strip()
        df["Units"] = pd.to_numeric(df["Units"], errors="coerce")
    if "NAV" in df.columns:
        df["NAV"] = df["NAV"].astype(str).str.replace("₹", "").str.replace(",", "").str.strip()
        df["NAV"] = pd.to_numeric(df["NAV"], errors="coerce")
    if "Amount" in df.columns:
        df["Amount"] = df["Amount"].astype(str).str.replace("₹", "").str.replace(",", "").str.strip()
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    # drop rows with no meaningful numeric info
    numeric_cols = [c for c in ["Units", "NAV", "Amount"] if c in df.columns]
    if numeric_cols:
        df = df.dropna(subset=numeric_cols, how="all")
    df = df.sort_values("Date", ascending=False).reset_index(drop=True)
    return df

# XIRR implementation (Newton-Raphson)
def xirr(cashflows, dates, guess=0.1):
    # cashflows: list of floats (negative = outflow), dates: list of datetimes
    # returns annual rate (e.g., 0.12 => 12%)
    if len(cashflows) != len(dates) or len(cashflows) == 0:
        raise ValueError("cashflows and dates must be same length > 0")
    # convert to days from first date
    days = [(d - dates[0]).days for d in dates]
    def npv(rate):
        return sum([cf / ((1 + rate) ** (d / 365.0)) for cf, d in zip(cashflows, days)])
    def npv_derivative(rate):
        return sum([- (d / 365.0) * cf / ((1 + rate) ** (1 + d/365.0)) for cf, d in zip(cashflows, days)])
    rate = guess
    for i in range(100):
        f = npv(rate)
        df = npv_derivative(rate)
        if df == 0:
            break
        new_rate = rate - f/df
        if abs(new_rate - rate) < 1e-6:
            rate = new_rate
            break
        rate = new_rate
    return rate

############################################

def units_to_sell_for_profit(df_invest, latest_nav, target_profit=125000):
    """
    Calculate how many units to sell (FIFO) to achieve target profit.
    Profit excludes investment amount.
    """
    # Sort by Date ascending (FIFO)
    df_fifo = df_invest.sort_values("Date").copy()
    df_fifo["CostPerUnit"] = df_fifo["Amount"] / df_fifo["Units"]

    total_profit = 0.0
    units_sold = 0.0
    sale_value = 0.0

    for _, row in df_fifo.iterrows():
        cost_per_unit = row["CostPerUnit"]
        profit_per_unit = latest_nav - cost_per_unit
        if profit_per_unit <= 0:
            continue  # no profit from this lot

        # Max profit possible from this lot
        lot_profit = profit_per_unit * row["Units"]

        if total_profit + lot_profit >= target_profit:
            # Only partial units needed from this lot
            remaining_profit = target_profit - total_profit
            units_needed = remaining_profit / profit_per_unit
            units_sold += units_needed
            sale_value += units_needed * latest_nav
            total_profit += remaining_profit
            break
        else:
            # Sell entire lot
            units_sold += row["Units"]
            sale_value += row["Units"] * latest_nav
            total_profit += lot_profit

    return {
        "Units to Sell": units_sold,
        "Sale Value": sale_value,
        "Profit Achieved": total_profit
    }

########################################################################

##########################################################################
# ---------------------------------------
# Single fund flow (your original UI) - unchanged behaviour + XIRR added
# ---------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("Single Fund View")
selected_fund = st.sidebar.radio("Select a Mutual Fund (Single view)", list(mutual_funds.keys()), index=0, key="single_select")

if selected_fund:
    fund_code = mutual_funds[selected_fund]
    st.subheader(f"📌 Selected Fund: {selected_fund}")

    logo_url = fund_logos.get(fund_code)
    if logo_url:
        st.image(logo_url, width=120)  # adjust width as needed


        # Fund details from MFAPI
    try:
        meta_url = f"https://api.mfapi.in/mf/{fund_code}"
        meta_resp = requests.get(meta_url, timeout=10)
        if meta_resp.status_code == 200:
            meta_json = meta_resp.json()
            if "meta" in meta_json:
                meta = meta_json["meta"]
                st.write("### 🏦 Fund Details")
                st.markdown(f"""
                - **Scheme Name:** {meta.get('scheme_name','N/A')}
                - **AMC:** {meta.get('fund_house','N/A')}
                - **Category:** {meta.get('scheme_category','N/A')}
                - **Scheme Type:** {meta.get('scheme_type','N/A')}
                - **Fund Code:** {fund_code}
                - **ISIN:** {meta.get('isin_growth','N/A')}
                """)
    except Exception:
        st.warning("Could not fetch fund details from API.")


    # st.write("### Step 1: Upload your investment CSV for this fund")
    # uploaded_file = st.file_uploader("Upload CSV (semicolon or comma)", type=["csv"], key=f"{selected_fund}_single_csv")

    # if uploaded_file is not None:
    #     raw = uploaded_file.read()
    #     try:
    #         df_invest = load_and_clean_csv_bytes(raw)
    #     except Exception as e:
    #         st.error(f"Failed to read CSV: {e}")
    #         st.stop()

    # Base folder where your CSVs are stored
    BASE_FOLDER = r"mutualfund"   # adjust to your actual folder path

#     st.write("### Step 1: Reading investment CSV automatically")

# # Construct file path based on fund name
    file_path = os.path.join(BASE_FOLDER, selected_fund, "fund.csv")
#     st.write(file_path)

file_path = os.path.join(BASE_FOLDER, selected_fund, "fund.csv")

if os.path.exists(file_path):
    try:
        with open(file_path, "rb") as f:
            raw = f.read()
        df_invest = load_and_clean_csv_bytes(raw)
    except Exception as e:
        st.error(f"Failed to read CSV for {selected_fund}: {e}")
        st.stop()

    st.subheader(f"💰 Investment Details: {selected_fund}")
    st.dataframe(df_invest, width=1000)
    st.write("### 📊 Average Buy NAV")

    # Weighted average NAV = sum(Units * NAV) / sum(Units)
    total_units = df_invest["Units"].sum()
    weighted_nav = (df_invest["Units"] * df_invest["NAV"]).sum() / total_units

    st.metric(value=f"{weighted_nav:.2f}")

    # Step 2: date range
    st.write("### Select NAV Date Range")
    default_start = df_invest["Date"].min().date()
    default_end = datetime.today().date()
    selected_dates = st.date_input(
        "Select NAV Date Range",
        value=(default_start, default_end),
        min_value=default_start,
        max_value=default_end,
        key=f"{selected_fund}_nav"
    )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        st.error("Please select a valid start and end date.")
        st.stop()

    # Continue with NAV fetch and portfolio summary...
else:
    st.error(f"No CSV file found for {selected_fund} in {BASE_FOLDER}")
    st.stop()


if st.button("Fetch NAV Data", key=f"fetch_{selected_fund}"):
            api_url = f"https://api.mfapi.in/mf/{fund_code}?startDate={start_date}&endDate={end_date}"
            st.write(f"📡 Fetching NAV data from API...")
            resp = requests.get(api_url)
            if resp.status_code != 200:
                st.error("API fetch failed.")
            else:
                j = resp.json()
                if "data" not in j or not j["data"]:
                    st.error("No NAV data for selected range.")
                else:
                    df_nav = pd.DataFrame(j["data"])
                    df_nav["date"] = pd.to_datetime(df_nav["date"], format="%d-%m-%Y")
                    df_nav["nav"] = pd.to_numeric(df_nav["nav"], errors="coerce")
                    df_nav = df_nav.sort_values("date")

                    col1, col2 = st.columns([3, 1])


                    # Plot interactive NAV chart
                    with col1:
                        st.subheader(f"📈 NAV Chart: {selected_fund}")
                        fig = px.line(df_nav, x="date", y="nav", labels={"date":"Date","nav":"NAV"}, template="plotly_white")
                        fig.update_traces(mode="lines+markers", hovertemplate="Date: %{x}<br>NAV: %{y}")
                        fig.update_layout(hovermode="x unified")
                        st.plotly_chart(fig, use_container_width=True)

                    # show NAV table
                    with col2:
                        st.subheader("📋 NAV Table")
                        # sort by date descending
                        df_nav["date"] = df_nav["date"].dt.date
                        df_nav_sorted = df_nav.sort_values(by="date", ascending=False)
                        st.dataframe(df_nav_sorted.reset_index(drop=True), use_container_width=True)

                    col11, col12, col13 = st.columns([1, 2, 2])

                  #  with col11:
                    # display only Date + NAV columns
                     #   st.dataframe(df_nav_sorted.reset_index(drop=True), use_container_width=True)


                    # # Current value & gains
                    # latest_nav = df_nav.iloc[-1]["nav"]
                    # # ensure Units and Amount present
                    # if "Units" not in df_invest.columns or "Amount" not in df_invest.columns:
                    #     st.warning("Uploaded CSV doesn't have Units or Amount columns. Skipping value/gain calculations.")
                    # else:
                    #     df_invest_current = df_invest.copy()
                    #     df_invest_current["Current Value"] = df_invest_current["Units"] * latest_nav
                    #     df_invest_current["Gain/Loss"] = df_invest_current["Current Value"] - df_invest_current["Amount"]
                    #     total_invested = df_invest_current["Amount"].sum()
                    #     total_current = df_invest_current["Current Value"].sum()
                    #     total_units = df_invest_current["Units"].sum()
                    #     total_gain = total_current - total_invested

                    latest_nav = df_nav.iloc[-1]["nav"]

                    if "Units" not in df_invest.columns or "Amount" not in df_invest.columns:
                        st.warning("Uploaded CSV doesn't have Units or Amount columns. Skipping value/gain calculations.")
                    else:
                        df_invest_current = df_invest.copy()
                        df_invest_current["Current Value"] = df_invest_current["Units"] * latest_nav
                        df_invest_current["Gain/Loss"] = df_invest_current["Current Value"] - df_invest_current["Amount"]
                    
                        # Sort by Date ascending (FIFO order)
                        df_invest_current = df_invest_current.sort_values("Date", ascending=True).reset_index(drop=True)
                        df_invest_current["Cumulative Gain"] = df_invest_current["Gain/Loss"].cumsum()
                    
                        # Totals
                        total_invested = df_invest_current["Amount"].sum()
                        total_current = df_invest_current["Current Value"].sum()
                        total_units = df_invest_current["Units"].sum()
                        total_gain = total_current - total_invested
                        total_cumulative = df_invest_current["Cumulative Gain"].iloc[-1]
                    
                        # Append totals row
                        totals_row = {
                            "Date": "TOTAL",
                            "Units": total_units,
                            "Amount": total_invested,
                            "Current Value": total_current,
                            "Gain/Loss": total_gain,
                            "Cumulative Gain": total_cumulative
                        }


    ############################################################################

                        # XIRR calculation: cashflows = investments (negative), final positive current value at last nav date
                        cashflows = []
                        dates = []
                        # investments as negative amounts (outflows)
                        for _, row in df_invest_current.iterrows():
                            if not pd.isna(row["Amount"]):
                                cashflows.append(-float(row["Amount"]))
                                dates.append(pd.to_datetime(row["Date"]))
                        # final positive
                        cashflows.append(float(total_current))
                        dates.append(pd.to_datetime(df_nav.iloc[-1]["date"]))
                        try:
                            irr = xirr(cashflows, dates)
                            irr_pct = irr * 100
                        except Exception:
                            irr_pct = None

                        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                        #col1.metric("Total Invested", f"₹ {total_invested:,.2f}")
                        col1.markdown(f"<h6>Total Invested</h6><p style='font-size:20px;'>₹ {total_invested:,.2f}</p>",unsafe_allow_html=True)
                        #col2.metric("Current Value", f"₹ {total_current:,.2f}")
                        col2.markdown(f"<h6>Current Value</h6><p style='font-size:20px;'>₹ {total_current:,.2f}</p>",unsafe_allow_html=True)
                        #col3.metric("Total Unis",f" {total_units:,.2f}")
                        col3.markdown(f"<h6>Total Unis</h6><p style='font-size:20px;'>{total_units:,.2f}</p>",unsafe_allow_html=True)
                        #col4.metric("Absolute Gain/Loss", f"₹ {total_gain:,.2f}")
                        col4.markdown(f"<h6>Absolute Gain/Loss</h6><p style='font-size:20px;'>₹ {total_gain:,.2f}</p>",unsafe_allow_html=True)
                        # col5.metric(
                        #     "XIRR (annual)",
                        #     f"{irr_pct:.2f}%" if isinstance(irr_pct, (int, float)) and not math.isnan(irr_pct) else "N/A"
                        # )

                        if (irr_pct is not None) and pd.notna(irr_pct):
                            xirr_str = f"{irr_pct:.2f}%"
                            color = "green" if irr_pct >= 0 else "red"  # optional: color-code by sign
                        else:
                            xirr_str = "N/A"
                            color = "#666"

                        col5.markdown(
                            f"""
                            <div>
                              <div style="font-size:16px; font-weight:600; color:#555;"><b>XIRR (annual)</b></div>
                              <div style="font-size:20px; color:{color};">{xirr_str}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        

                        
                        latest_nav_api = None
                        latest_nav_date = None
                        try:
                            latest_api_url = f"https://api.mfapi.in/mf/{fund_code}/latest"
                            latest_api_resp = requests.get(latest_api_url, timeout=10)
                            if latest_api_resp.status_code == 200:
                                latest_api_json = latest_api_resp.json()
                                if "data" in latest_api_json and len(latest_api_json["data"]) > 0:
                                    item = latest_api_json["data"][0]
                                    latest_nav_api = float(item["nav"])
                                    latest_nav_date = item.get("date", None)
                        except Exception:
                            pass

                        # col6.metric(
                        #     "Latest NAV (API)",
                        #     f"₹ {latest_nav_api:,.4f}" if latest_nav_api else "N/A"
                        # )
                        if latest_nav_api:
                            nav_str = f"₹ {latest_nav_api:,.4f}"
                        else:
                            nav_str = "N/A"

                        col6.markdown(
                            f"""
                            <div>
                              <div style="font-size:16px; font-weight:600; color:#555;"><b>Latest NAV (API)</b></div>
                              <div style="font-size:20px; color:#333;">{nav_str}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # col7.metric(
                        #     "NAV Date",
                        #     latest_nav_date if latest_nav_date else "N/A"
                        # )
                        if latest_nav_date:
                            nav_date = latest_nav_date
                        else:
                            nav_date = "N/A"
                        col7.markdown(
                            f"""
                            <div>
                              <div style="font-size:16px; font-weight:600; color:#555;"><b>NAV Date</b></div>
                              <div style="font-size:20px; color:#333;">{nav_date}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
#####################################
                        # if latest_nav_api:
                        #     result = units_to_sell_for_profit(df_invest, latest_nav_api, target_profit=125000)
                        #     st.subheader("🎯 Units to Sell for Target Profit")
                        #     st.metric("Units to Sell", f"{result['Units to Sell']:.2f}")
                        #     st.metric("Sale Value", f"₹ {result['Sale Value']:,.2f}")
                        #     st.metric("Profit Achieved", f"₹ {result['Profit Achieved']:,.2f}")
#######################################

                        df_invest_current = pd.concat([df_invest_current, pd.DataFrame([totals_row])], ignore_index=True)
                    
                        # Convert Date column to string (to keep TOTAL visible)
                        df_invest_current["Date"] = df_invest_current["Date"].astype(str)
                    
                        # --- Styling function ---
                        def highlight_total(row):
                            return ['background-color: #f0f0f0; font-weight: bold; color: darkblue;' 
                                    if row["Date"] == "TOTAL" else '' for _ in row]
                    
                        styled_df = df_invest_current.style.apply(highlight_total, axis=1)
                    
                        # Display
                        st.subheader("📋 Investment Details with Current Value, Gain/Loss & Cumulative Gain")
                        st.dataframe(styled_df, width=1000)

                        # st.subheader("📋 Investment Details with Current Value & Gain/Loss")
                        # df_invest_current['Date'] = pd.to_datetime(df_invest_current['Date']).dt.date
                        # st.dataframe(df_invest_current.sort_values("Date", ascending=False).reset_index(drop=True), width=1000)

# -----------------------
# Overview (all funds at once)
# -----------------------


if overview_button:
    st.header("📦 Complete Portfolio Overview")

    BASE_FOLDER = r"mutualfund"

    all_funds = []
    per_fund_summary = []

    # loop through each fund in your mutual_funds dict
    for fund_name in mutual_funds.keys():
        file_path = os.path.join(BASE_FOLDER, fund_name, "fund.csv")
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    raw = f.read()
                dff = load_and_clean_csv_bytes(raw)
                all_funds.append((fund_name, dff))
            except Exception as e:
                st.error(f"Failed to parse {fund_name}: {e}")
                continue

    if not all_funds:
        st.error("No valid CSVs found in folders.")
    else:
        total_invested_all = 0.0
        total_current_all = 0.0

        for fund_name, dff in all_funds:
            invested = dff["Amount"].sum() if "Amount" in dff.columns else 0.0

            # fetch latest NAV from API if possible
            latest_nav = None
            latest_nav_date = None

         
            matched_code = mutual_funds.get(fund_name)

            if matched_code:
                try:
                    api_url = f"https://api.mfapi.in/mf/{matched_code}?startDate=2020-01-01&endDate={datetime.today().strftime('%Y-%m-%d')}"
                    
                    r = requests.get(api_url, timeout=10)
                    
                    jr = r.json()
                   
                    if "data" in jr and jr["data"]:
                        navs = pd.DataFrame(jr["data"])
                        navs["date"] = pd.to_datetime(navs["date"], format="%d-%m-%Y")
                        navs["nav"] = pd.to_numeric(navs["nav"], errors="coerce")
                        navs = navs.sort_values("date")
                        latest_nav = float(navs.iloc[-1]["nav"])
                        latest_nav_date = navs.iloc[-1]["date"] 
                      
                except Exception:
                    pass

            if latest_nav is None and "NAV" in dff.columns and dff["NAV"].notna().any():
                latest_nav = float(dff["NAV"].dropna().iloc[0])
                latest_nav_date = pd.to_datetime(dff["Date"].max())
            if latest_nav is None:
                latest_nav = 0.0
                latest_nav_date = None


            units_sum = dff["Units"].sum() if "Units" in dff.columns else 0.0
            current_value = units_sum * latest_nav
            total_invested_all += invested
            total_current_all += current_value

            # compute XIRR for fund
            cashflows, dates = [], []
            for _, row in dff.iterrows():
                if "Amount" in dff.columns and not pd.isna(row["Amount"]):
                    cashflows.append(-float(row["Amount"]))
                    dates.append(pd.to_datetime(row["Date"]))
            cashflows.append(float(current_value))
            dates.append(pd.to_datetime(dff["Date"].max()))
            try:
                irr = xirr(cashflows, dates)
                irr_pct = irr * 100
            except Exception:
                irr_pct = None

            per_fund_summary.append({
                "Fund": fund_name,
                "Invested": invested,
                "Units": units_sum,
                "Latest NAV": latest_nav,
                "Latest NAV Date": (latest_nav_date.strftime("%Y-%m-%d") if latest_nav_date is not None else "N/A"),
                "Current Value": current_value,
                "XIRR (%)": (f"{irr_pct:.2f}%" if isinstance(irr_pct, (int, float)) and not math.isnan(irr_pct) else "N/A")
            })

        # show portfolio metrics
        st.subheader("Portfolio Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Invested", f"₹ {total_invested_all:,.2f}")
        col2.metric("Total Current Value", f"₹ {total_current_all:,.2f}")
        col3.metric("Total Gain/Loss", f"₹ {total_current_all - total_invested_all:,.2f}")

        # per-fund table
        st.subheader("Per-fund Summary")
        df_summary = pd.DataFrame(per_fund_summary).sort_values("Current Value", ascending=False).reset_index(drop=True)
        st.dataframe(df_summary, width=1000)

        # Allocation pie
        st.subheader("Allocation: Current Value by Fund")
        figp = px.pie(df_summary, names="Fund", values="Current Value", title="Portfolio Allocation")
        st.plotly_chart(figp, use_container_width=True)

        # Overall XIRR
        all_cashflows, all_dates = [], []
        for fund_name, dff in all_funds:
            for _, row in dff.iterrows():
                if "Amount" in dff.columns and not pd.isna(row["Amount"]):
                    all_cashflows.append(-float(row["Amount"]))
                    all_dates.append(pd.to_datetime(row["Date"]))
        overall_final_date = max([dff["Date"].max() for (_, dff) in all_funds])
        all_cashflows.append(float(total_current_all))
        all_dates.append(pd.to_datetime(overall_final_date))
        try:
            overall_irr = xirr(all_cashflows, all_dates)
            st.metric("Portfolio XIRR (annual)", f"{overall_irr*100:.2f}%")
        except Exception:
            st.metric("Portfolio XIRR (annual)", "N/A")






















































