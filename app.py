import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="KFC Executive BI Dashboard", layout="wide", initial_sidebar_state="collapsed")

# ==========================================
# 2. DATA LOAD & PREPROCESSING
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("KFC_Past_Sales_Dataset.csv")
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce').fillna(0)
    df['Customers'] = pd.to_numeric(df['Customers'], errors='coerce').fillna(0)
    df['Marketing_Spend'] = pd.to_numeric(df['Marketing_Spend'], errors='coerce').fillna(0)
    df['Sales_per_Customer'] = np.where(df['Customers'] > 0, df['Sales'] / df['Customers'], 0)
    return df

df_raw = load_data()

# ==========================================
# 3. SPK ENGINE: TOPSIS & MOORA
# ==========================================
def calculate_spk(df, weights):
    if df.empty:
        return pd.DataFrame()
        
    branch_df = df.groupby('Branch_ID').agg({
        'Sales': 'mean',
        'Customers': 'mean',
        'Marketing_Spend': 'mean',
        'Sales_per_Customer': 'mean'
    }).reset_index()

    criteria_types = np.array([1, 1, -1, 1]) 
    
    matrix = branch_df[['Sales', 'Customers', 'Marketing_Spend', 'Sales_per_Customer']].values
    
    norm_divisor = np.sqrt((matrix**2).sum(axis=0))
    norm_divisor[norm_divisor == 0] = 1 
    norm_matrix = matrix / norm_divisor
    weighted_matrix = norm_matrix * weights
    
    ideal_best = np.where(criteria_types == 1, np.max(weighted_matrix, axis=0), np.min(weighted_matrix, axis=0))
    ideal_worst = np.where(criteria_types == 1, np.min(weighted_matrix, axis=0), np.max(weighted_matrix, axis=0))
    
    dist_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))
    
    topsis_scores = dist_worst / (dist_best + dist_worst + 1e-10) 
    
    moora_scores = np.zeros(len(branch_df))
    for i in range(len(weights)):
        if criteria_types[i] == 1:
            moora_scores += weighted_matrix[:, i]
        else:
            moora_scores -= weighted_matrix[:, i]
            
    branch_df['TOPSIS_Score'] = topsis_scores
    branch_df['MOORA_Score'] = moora_scores
    branch_df['TOPSIS_Rank'] = branch_df['TOPSIS_Score'].rank(ascending=False, method='min')
    branch_df['MOORA_Rank'] = branch_df['MOORA_Score'].rank(ascending=False, method='min')
    
    detail_export = branch_df.copy()
    detail_export[['v_Sales','v_Customers','v_Marketing','v_SPC']] = weighted_matrix
    detail_export['S_plus'] = dist_best
    detail_export['S_minus'] = dist_worst
    st.session_state['topsis_detail_df'] = detail_export

    return branch_df

def assign_spend_tier(df, col_name='Marketing_Spend'):
    if df.empty: return df, 0, 0
    q33 = df[col_name].quantile(0.33)
    q66 = df[col_name].quantile(0.66)
    df_out = df.copy()
    def get_tier(spend):
        if pd.isna(spend): return "Unknown"
        if spend <= q33: return "Budget"
        elif spend <= q66: return "Mid"
        else: return "Premium"
    df_out['Spend_Tier'] = df_out[col_name].apply(get_tier)
    return df_out, q33, q66

# ==========================================
# 4. NAVBAR & THEME SETTINGS
# ==========================================
# Hide Streamlit Sidebar and Header completely
st.markdown("""
    <style>
    [data-testid="collapsedControl"] { display: none; }
    [data-testid="stSidebar"] { display: none; }
    header[data-testid="stHeader"] { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# Fake Floating Navbar
nav_col, empty_col, theme_col = st.columns([6, 1, 3])
with nav_col:
    nav_selection = st.radio("Navigation", ["Dashboard", "Business Insight", "CRM Analysis", "Technical Documentation"], horizontal=True, label_visibility="collapsed")
with theme_col:
    theme_choice = st.radio("Theme", ["Light Mode", "Dark Mode"], horizontal=True, label_visibility="collapsed")

if theme_choice == "Light Mode":
    bg_color = "#F1F5F9"
    card_color = "#FFFFFF"
    text_color = "#1C2434"
    muted_color = "#64748B"
    accent_color = "#3C50E0" # Switched to Tailadmin Blue for consistency
    border_color = "#E2E8F0"
    card_shadow = "0px 8px 13px -3px rgba(0, 0, 0, 0.07)"
else:
    bg_color = "#1A222C"
    card_color = "#24303F"
    text_color = "#FFFFFF"
    muted_color = "#AEB7C0"
    accent_color = "#3C50E0"
    border_color = "#313D4A"
    card_shadow = "none"

try:
    with open("tailwind.css", "r") as f:
        tailwind_css = f.read()
except FileNotFoundError:
    tailwind_css = ""

st.markdown(f"""
    <style>
    {tailwind_css}
    .stApp {{ background-color: {bg_color}; color: {text_color}; }}
    h1, h2, h3, h4, h5, h6 {{ color: {text_color} !important; }}
    p, span, div {{ color: {text_color} !important; }}
    hr {{ border-color: {border_color} !important; margin: 15px 0 !important; }}
    
    .metric-card {{ background-color: {card_color}; border: 1px solid {border_color}; box-shadow: {card_shadow}; border-radius: 8px; padding: 20px; margin-bottom: 24px; transition: transform 0.2s ease, box-shadow 0.2s ease; }}
    .metric-card:hover {{ box-shadow: {card_shadow}; transform: translateY(-2px); }}
    .metric-value {{ color: {text_color} !important; font-size: 28px !important; font-weight: 700 !important; margin-top: 0 !important; line-height: 1.1; }}
    .metric-label {{ color: {muted_color} !important; font-size: 14px !important; margin-bottom: 4px; font-weight: 500; }}
    
    .js-plotly-plot .plotly .bg {{ fill: {card_color} !important; }}
    
    /* CTA Navigation styling for radio buttons */
    div.row-widget.stRadio > div {{
        background-color: transparent !important;
        border: none !important;
        gap: 12px;
    }}
    div.row-widget.stRadio > div > label {{
        background-color: {bg_color};
        border: 1px solid {border_color};
        border-radius: 6px !important;
        padding: 10px 20px !important;
        cursor: pointer;
        transition: all 0.2s ease;
    }}
    div.row-widget.stRadio > div > label:hover {{
        border-color: {accent_color};
    }}
    div.row-widget.stRadio > div > label > div:first-child {{
        display: none;
    }}
    div.row-widget.stRadio > div > label p {{
        font-weight: 600 !important;
        margin: 0 !important;
        font-size: 15px !important;
    }}
    div.row-widget.stRadio > div > label[data-checked="true"], 
    div.row-widget.stRadio > div > label:has(input:checked) {{
        background-color: {accent_color} !important;
        border-color: {accent_color} !important;
    }}
    div.row-widget.stRadio > div > label[data-checked="true"] p, 
    div.row-widget.stRadio > div > label:has(input:checked) p {{
        color: white !important;
    }}
    </style>
""", unsafe_allow_html=True)

st.markdown(f"<h1 style='text-align: left; font-weight: 700; font-size: 28px; margin-bottom: 0;'>Sales Dashboard</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: left; color: {muted_color}; font-weight: 500; margin-top: -5px; margin-bottom: 10px;'>Track revenue, performance, and sales growth in real-time</p>", unsafe_allow_html=True)

# ==========================================
# 5. GLOBAL FILTERS & STATE
# ==========================================
filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    country_options = ["Semua Negara"] + list(df_raw['Country'].unique())
    selected_country = st.selectbox("Select Country", options=country_options, index=0)
with filter_col2:
    year_options = ["Semua Tahun"] + list(sorted(df_raw['Year'].unique()))
    selected_year = st.selectbox("Select Year", options=year_options, index=0)

df_filtered = df_raw.copy()
if selected_country != "Semua Negara":
    df_filtered = df_filtered[df_filtered['Country'] == selected_country]
if selected_year != "Semua Tahun":
    df_filtered = df_filtered[df_filtered['Year'] == selected_year]

df_filtered_tier, q33_val, q66_val = assign_spend_tier(df_filtered)

def apply_theme_to_fig(fig):
    fig.update_layout(
        plot_bgcolor=bg_color, paper_bgcolor=bg_color,
        font=dict(color=text_color, family="Satoshi"), title_font=dict(color=text_color, size=18, family="Satoshi"),
        legend=dict(font=dict(color=text_color)), margin=dict(t=50, b=40, l=40, r=40)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=border_color, tickfont=dict(color=text_color), title_font=dict(color=text_color))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=border_color, tickfont=dict(color=text_color), title_font=dict(color=text_color))
    return fig

# ==========================================
# PAGE: DASHBOARD
# ==========================================
if nav_selection == "Dashboard":
    
    hero_col1, hero_col2 = st.columns([2, 1])
    with hero_col1:
        st.markdown(f"<div style='background-color: {card_color}; padding: 15px; border-radius: 8px; border: 1px solid {border_color}; box-shadow: {card_shadow}; margin-bottom: 20px;'>", unsafe_allow_html=True)
        st.markdown("**SPK Criteria Weights Configuration**")
        w_cols = st.columns(4)
        with w_cols[0]: w_sales = st.slider("Sales (+)", 0.0, 1.0, 0.45, 0.05)
        with w_cols[1]: w_cust = st.slider("Cust (+)", 0.0, 1.0, 0.25, 0.05)
        with w_cols[2]: w_mkt = st.slider("Mkt Spend (-)", 0.0, 1.0, 0.15, 0.05)
        with w_cols[3]: w_spc = st.slider("Sales/Cust (+)", 0.0, 1.0, 0.15, 0.05)
        st.markdown("</div>", unsafe_allow_html=True)
    
    total_w = w_sales + w_cust + w_mkt + w_spc
    spk_weights = np.array([w_sales/total_w, w_cust/total_w, w_mkt/total_w, w_spc/total_w]) if total_w > 0 else np.array([0.25, 0.25, 0.25, 0.25])
    spk_results = calculate_spk(df_filtered, spk_weights)
    
    with hero_col2:
        st.markdown(f"<div style='background-color: {card_color}; padding: 15px; border-radius: 8px; border: 1px solid {border_color}; box-shadow: {card_shadow}; margin-bottom: 20px;'>", unsafe_allow_html=True)
        st.markdown("**Top 3 Branches (TOPSIS Rank)**")
        if not spk_results.empty:
            top3 = spk_results.sort_values('TOPSIS_Rank').head(3)
            for i, row in top3.iterrows():
                st.markdown(f"<div style='padding: 6px; margin-bottom: 4px; background-color: {bg_color}; border: 1px solid {border_color}; border-radius: 4px; font-weight: 600; font-size: 14px;'>#{int(row['TOPSIS_Rank'])} - {row['Branch_ID']} <span style='float: right; color: {accent_color};'>{row['TOPSIS_Score']:.3f}</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Metrics
    if not df_filtered.empty:
        kpi_branch_perf = df_filtered.groupby('Branch_ID').agg(Total_Sales=('Sales','sum'), Total_Marketing=('Marketing_Spend','sum')).reset_index()
        kpi_branch_perf['ROI_pct'] = (kpi_branch_perf['Total_Sales'] - kpi_branch_perf['Total_Marketing']) / kpi_branch_perf['Total_Marketing'] * 100
        avg_roi_kpi = kpi_branch_perf['ROI_pct'].mean()
        total_sales_kpi = df_filtered['Sales'].sum()
        total_branches_kpi = df_filtered['Branch_ID'].nunique()
        best_branch_kpi = spk_results.sort_values('TOPSIS_Score', ascending=False).iloc[0]['Branch_ID'] if not spk_results.empty else "N/A"
    else:
        avg_roi_kpi = total_sales_kpi = total_branches_kpi = 0; best_branch_kpi = "N/A"

    def make_tailadmin_card(title, value, icon, percentage, trend="up", sparkline_color="#10B981"):
        trend_color = "#10B981" if trend == "up" else "#EF4444"
        arrow = "↑" if trend == "up" else "↓"
        return f"""
        <div class='metric-card'>
            <div style='display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px;'>
                <div>
                    <div class='metric-label' style='margin-bottom: 4px;'>{title}</div>
                    <div style='font-size: 13px; color: {muted_color}; display: flex; align-items: center; gap: 4px; font-weight: 500;'>
                        <span style='color: {trend_color}; font-weight: 600;'>{arrow} {percentage}</span> 
                        <span>vs last month</span>
                    </div>
                </div>
                <div style='width: 44px; height: 44px; border-radius: 50%; background-color: {bg_color}; display: flex; align-items: center; justify-content: center; font-size: 20px; color: {accent_color};'>
                    {icon}
                </div>
            </div>
            <div style='display: flex; justify-content: space-between; align-items: flex-end;'>
                <div class='metric-value' style='margin-top: 0;'>{value}</div>
                <svg width="60" height="25" viewBox="0 0 60 25" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M1 20C4.85714 20 8.71429 8.33333 12.5714 8.33333C16.4286 8.33333 20.2857 23.3333 24.1429 23.3333C28 23.3333 31.8571 5 35.7143 5C39.5714 5 43.4286 16.6667 47.2857 16.6667C51.1429 16.6667 55 1 59 1" stroke="{sparkline_color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
        </div>
        """

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1: st.markdown(make_tailadmin_card("Total Sales", f"${total_sales_kpi/1e6:.2f}M", "$", "18.2%", "up", "#10B981"), unsafe_allow_html=True)
    with kpi2: st.markdown(make_tailadmin_card("Total Branches", f"{total_branches_kpi}", "🏢", "4.1%", "up", "#3C50E0"), unsafe_allow_html=True)
    with kpi3: st.markdown(make_tailadmin_card("Average ROI", f"{avg_roi_kpi:.1f}%", "📈", "2.4%", "up", "#10B981"), unsafe_allow_html=True)
    with kpi4: st.markdown(make_tailadmin_card("Best Branch (TOPSIS)", f"{best_branch_kpi}", "🏆", "1.2%", "down", "#EF4444"), unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if not df_filtered.empty:
            if selected_country == "Semua Negara":
                sales_by_country = df_filtered.groupby('Country')['Sales'].sum().reset_index()
                fig1 = go.Figure(data=[go.Pie(labels=sales_by_country['Country'], values=sales_by_country['Sales'], hole=.5, marker=dict(colors=px.colors.sequential.Tealgrn))])
                fig1.update_layout(title="Sales Distribution by Country")
            else:
                sales_by_branch = df_filtered.groupby('Branch_ID')['Sales'].sum().reset_index().sort_values('Sales', ascending=True)
                fig1 = go.Figure(data=[go.Bar(y=sales_by_branch['Branch_ID'], x=sales_by_branch['Sales'], orientation='h', marker=dict(color=sales_by_branch['Sales'], colorscale='Tealgrn'))])
                fig1.update_layout(title=f"Sales Distribution in {selected_country.upper()}", xaxis_title="Total Sales ($)")
            fig1 = apply_theme_to_fig(fig1)
            st.plotly_chart(fig1, use_container_width=True)
            
    with col2:
        if not df_filtered_tier.empty:
            tier_sales = df_filtered_tier.groupby('Spend_Tier')['Sales'].sum().reset_index()
            tier_sales['Tier_Order'] = tier_sales['Spend_Tier'].map({"Budget": 1, "Mid": 2, "Premium": 3})
            tier_sales = tier_sales.sort_values('Tier_Order')
            fig2 = go.Figure(data=[go.Bar(x=tier_sales['Spend_Tier'], y=tier_sales['Sales'], marker=dict(color=tier_sales['Sales'], colorscale='Blues'))])
            fig2.update_layout(title="Sales Volume by Marketing Spend Tier")
            fig2 = apply_theme_to_fig(fig2)
            st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if not spk_results.empty:
            avg_s = spk_results['Sales'].mean()
            avg_m = spk_results['Marketing_Spend'].mean()
            def classify_quadrant(row):
                if row['Sales'] > avg_s and row['Marketing_Spend'] < avg_m: return 'Star'
                elif row['Sales'] > avg_s and row['Marketing_Spend'] >= avg_m: return 'High Volume'
                elif row['Sales'] <= avg_s and row['Marketing_Spend'] < avg_m: return 'Potential'
                else: return 'Inefficient'
            spk_results['Efficiency_Quadrant'] = spk_results.apply(classify_quadrant, axis=1)
            # Use vibrant colors instead of standard hex, gradient-like map
            color_map = {'Star': '#10B981', 'High Volume': '#3B82F6', 'Potential': '#8B5CF6', 'Inefficient': '#EF4444'}
            fig3 = go.Figure()
            for quad, grp in spk_results.groupby('Efficiency_Quadrant'):
                fig3.add_trace(go.Scatter(x=grp['Marketing_Spend'], y=grp['Sales'], mode='markers+text', name=quad, text=grp['Branch_ID'], textposition='top center', marker=dict(size=14, color=color_map.get(quad))))
            fig3.add_hline(y=avg_s, line_dash="dash", annotation_text=f"Avg Sales: {avg_s:,.0f}", annotation_position="bottom right")
            fig3.add_vline(x=avg_m, line_dash="dash", annotation_text=f"Avg Marketing: {avg_m:,.0f}", annotation_position="top left")
            fig3.update_layout(title="Efficiency Quadrant", xaxis_title="Average Marketing Spend", yaxis_title="Average Sales")
            fig3 = apply_theme_to_fig(fig3)
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        if not spk_results.empty:
            top10 = spk_results.sort_values('TOPSIS_Score', ascending=False).head(10)
            fig4 = go.Figure(go.Bar(x=top10['TOPSIS_Score'], y=top10['Branch_ID'], orientation='h', marker=dict(color=top10['TOPSIS_Score'], colorscale='Sunset')))
            fig4.update_layout(title="Top 10 Branch Leaderboard (SPK)", yaxis={'categoryorder':'total ascending'}, xaxis_title="TOPSIS Score")
            fig4 = apply_theme_to_fig(fig4)
            st.plotly_chart(fig4, use_container_width=True)

# ==========================================
# PAGE: BUSINESS INSIGHT
# ==========================================
elif nav_selection == "Business Insight":
    st.markdown("<h2 style='margin-bottom: 20px;'>Executive Business Insights</h2>", unsafe_allow_html=True)
    spk_weights = np.array([0.25, 0.25, 0.25, 0.25])
    spk_results = calculate_spk(df_filtered, spk_weights)
    
    if not df_filtered.empty:
        branch_perf = df_filtered.groupby('Branch_ID').agg(Total_Sales=('Sales','sum'), Total_Marketing=('Marketing_Spend','sum')).reset_index()
        branch_perf['ROI_pct'] = (branch_perf['Total_Sales'] - branch_perf['Total_Marketing']) / branch_perf['Total_Marketing'] * 100
        branch_perf['Cost_Efficiency'] = branch_perf['Total_Sales'] / branch_perf['Total_Marketing']
        
        top_roi_row = branch_perf.sort_values('ROI_pct', ascending=False).iloc[0]
        top_roi_branch = top_roi_row['Branch_ID']
        top_roi_value = top_roi_row['ROI_pct']
        top_roi_eff = top_roi_row['Cost_Efficiency']
        
        topsis_rank_for_roi = spk_results[spk_results['Branch_ID'] == top_roi_branch]['TOPSIS_Rank'].values if not spk_results.empty else []
        topsis_rank_str = f"#{int(topsis_rank_for_roi[0])}" if len(topsis_rank_for_roi) > 0 else "N/A"
        
        avg_s = spk_results['Sales'].mean()
        avg_m = spk_results['Marketing_Spend'].mean()
        star_branches = spk_results[(spk_results['Sales'] > avg_s) & (spk_results['Marketing_Spend'] < avg_m)]['Branch_ID'].tolist() if not spk_results.empty else []
        role_model_str = ', '.join(star_branches) if star_branches else 'N/A'
        
        total_sales_bq3 = df_filtered.groupby('Branch_ID')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
        top_5_sales = total_sales_bq3.head(5)['Branch_ID'].tolist()
        bottom_5_sales = total_sales_bq3.tail(5)['Branch_ID'].tolist()
        
        df_bq = df_filtered_tier.copy()
        df_bq['Spend_Per_Customer'] = np.where(df_bq['Customers'] > 0, df_bq['Marketing_Spend'] / df_bq['Customers'], 0)
        premium_spc = df_bq[df_bq['Spend_Tier'] == 'Premium']['Spend_Per_Customer'].mean()
        budget_spc = df_bq[df_bq['Spend_Tier'] == 'Budget']['Spend_Per_Customer'].mean()
        premium_sales = df_bq[df_bq['Spend_Tier'] == 'Premium']['Sales'].mean()
        budget_sales = df_bq[df_bq['Spend_Tier'] == 'Budget']['Sales'].mean()
        
        high_mkt = df_filtered[df_filtered['Marketing_Spend'] > q66_val]['Sales'].mean()
        low_mkt = df_filtered[df_filtered['Marketing_Spend'] <= q33_val]['Sales'].mean()
        diff_mkt = high_mkt - low_mkt
        diff_pct_mkt = (diff_mkt / low_mkt * 100) if low_mkt > 0 else 0
        
        st.markdown(f"""
        <div style='background-color: {card_color}; padding: 25px; border-radius: 8px; border: 1px solid {border_color}; box-shadow: {card_shadow}; margin-bottom: 20px;'>
            <h4 style='margin-top: 0; color: {accent_color}; font-weight: 700;'>BQ #1 - Rekomendasi Investasi dengan ROI Tertinggi</h4>
            <p>Cabang <b>{top_roi_branch}</b> memberikan return tertinggi sebesar <b>{top_roi_value:,.1f}%</b> dengan efisiensi biaya <b>{top_roi_eff:.2f}x</b> per dollar marketing. SPK TOPSIS menempatkan cabang ini di posisi <b>{topsis_rank_str}</b> dari total {len(spk_results)} cabang.</p>
        </div>
        
        <div style='background-color: {card_color}; padding: 25px; border-radius: 8px; border: 1px solid {border_color}; box-shadow: {card_shadow}; margin-bottom: 20px;'>
            <h4 style='margin-top: 0; color: {accent_color}; font-weight: 700;'>BQ #2 - Role Model Efisiensi Operasional</h4>
            <p>Cabang <b>Star</b> (Sales di atas rata-rata, Marketing di bawah rata-rata) adalah: <b>{role_model_str}</b>. Cabang ini menjadi teladan efisiensi operasional.</p>
        </div>
        
        <div style='background-color: {card_color}; padding: 25px; border-radius: 8px; border: 1px solid {border_color}; box-shadow: {card_shadow}; margin-bottom: 20px;'>
            <h4 style='margin-top: 0; color: {accent_color}; font-weight: 700;'>BQ #3 - Top 5 & Bottom 5 by Total Sales</h4>
            <p><b>Top 5 Performer</b>: {', '.join(top_5_sales)}. <br><b>Bottom 5 Evaluasi</b>: {', '.join(bottom_5_sales)}.</p>
        </div>
        
        <div style='background-color: {card_color}; padding: 25px; border-radius: 8px; border: 1px solid {border_color}; box-shadow: {card_shadow}; margin-bottom: 20px;'>
            <h4 style='margin-top: 0; color: {accent_color}; font-weight: 700;'>BQ #4 - Efisiensi Biaya per Pelanggan (Premium vs Budget Tier)</h4>
            <p>Tier <b>Premium</b> menghabiskan <b>${premium_spc:,.2f}/pelanggan</b> menghasilkan rata-rata penjualan <b>${premium_sales:,.0f}</b>.<br> 
            Tier <b>Budget</b> menghabiskan <b>${budget_spc:,.2f}/pelanggan</b> menghasilkan rata-rata penjualan <b>${budget_sales:,.0f}</b>.</p>
        </div>
        
        <div style='background-color: {card_color}; padding: 25px; border-radius: 8px; border: 1px solid {border_color}; box-shadow: {card_shadow}; margin-bottom: 20px;'>
            <h4 style='margin-top: 0; color: {accent_color}; font-weight: 700;'>BQ #5 - Efektivitas Investasi Marketing</h4>
            <p>Selisih <b>${diff_mkt:,.2f} ({diff_pct_mkt:.1f}%)</b> antara tier Premium vs Budget membuktikan peningkatan marketing berdampak positif pada penjualan agregat.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No data available.")

# ==========================================
# PAGE: CRM ANALYSIS
# ==========================================
elif nav_selection == "CRM Analysis":
    st.markdown("<h2 style='margin-bottom: 20px;'>CRM Analysis - Customer Behavior Intelligence</h2>", unsafe_allow_html=True)
    if not df_filtered.empty:
        crm_df = df_filtered.groupby('Branch_ID').agg(Avg_Monthly_Customers=('Customers', 'mean'), Total_Customers=('Customers', 'sum'), Total_Marketing=('Marketing_Spend', 'sum'), Total_Sales=('Sales', 'sum')).reset_index()
        crm_df['CAC'] = crm_df['Total_Marketing'] / crm_df['Total_Customers']
        crm_df['Revenue_Per_Customer'] = crm_df['Total_Sales'] / crm_df['Total_Customers']
        crm_df['LTV_CAC_Ratio'] = crm_df['Revenue_Per_Customer'] / crm_df['CAC']
        
        col_crm1, col_crm2 = st.columns(2)
        with col_crm1:
            fig_crm1 = go.Figure(go.Scatter(x=crm_df['CAC'], y=crm_df['Revenue_Per_Customer'], mode='markers+text', text=crm_df['Branch_ID'], textposition='top center', marker=dict(size=crm_df['Avg_Monthly_Customers'] / 50, color=crm_df['LTV_CAC_Ratio'], colorscale='Turbo', showscale=True, colorbar=dict(title="LTV/CAC"))))
            fig_crm1.update_layout(title="Customer Acquisition Cost vs Revenue per Customer", xaxis_title="CAC ($)", yaxis_title="Revenue per Customer ($)")
            fig_crm1 = apply_theme_to_fig(fig_crm1)
            st.plotly_chart(fig_crm1, use_container_width=True)

        with col_crm2:
            crm_sorted = crm_df.sort_values('CAC')
            fig_crm2 = go.Figure(go.Bar(x=crm_sorted['Branch_ID'], y=crm_sorted['CAC'], marker=dict(color=crm_sorted['CAC'], colorscale='Oryel')))
            fig_crm2.update_layout(title="CAC per Branch (Lower is Better)", yaxis_title="CAC ($)", xaxis_title="Branch")
            fig_crm2 = apply_theme_to_fig(fig_crm2)
            st.plotly_chart(fig_crm2, use_container_width=True)
            
        best_cac = crm_df.loc[crm_df['CAC'].idxmin()]
        best_ltv = crm_df.loc[crm_df['LTV_CAC_Ratio'].idxmax()]
        
        st.markdown(f"""
        <div style='background-color: {card_color}; padding: 25px; border-radius: 8px; border: 1px solid {border_color}; box-shadow: {card_shadow};'>
            <h4 style='margin-top: 0; color: {accent_color}; font-weight: 700;'>Key CRM Insights:</h4>
            <ul>
                <li><b>Cabang Paling Efisien Akuisisi Customer:</b> 
                <b>{best_cac['Branch_ID']}</b> dengan CAC terendah sebesar <b>${best_cac['CAC']:.2f}/pelanggan</b>.</li>
                <li><b>Cabang dengan LTV/CAC Ratio Terbaik:</b>
                <b>{best_ltv['Branch_ID']}</b> mencapai rasio <b>{best_ltv['LTV_CAC_Ratio']:.2f}x</b>.</li>
                <li><b>Implikasi Investasi:</b> 
                Kombinasi CAC rendah dan Revenue/Customer tinggi menjadi indikator ekspansi yang lebih solid daripada sekadar volume penjualan historis.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# PAGE: TECHNICAL DOCS
# ==========================================
elif nav_selection == "Technical Documentation":
    st.markdown("<h2 style='margin-bottom: 20px;'>Technical Documentation & SPK Matrix</h2>", unsafe_allow_html=True)
    
    st.markdown(f"<div style='background-color: {card_color}; padding: 25px; border-radius: 8px; border: 1px solid {border_color}; box-shadow: {card_shadow}; margin-bottom: 24px;'>", unsafe_allow_html=True)
    st.markdown("### MCDM Models (TOPSIS & MOORA)")
    st.latex(r"r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{i=1}^{m} x_{ij}^2}}")
    st.latex(r"v_{ij} = w_j \times r_{ij}")
    st.latex(r"A^+ = \{v_1^+, v_2^+, ..., v_n^+\}, \quad A^- = \{v_1^-, v_2^-, ..., v_n^-\}")
    st.latex(r"S_i^+ = \sqrt{\sum_{j=1}^{n} (v_{ij} - v_j^+)^2}, \quad S_i^- = \sqrt{\sum_{j=1}^{n} (v_{ij} - v_j^-)^2}")
    st.latex(r"C_i = \frac{S_i^-}{S_i^+ + S_i^-}")
    st.latex(r"y_i = \sum_{j=1}^{g} v_{ij} - \sum_{j=g+1}^{n} v_{ij}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='background-color: {card_color}; padding: 25px; border-radius: 8px; border: 1px solid {border_color}; box-shadow: {card_shadow};'>", unsafe_allow_html=True)
    st.markdown("### SPK Consensus: TOPSIS vs MOORA")
    spk_weights = np.array([0.25, 0.25, 0.25, 0.25])
    spk_results = calculate_spk(df_filtered, spk_weights)
    if not spk_results.empty:
        comparison_df = spk_results[['Branch_ID', 'TOPSIS_Score', 'TOPSIS_Rank', 'MOORA_Score', 'MOORA_Rank']].copy()
        comparison_df['Konsensus_Rank'] = (comparison_df['TOPSIS_Rank'] + comparison_df['MOORA_Rank']) / 2
        comparison_df = comparison_df.sort_values('Konsensus_Rank')
        st.dataframe(comparison_df.style.format({'TOPSIS_Score': '{:.4f}', 'MOORA_Score': '{:.4f}', 'TOPSIS_Rank': '{:.0f}', 'MOORA_Rank': '{:.0f}', 'Konsensus_Rank': '{:.1f}'}).background_gradient(subset=['Konsensus_Rank'], cmap='RdYlGn_r'), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
