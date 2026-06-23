import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="KFC Executive BI Dashboard", layout="wide", initial_sidebar_state="collapsed")

# ==========================================
# 2. DATA LOAD & PREPROCESSING
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("refference/KFC_Past_Sales_Dataset.csv")
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce').fillna(0)
    df['Customers'] = pd.to_numeric(df['Customers'], errors='coerce').fillna(0)
    df['Marketing_Spend'] = pd.to_numeric(df['Marketing_Spend'], errors='coerce').fillna(0)
    df['Sales_per_Customer'] = np.where(df['Customers'] > 0, df['Sales'] / df['Customers'], 0)
    return df

df_raw = load_data()

# ==========================================
# 3. SPK ENGINE: TOPSIS & MOORA
# ==========================================
def calculate_spk(df):
    if df.empty:
        return pd.DataFrame()
        
    branch_df = df.groupby('Branch_ID').agg({
        'Sales': 'mean',
        'Customers': 'mean',
        'Marketing_Spend': 'mean',
        'Sales_per_Customer': 'mean'
    }).reset_index()

    weights = np.array([0.45, 0.25, 0.15, 0.15])
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
    
    topsis_scores = dist_worst / (dist_best + dist_worst + 1e-10) # avoid div by zero
    
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
    
    return branch_df

# ==========================================
# 4. UI/UX: THEME & CSS INJECTION
# ==========================================
st.sidebar.title("KFC Global BI")
theme_choice = st.sidebar.radio("Theme / UI Mode", ["Light Mode", "Dark Mode"])

if theme_choice == "Light Mode":
    bg_color = "#FFFFFF"
    card_color = "#F9F9F9"
    text_color = "#000000"
    accent_color = "#767676"
    border_color = "#E5E5E5"
else:
    bg_color = "#111111"
    card_color = "#1A1A1A"
    text_color = "#FFFFFF"
    accent_color = "#9E9E9E"
    border_color = "#2E2E2E"

st.markdown(f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }}
    .css-1d391kg, .css-1lcbmhc {{
        background-color: {card_color};
    }}
    h1, h2, h3, h4, h5, h6, p, span, div {{
        color: {text_color} !important;
    }}
    .st-bb, .st-cb, .st-db, .st-eb {{
        border-color: {border_color};
    }}
    .metric-card {{
        background-color: {card_color};
        border: 1px solid {border_color};
        padding: 20px;
        border-radius: 0px;
        text-align: center;
        margin-bottom: 20px;
    }}
    .metric-value {{
        font-size: 28px;
        font-weight: bold;
        color: {text_color};
    }}
    .metric-label {{
        font-size: 14px;
        color: {accent_color};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>KFC EXECUTIVE DASHBOARD</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color: {accent_color}'>STRATEGIC BUSINESS INTELLIGENCE & MULTI-CRITERIA DECISION SUPPORT</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# 5. NAVBAR FILTERS
# ==========================================
nav_col1, nav_col2 = st.columns(2)
with nav_col1:
    country_options = ["Semua Negara"] + list(df_raw['Country'].unique())
    selected_country = st.selectbox("Select Country", options=country_options, index=0)
with nav_col2:
    year_options = ["Semua Tahun"] + list(sorted(df_raw['Year'].unique()))
    selected_year = st.selectbox("Select Year", options=year_options, index=0)

df_filtered = df_raw.copy()
if selected_country != "Semua Negara":
    df_filtered = df_filtered[df_filtered['Country'] == selected_country]
if selected_year != "Semua Tahun":
    df_filtered = df_filtered[df_filtered['Year'] == selected_year]
spk_results = calculate_spk(df_filtered)

st.markdown("---")

# ==========================================
# 6. CHARTS & VISUALIZATIONS
# ==========================================
def apply_theme_to_fig(fig):
    fig.update_layout(
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color, family="Helvetica Neue"),
        title_font=dict(color=text_color, size=18, family="Helvetica Neue"),
        legend=dict(font=dict(color=text_color)),
        margin=dict(t=50, b=40, l=40, r=40)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=border_color, tickfont=dict(color=text_color), title_font=dict(color=text_color))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=border_color, tickfont=dict(color=text_color), title_font=dict(color=text_color))
    return fig

col1, col2 = st.columns(2)

# Chart 1: Donut Chart - Sales Distribution by Country
with col1:
    if not df_filtered.empty:
        sales_by_country = df_filtered.groupby('Country')['Sales'].sum().reset_index()
        fig1 = go.Figure(data=[go.Pie(labels=sales_by_country['Country'], values=sales_by_country['Sales'], hole=.5, 
                                      marker=dict(colors=['#333333', '#666666', '#999999', '#CCCCCC', '#555555']))])
        fig1.update_layout(title="TOTAL SALES DISTRIBUTION BY COUNTRY")
        fig1 = apply_theme_to_fig(fig1)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")

# Chart 2: Bar Chart - Marketing Spend Tier Analysis
with col2:
    if not df_filtered.empty:
        q33 = df_filtered['Marketing_Spend'].quantile(0.33)
        q66 = df_filtered['Marketing_Spend'].quantile(0.66)
        
        def get_tier(spend):
            if pd.isna(spend): return "Unknown"
            if spend <= q33: return "Budget / Low"
            elif spend <= q66: return "Mid-Range"
            else: return "Premium / High"
            
        df_filtered_tier = df_filtered.copy()
        df_filtered_tier['Spend_Tier'] = df_filtered_tier['Marketing_Spend'].apply(get_tier)
        tier_sales = df_filtered_tier.groupby('Spend_Tier')['Sales'].sum().reset_index()
        tier_sales['Tier_Order'] = tier_sales['Spend_Tier'].map({"Budget / Low": 1, "Mid-Range": 2, "Premium / High": 3})
        tier_sales = tier_sales.sort_values('Tier_Order')
        
        fig2 = go.Figure(data=[go.Bar(x=tier_sales['Spend_Tier'], y=tier_sales['Sales'], marker_color=accent_color)])
        fig2.update_layout(title="SALES VOLUME BY MARKETING SPEND TIER")
        fig2 = apply_theme_to_fig(fig2)
        st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

# Chart 3: Stacked Bar Chart - Sales Grouped by Country, Sub-divided by Year
with col3:
    if not df_filtered.empty:
        sales_country_year = df_filtered.groupby(['Country', 'Year'])['Sales'].sum().reset_index()
        fig3 = go.Figure()
        colors = ['#111111', '#444444', '#777777', '#AAAAAA', '#DDDDDD'] if theme_choice == "Light Mode" else ['#FFFFFF', '#CCCCCC', '#999999', '#666666', '#333333']
        for idx, year in enumerate(sorted(sales_country_year['Year'].unique())):
            year_data = sales_country_year[sales_country_year['Year'] == year]
            fig3.add_trace(go.Bar(x=year_data['Country'], y=year_data['Sales'], name=str(year), marker_color=colors[idx % len(colors)]))
        fig3.update_layout(title="SALES BREAKDOWN: COUNTRY & YEAR", barmode='stack')
        fig3 = apply_theme_to_fig(fig3)
        st.plotly_chart(fig3, use_container_width=True)

# Chart 4: Horizontal Bar Chart - Top 10 Branches Leaderboard (TOPSIS)
with col4:
    if not spk_results.empty:
        top10_branches = spk_results.sort_values('TOPSIS_Score', ascending=False).head(10)
        fig4 = go.Figure(go.Bar(
                x=top10_branches['TOPSIS_Score'],
                y=top10_branches['Branch_ID'],
                orientation='h',
                marker_color=text_color
        ))
        fig4.update_layout(title="TOP 10 BRANCH LEADERBOARD (SPK SCORE)", yaxis={'categoryorder':'total ascending'})
        fig4 = apply_theme_to_fig(fig4)
        st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ==========================================
# 7. BUSINESS QUESTIONS & ANSWERS (BAHASA INDONESIA)
# ==========================================
st.markdown("<h2>EXECUTIVE BUSINESS INSIGHTS (Q&A)</h2>", unsafe_allow_html=True)

if not df_filtered.empty:
    avg_spend = df_filtered['Marketing_Spend'].median()
    high_promo = df_filtered[df_filtered['Marketing_Spend'] > avg_spend]['Sales'].mean()
    low_promo = df_filtered[df_filtered['Marketing_Spend'] <= avg_spend]['Sales'].mean()
    
    high_promo = high_promo if not pd.isna(high_promo) else 0
    low_promo = low_promo if not pd.isna(low_promo) else 0
    diff_promo = high_promo - low_promo
    diff_pct = (diff_promo / low_promo) * 100 if low_promo > 0 else 0

    country_hotspot = df_filtered.groupby('Country')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
    if not country_hotspot.empty:
        top_country_name = country_hotspot.iloc[0]['Country']
        top_country_sales = country_hotspot.iloc[0]['Sales']
    else:
        top_country_name = "N/A"
        top_country_sales = 0

    year_trend = df_filtered.groupby('Year')['Sales'].sum().reset_index().sort_values('Year')
    if not year_trend.empty:
        top_year = year_trend.iloc[-1]['Year']
        top_year_sales = year_trend.iloc[-1]['Sales']
    else:
        top_year = "N/A"
        top_year_sales = 0

    df_filtered_spc = df_filtered.copy()
    df_filtered_spc['Spend_Per_Customer'] = np.where(df_filtered_spc['Customers'] > 0, df_filtered_spc['Marketing_Spend'] / df_filtered_spc['Customers'], 0)
    high_tier_spc = df_filtered_spc[df_filtered_spc['Marketing_Spend'] > avg_spend]['Spend_Per_Customer'].mean()
    low_tier_spc = df_filtered_spc[df_filtered_spc['Marketing_Spend'] <= avg_spend]['Spend_Per_Customer'].mean()
    
    high_tier_spc = high_tier_spc if not pd.isna(high_tier_spc) else 0
    low_tier_spc = low_tier_spc if not pd.isna(low_tier_spc) else 0

    if not spk_results.empty:
        top_5_restock = spk_results.sort_values('TOPSIS_Score', ascending=False).head(5)['Branch_ID'].tolist()
        bottom_5_clearance = spk_results.sort_values('TOPSIS_Score', ascending=True).head(5)['Branch_ID'].tolist()
    else:
        top_5_restock = []
        bottom_5_clearance = []

    st.markdown(f"""
    <div style='background-color: {card_color}; padding: 20px; border: 1px solid {border_color}; border-radius: 0px;'>
        <ul style='color: {accent_color}; font-size: 16px; line-height: 1.8;'>
            <li><b>Efektivitas Investasi Marketing:</b> Rata-rata penjualan untuk cabang dengan pengeluaran marketing di atas median mencapai <b>${high_promo:,.2f}</b>, dibandingkan dengan <b>${low_promo:,.2f}</b> pada cabang di bawah median. Terdapat selisih signifikan sebesar <b>${diff_promo:,.2f} ({diff_pct:.2f}%)</b>, membuktikan bahwa investasi pemasaran secara langsung mendorong volume penjualan.</li>
            <br>
            <li><b>Hotspot Penjualan Tertinggi:</b> Melalui agregasi total penjualan berdasarkan area yang difilter, <b>{top_country_name}</b> menjadi hotspot absolut dengan kontribusi penjualan sebesar <b>${top_country_sales:,.2f}</b>. Sangat disarankan untuk memprioritaskan alokasi pasokan bahan baku dan inisiatif ekspansi strategis pada area ini.</li>
            <br>
            <li><b>Tren Serapan Pasar:</b> Berdasarkan data historis terpilih, volume total penjualan tertinggi tercatat pada tahun <b>{top_year}</b> dengan total <b>${top_year_sales:,.2f}</b>. Angka ini merepresentasikan efektivitas penetrasi pasar dari waktu ke waktu.</li>
            <br>
            <li><b>Efisiensi Biaya per Pelanggan (Premium vs Budget):</b> Cabang dengan pengeluaran marketing tier premium menghabiskan rata-rata <b>${high_tier_spc:,.2f}</b> per pelanggan, sedangkan cabang dengan budget lebih rendah menghabiskan <b>${low_tier_spc:,.2f}</b> per pelanggan. Analisis ini mengindikasikan bahwa ada batas sensitivitas di mana peningkatan dana promosi tidak selalu berbanding lurus dengan efisiensi akuisisi pelanggan baru.</li>
            <br>
            <li><b>Rekomendasi Tindakan (MCDM Terintegrasi):</b> Menggunakan model objektif pada data saat ini, cabang yang direkomendasikan untuk <b>Prioritas Reward & Ekspansi (Top 5)</b> adalah <b>{', '.join(top_5_restock)}</b>. Sementara itu, cabang yang paling mendesak untuk <b>Prioritas Revamp & Evaluasi Operasional (Bottom 5)</b> adalah <b>{', '.join(bottom_5_clearance)}</b>.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Pilih minimal satu negara dan tahun untuk melihat Business Insights.")

st.markdown("---")

# ==========================================
# 8. COLLAPSIBLE TECHNICAL DOCUMENTATION
# ==========================================
with st.expander("VIEW TECHNICAL DOCUMENTATION & MATHEMATICAL MODELS"):
    st.markdown("### Model Pengambilan Keputusan Multi-Kriteria (MCDM)")
    
    st.markdown("**1. Vector Normalization (TOPSIS & MOORA)**")
    st.markdown("Mengubah dimensi kriteria yang berbeda menjadi skala matriks ternormalisasi ($r_{ij}$).")
    st.latex(r"r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{i=1}^{m} x_{ij}^2}}")
    
    st.markdown("**2. Weighted Normalized Decision Matrix**")
    st.markdown("Mengalikan matriks ternormalisasi dengan bobot kriteria ($w_j$).")
    st.latex(r"v_{ij} = w_j \times r_{ij}")
    
    st.markdown("**3. TOPSIS: Ideal Best ($A^+$) and Ideal Worst ($A^-$)**")
    st.markdown("Mencari nilai maksimum untuk kriteria *Benefit* dan minimum untuk kriteria *Cost*.")
    st.latex(r"A^+ = \{v_1^+, v_2^+, ..., v_n^+\}, \quad A^- = \{v_1^-, v_2^-, ..., v_n^-\}")
    
    st.markdown("**4. TOPSIS: Separation Measures**")
    st.markdown("Menghitung jarak Euclidean dari Solusi Ideal Positif ($S_i^+$) dan Negatif ($S_i^-$).")
    st.latex(r"S_i^+ = \sqrt{\sum_{j=1}^{n} (v_{ij} - v_j^+)^2}, \quad S_i^- = \sqrt{\sum_{j=1}^{n} (v_{ij} - v_j^-)^2}")
    
    st.markdown("**5. TOPSIS: Closeness Coefficient**")
    st.markdown("Skor akhir untuk pemeringkatan (semakin mendekati 1 semakin baik).")
    st.latex(r"C_i = \frac{S_i^-}{S_i^+ + S_i^-}")
    
    st.markdown("**6. MOORA: Optimization Equation**")
    st.markdown("Skor dihitung dengan mengurangi total kriteria *Cost* dari total kriteria *Benefit*.")
    st.latex(r"y_i = \sum_{j=1}^{g} v_{ij} - \sum_{j=g+1}^{n} v_{ij}")
