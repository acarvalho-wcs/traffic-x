import pandas as pd
import numpy as np
import re
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2_contingency
import networkx as nx

# Load and clean data (handles multiple species)
def load_and_clean_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.replace('\xa0', '', regex=True)
    df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})').astype(float)

    def expand_multi_species_rows(df):
        expanded_rows = []
        for _, row in df.iterrows():
            matches = re.findall(r'(\d+)\s*([A-Z]{2,})', str(row['N seized specimens']))
            if matches:
                for qty, species in matches:
                    new_row = row.copy()
                    new_row['N_seized'] = float(qty)
                    new_row['Species'] = species
                    expanded_rows.append(new_row)
            else:
                expanded_rows.append(row)
        return pd.DataFrame(expanded_rows)

    df = expand_multi_species_rows(df)
    df = df.reset_index(drop=True)
    return df

# Compute crime score
def org_crime_score(df, binary_features, species_col='Species', year_col='Year',
                    count_col='N_seized', country_col='Country of offenders',
                    location_col='Location of seizure or shipment', weights=None):
    default_weights = {'trend': 0.25, 'chi2': 0.15, 'anomaly': 0.20, 'network': 0.30}
    if weights:
        default_weights.update({k: weights.get(k, v) for k, v in default_weights.items()})

    score = 0
    log = {}

    # Trend
    annual = df.groupby(year_col)[count_col].sum().reset_index()
    if len(annual) > 1:
        model = LinearRegression().fit(annual[[year_col]], annual[count_col])
        r2 = model.score(annual[[year_col]], annual[count_col])
        if r2 > 0.4:
            score += default_weights['trend']
            log['trend'] = f'+{default_weights["trend"]:.2f} (R² = {r2:.2f})'
        elif r2 < 0.05:
            score -= default_weights['trend']
            log['trend'] = f'-{default_weights["trend"]:.2f} (R² = {r2:.2f})'
        else:
            log['trend'] = f'0 (R² = {r2:.2f})'

    # Chi-squared
    if 'Species' in df.columns:
        contingency = pd.crosstab(df['Species'], df[year_col] > 2022)
        if contingency.shape == (2, 2):
            chi2, p, _, _ = chi2_contingency(contingency)
            if p < 0.05:
                score += default_weights['chi2']
                log['chi2'] = f'+{default_weights["chi2"]:.2f} (p = {p:.3f})'
            else:
                log['chi2'] = f'0 (p = {p:.3f})'

    # Anomaly detection
    if all(f in df.columns for f in binary_features):
        X = StandardScaler().fit_transform(df[binary_features])
        iforest = IsolationForest(random_state=42).fit_predict(X)
        lof = LocalOutlierFactor().fit_predict(X)
        dbscan = DBSCAN(eps=1.2, min_samples=2).fit_predict(X)
        outlier_votes = sum(pd.Series([iforest, lof, dbscan]).apply(lambda x: (np.array(x) == -1).sum()))
        ratio = outlier_votes / (len(df) * 3)
        if ratio > 0.15:
            score += default_weights['anomaly']
            log['anomalies'] = f'+{default_weights["anomaly"]:.2f} ({int(ratio*100)}% consensus)'
        else:
            log['anomalies'] = '0 (low outlier consensus)'

    # Network structure
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row['Case #'])

    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if i >= j:
                continue
            shared = sum([
                row1[year_col] == row2[year_col],
                row1[species_col] == row2[species_col],
                row1[country_col] == row2[country_col]
            ])
            if shared >= 2:
                G.add_edge(row1['Case #'], row2['Case #'])

    density = nx.density(G)
    components = nx.number_connected_components(G)
    if density > 0.2 and components < len(df) / 3:
        score += default_weights['network']
        log['network'] = f'+{default_weights["network"]:.2f} (density = {density:.2f}, {components} comps)'
    else:
        log['network'] = f'0 (density = {density:.2f}, {components} comps)'

    return max(-1.0, min(1.0, score)), log

# CUSUM helper
def compute_cusum(data, target_mean=None):
    if target_mean is None:
        target_mean = np.mean(data)
    pos, neg = [0], [0]
    for i in range(1, len(data)):
        diff = data[i] - target_mean
        pos.append(max(0, pos[-1] + diff))
        neg.append(min(0, neg[-1] + diff))
    return pos, neg

# Streamlit app interface
st.set_page_config(layout="wide")
st.title("Traffic-X: AI-Powered Counter Wildlife Trafficking Intelligence Suite")

st.markdown("""
**👋 First time here? Here's how to use Traffic-X**

Welcome to **Traffic-X**, your intelligent assistant for analyzing wildlife trafficking cases.

This app is designed to help **researchers, analysts, and wildlife protection agencies** study cases quickly and efficiently.

---

### ✅ What you can do:
- 📂 Upload your `.xlsx` dataset with trafficking cases  
- 📈 Analyze trends by species over time  
- 📊 Calculate an *organized crime score* using statistics and AI  
- 🕸️ Detect anomalies and explore networks between related cases  

---

### 📌 How to use:
1. Upload your Excel file using the uploader below or drag your file into the chat  
2. Select numerical features for anomaly detection (e.g., number of specimens, year)  
3. View interactive charts and insights — no programming needed!

---

Need an example dataset? Let us know!
""")


uploaded_file = st.file_uploader("Upload your trafficking case dataset (.xlsx)", type=["xlsx"])
if uploaded_file is not None:
    df = load_and_clean_data(uploaded_file)
    st.success("Data loaded and cleaned successfully!")
    st.dataframe(df)

    st.subheader("Organized Crime Score")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    selected = st.multiselect("Select features for anomaly detection", numeric_cols)
    if selected:
        score, log = org_crime_score(df, binary_features=selected)
        st.metric("Crime Score", f"{score:.2f}")
        st.json(log)

    st.subheader("Trend + CUSUM by Species")
    species_list = df["Species"].unique().tolist()
    selected_species = st.selectbox("Choose a species", species_list)
    if selected_species:
        subset = df[df["Species"] == selected_species]
        trend_data = subset.groupby("Year")["N_seized"].sum().reset_index()
        pos_cusum, neg_cusum = compute_cusum(trend_data["N_seized"].values)

        fig, ax = plt.subplots()
        ax.plot(trend_data["Year"], trend_data["N_seized"], marker='o', label="Trend")
        ax.plot(trend_data["Year"], pos_cusum, linestyle='--', label="CUSUM+")
        ax.plot(trend_data["Year"], neg_cusum, linestyle='--', label="CUSUM-")
        ax.set_title(f"{selected_species} - Trend & CUSUM")
        ax.set_xlabel("Year")
        ax.set_ylabel("Seized Specimens")
        ax.legend()
        st.pyplot(fig)

    st.subheader("📉 Regression Analysis")

    target_var = st.selectbox("Select the target variable (Y)", numeric_cols, key="target")
    predictors = st.multiselect("Select predictor variables (X)", [col for col in numeric_cols if col != target_var], key="predictors")

    if target_var and predictors:
        X = df[predictors]
        y = df[target_var]
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = model.score(X, y)

        st.write(f"**R² score:** {r2:.3f}")
        st.write("**Coefficients:**")
        for feature, coef in zip(predictors, model.coef_):
            st.write(f"- {feature}: {coef:.4f}")
        st.write(f"**Intercept:** {model.intercept_:.4f}")

        fig, ax = plt.subplots()
        ax.scatter(y, y_pred)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.set_title("Observed vs Predicted")
        st.pyplot(fig)


        
    st.subheader("🌐 Kernel Density Analysis")

    kde_var = st.selectbox("Select a numeric variable for KDE", numeric_cols, key="kde")
    group_var = st.selectbox("Group KDE by (optional)", df.columns, index=0, key="kde_group")

    if kde_var:
        fig, ax = plt.subplots()
        if group_var and group_var in df.columns and df[group_var].nunique() < 10:
            for label, subset in df.groupby(group_var):
                subset[kde_var].plot.kde(ax=ax, label=str(label))
        else:
            df[kde_var].plot.kde(ax=ax, label=kde_var)
        ax.set_title(f"Kernel Density Estimate for {kde_var}")
        ax.set_xlabel(kde_var)
        ax.legend()
        st.pyplot(fig)

        # Export download button
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.download_button("📥 Download KDE chart", data=buf.getvalue(), file_name="kde_plot.png", mime="image/png")

    st.markdown("---")
    st.subheader("🌐 Bivariate KDE (2D Density Plot)")

    x_var = st.selectbox("Select X variable", numeric_cols, key="kde_x")
    y_var = st.selectbox("Select Y variable", [col for col in numeric_cols if col != x_var], key="kde_y")

    if x_var and y_var:
        import seaborn as sns
        fig2, ax2 = plt.subplots()
        sns.kdeplot(data=df, x=x_var, y=y_var, fill=True, cmap="viridis", ax=ax2)
        ax2.set_title(f"Bivariate KDE: {x_var} vs {y_var}")
        st.pyplot(fig2)


else:
    st.info("Please upload a dataset to begin.")
