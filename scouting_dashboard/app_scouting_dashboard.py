
# =========================================================
# STREAMLIT DASHBOARD — SCOUTING NO FUTEBOL
# Dashboard-ready: Player DNA + Market + Clustering + Similarity + Recommendation
# =========================================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# =========================================================

st.set_page_config(
    page_title="Scouting Dashboard | Copa América 2024",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Scouting Dashboard — Copa América 2024")
st.caption(
    "Dashboard para análise de perfis Player DNA, similaridade técnica, mercado e recomendação de jogadores."
)


# =========================================================
# 2. LOAD DATA
# =========================================================

@st.cache_data
def load_dashboard_data(uploaded_file=None, default_path="outputs/dashboard_scouting_data.xlsx"):
    """
    Carrega o ficheiro exportado pelo notebook.
    Espera uma folha chamada Dashboard_Data; se não existir, lê a primeira folha.
    """

    if uploaded_file is not None:
        try:
            return pd.read_excel(uploaded_file, sheet_name="Dashboard_Data")
        except Exception:
            return pd.read_excel(uploaded_file)

    candidate_paths = [
        Path(default_path),
        Path("dashboard_scouting_data.xlsx"),
        Path("outputs/dashboard_scouting_data.csv"),
        Path("dashboard_scouting_data.csv"),
    ]

    for path in candidate_paths:
        if path.exists():
            if path.suffix.lower() == ".csv":
                return pd.read_csv(path)
            try:
                return pd.read_excel(path, sheet_name="Dashboard_Data")
            except Exception:
                return pd.read_excel(path)

    return None


with st.sidebar:
    st.header("1. Dados")
    uploaded_file = st.file_uploader(
        "Carregar ficheiro exportado do notebook",
        type=["xlsx", "csv"]
    )

data = load_dashboard_data(uploaded_file=uploaded_file)

if data is None:
    st.warning(
        "Ainda não foi encontrado o ficheiro de dados. "
        "Exporta `outputs/dashboard_scouting_data.xlsx` no notebook ou carrega o ficheiro na sidebar."
    )
    st.stop()

df = data.copy()


# =========================================================
# 3. CONFIGURAÇÕES E NORMALIZAÇÃO
# =========================================================

PROFILE_CONFIG = {
    "extremo_1x1": {
        "label": "Extremo 1x1",
        "role_family": "Winger",
        "fit_col": "extremo_1x1_fit_score",
        "dna_col": "dna_extremo_1x1",
    },
    "extremo_finalizador": {
        "label": "Extremo Finalizador",
        "role_family": "Winger",
        "fit_col": "extremo_finalizador_fit_score",
        "dna_col": "dna_extremo_finalizador",
    },
    "extremo_completo": {
        "label": "Extremo Completo",
        "role_family": "Winger",
        "fit_col": "extremo_completo_fit_score",
        "dna_col": "dna_extremo_completo",
    },
    "medio_ofensivo_criativo": {
        "label": "Médio Ofensivo Criativo",
        "role_family": "Attacking Midfielder",
        "fit_col": "medio_ofensivo_criativo_fit_score",
        "dna_col": "dna_medio_ofensivo_criativo",
    },
    "lateral_ofensivo": {
        "label": "Lateral Ofensivo",
        "role_family": "Fullback",
        "fit_col": "lateral_ofensivo_fit_score",
        "dna_col": "dna_lateral_ofensivo",
    },
}

TECHNICAL_FEATURES = [
    "one_v_one_score",
    "progression_score",
    "xA_quality_score",
    "xg_quality_score",
    "gplus_proxy_score",
    "cross_quality_score",
]

FEATURE_LABELS = {
    "one_v_one_score": "1x1",
    "progression_score": "Progressão",
    "xA_quality_score": "Criação / xA",
    "xg_quality_score": "Finalização / xG",
    "gplus_proxy_score": "Impacto Global",
    "cross_quality_score": "Cruzamento",
}

# Coluna oficial de minutos
if "minutes_copa_america_2024" in df.columns:
    OFFICIAL_MINUTES_COL = "minutes_copa_america_2024"
elif "minutes_played" in df.columns:
    OFFICIAL_MINUTES_COL = "minutes_played"
else:
    OFFICIAL_MINUTES_COL = None

if OFFICIAL_MINUTES_COL is None:
    st.error("O dataset não tem coluna de minutos: esperado `minutes_copa_america_2024` ou `minutes_played`.")
    st.stop()

# Tipos numéricos
numeric_cols = [
    OFFICIAL_MINUTES_COL,
    "age",
    "market_value_eur_2024",
    "market_opportunity_score",
    "cluster_id",
] + TECHNICAL_FEATURES

for cfg in PROFILE_CONFIG.values():
    numeric_cols.extend([cfg["fit_col"], cfg["dna_col"]])

for col in set(numeric_cols):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Garantir fit scores caso só existam DNA scores
for key, cfg in PROFILE_CONFIG.items():
    fit_col = cfg["fit_col"]
    dna_col = cfg["dna_col"]

    if fit_col not in df.columns and dna_col in df.columns:
        df[fit_col] = df[dna_col]

if "market_opportunity_score" not in df.columns:
    df["market_opportunity_score"] = 0.5

df["market_opportunity_score"] = pd.to_numeric(
    df["market_opportunity_score"],
    errors="coerce"
).fillna(0.5)

# Garantir cluster columns
if "cluster_id" not in df.columns:
    df["cluster_id"] = np.nan

if "cluster_label" not in df.columns:
    df["cluster_label"] = "Sem cluster"

if "cluster_description" not in df.columns:
    df["cluster_description"] = df["cluster_label"]


# =========================================================
# 4. FUNÇÕES ANALÍTICAS
# =========================================================

def safe_player_label(row):
    team = row.get("team_name", "")
    pos = row.get("position", "")
    return f"{row['player_name']} | {pos} | {team}"


def compute_cluster_fit_score(data, fit_col):
    data = data.copy()

    # Se não existir cluster_id, usa valor neutro
    if "cluster_id" not in data.columns:
        data["cluster_fit_score"] = 0.5
        return data

    # Se não existir fit_col, usa valor neutro
    if fit_col not in data.columns:
        data["cluster_fit_score"] = 0.5
        return data

    # Garantir numéricos
    data[fit_col] = pd.to_numeric(
        data[fit_col],
        errors="coerce"
    ).fillna(0)

    # Calcular força média do cluster
    cluster_strength = (
        data
        .dropna(subset=["cluster_id"])
        .groupby(["role_family", "cluster_id"], as_index=False)
        .agg(
            cluster_avg_fit=(fit_col, "mean"),
            cluster_n_players=("player_name", "count")
        )
    )

    if len(cluster_strength) == 0:
        data["cluster_fit_score"] = 0.5
        return data

    cluster_strength["cluster_fit_score"] = 0.5

    for role in cluster_strength["role_family"].dropna().unique():
        mask = cluster_strength["role_family"] == role
        values = cluster_strength.loc[mask, "cluster_avg_fit"]

        min_value = values.min()
        max_value = values.max()

        if max_value > min_value:
            cluster_strength.loc[mask, "cluster_fit_score"] = (
                (values - min_value) / (max_value - min_value)
            )
        else:
            cluster_strength.loc[mask, "cluster_fit_score"] = 0.5

    data = data.merge(
        cluster_strength[
            [
                "role_family",
                "cluster_id",
                "cluster_avg_fit",
                "cluster_n_players",
                "cluster_fit_score"
            ]
        ],
        on=["role_family", "cluster_id"],
        how="left"
    )

    data["cluster_fit_score"] = data["cluster_fit_score"].fillna(0.5)

    return data


def compute_similar_players(
    role_df,
    target_player_id,
    features,
    fit_col,
    same_cluster_only=False,
    same_cluster_bonus=True,
    exclude_same_team=False,
    min_minutes=0,
    max_age=None,
    max_market_value=None,
    top_n=15,
    weights=None,
):
    """
    Similaridade técnica:
    - base por role_family;
    - MinMaxScaler;
    - cosine similarity;
    - mercado aplicado depois como filtro;
    - cluster como filtro ou bónus opcional.
    """

    data = role_df.copy()

    if len(data) < 3:
        return pd.DataFrame()

    data["statsbomb_player_id"] = pd.to_numeric(data["statsbomb_player_id"], errors="coerce")
    data = data.dropna(subset=["statsbomb_player_id"]).copy()
    data["statsbomb_player_id"] = data["statsbomb_player_id"].astype(int)

    target_matches = data[data["statsbomb_player_id"] == int(target_player_id)]

    if len(target_matches) == 0:
        return pd.DataFrame()

    target_row = target_matches.iloc[0]
    target_cluster = target_row.get("cluster_id", np.nan)
    target_team = target_row.get("team_name", None)

    available_features = [f for f in features if f in data.columns]

    if len(available_features) == 0:
        return pd.DataFrame()

    X = data[available_features].copy()

    for col in available_features:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    sim_matrix = cosine_similarity(X_scaled)

    target_idx = target_matches.index[0]
    # role_df may not have continuous index, so map to positional index
    positional_idx = data.index.get_loc(target_idx)

    data["similarity_score"] = sim_matrix[positional_idx]

    output = data[data["statsbomb_player_id"] != int(target_player_id)].copy()

    # filtros pós-similaridade
    output = output[output[OFFICIAL_MINUTES_COL] >= min_minutes].copy()

    if max_age is not None and "age" in output.columns:
        output = output[
            output["age"].isna() | (output["age"] <= max_age)
        ].copy()

    if max_market_value is not None and "market_value_eur_2024" in output.columns:
        output = output[
            output["market_value_eur_2024"].isna() |
            (output["market_value_eur_2024"] <= max_market_value)
        ].copy()

    if exclude_same_team and "team_name" in output.columns and target_team is not None:
        output = output[output["team_name"] != target_team].copy()

    # cluster como filtro ou bónus
    if "cluster_id" in output.columns and pd.notna(target_cluster):
        output["same_cluster"] = output["cluster_id"] == target_cluster

        if same_cluster_only:
            output = output[output["same_cluster"]].copy()

        if same_cluster_bonus:
            output["similarity_score_adjusted"] = np.where(
                output["same_cluster"],
                output["similarity_score"] * 1.05,
                output["similarity_score"]
            )
        else:
            output["similarity_score_adjusted"] = output["similarity_score"]
    else:
        output["same_cluster"] = False
        output["similarity_score_adjusted"] = output["similarity_score"]

    output["similarity_score_adjusted"] = output["similarity_score_adjusted"].clip(upper=1)

    # Cluster fit
    output = compute_cluster_fit_score(output, fit_col)

    # Pesos
    if weights is None:
        weights = {
            "similarity": 0.40,
            "technical": 0.35,
            "market": 0.15,
            "cluster": 0.10,
        }

    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    output[fit_col] = pd.to_numeric(output[fit_col], errors="coerce").fillna(0)
    output["market_opportunity_score"] = pd.to_numeric(
        output["market_opportunity_score"],
        errors="coerce"
    ).fillna(0.5)
    output["cluster_fit_score"] = pd.to_numeric(
        output["cluster_fit_score"],
        errors="coerce"
    ).fillna(0.5)

    output["recommendation_score"] = (
        output["similarity_score_adjusted"] * weights["similarity"] +
        output[fit_col] * weights["technical"] +
        output["market_opportunity_score"] * weights["market"] +
        output["cluster_fit_score"] * weights["cluster"]
    ).round(4)

    output["similarity_score"] = output["similarity_score"].round(4)
    output["similarity_score_adjusted"] = output["similarity_score_adjusted"].round(4)

    return (
        output
        .sort_values("recommendation_score", ascending=False)
        .head(top_n)
        .copy()
    )


def radar_chart(players_df, player_names, features):
    """
    Radar comparativo para jogadores selecionados.
    """

    available_features = [f for f in features if f in players_df.columns]

    fig = go.Figure()

    labels = [FEATURE_LABELS.get(f, f) for f in available_features]

    for player_name in player_names:
        row_df = players_df[players_df["player_name"] == player_name]

        if len(row_df) == 0:
            continue

        row = row_df.iloc[0]
        values = [
            float(row[f]) if pd.notna(row[f]) else 0
            for f in available_features
        ]

        # fechar o polígono
        values_closed = values + [values[0]]
        labels_closed = labels + [labels[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=labels_closed,
                fill="toself",
                name=player_name
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        height=520,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


def format_eur(value):
    if pd.isna(value):
        return "n.d."
    if value >= 1_000_000:
        return f"€{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"€{value / 1_000:.0f}k"
    return f"€{value:.0f}"


# =========================================================
# 5. SIDEBAR
# =========================================================

with st.sidebar:
    st.header("2. Filtros de Scouting")

    profile_labels = {cfg["label"]: key for key, cfg in PROFILE_CONFIG.items()}

    selected_profile_label = st.selectbox(
        "Perfil Player DNA",
        list(profile_labels.keys()),
        index=2
    )

    selected_profile = profile_labels[selected_profile_label]
    profile_cfg = PROFILE_CONFIG[selected_profile]

    expected_role = profile_cfg["role_family"]
    fit_col = profile_cfg["fit_col"]

    if fit_col not in df.columns:
        st.error(f"A coluna `{fit_col}` não existe no dataset.")
        st.stop()

    role_df = df[df["role_family"] == expected_role].copy()

    min_minutes = int(st.slider(
        "Minutos mínimos",
        min_value=0,
        max_value=int(max(df[OFFICIAL_MINUTES_COL].max(), 90)),
        value=min(180, int(max(df[OFFICIAL_MINUTES_COL].max(), 90))),
        step=30
    ))

    if "age" in role_df.columns and role_df["age"].notna().any():
        min_age = int(np.nanmin(role_df["age"]))
        max_age_data = int(np.nanmax(role_df["age"]))
        max_age = st.slider(
            "Idade máxima",
            min_value=max(15, min_age),
            max_value=max(16, max_age_data),
            value=min(26, max_age_data),
            step=1
        )
    else:
        max_age = None

    if "market_value_eur_2024" in role_df.columns and role_df["market_value_eur_2024"].notna().any():
        max_mv_data = float(np.nanmax(role_df["market_value_eur_2024"]))
        max_market_value_m = st.slider(
            "Valor de mercado máximo (€M)",
            min_value=0.0,
            max_value=max(1.0, round(max_mv_data / 1_000_000, 1)),
            value=min(35.0, max(1.0, round(max_mv_data / 1_000_000, 1))),
            step=0.5
        )
        max_market_value = max_market_value_m * 1_000_000
    else:
        max_market_value = None

    same_cluster_only = st.checkbox(
        "Apenas mesmo cluster",
        value=False
    )

    same_cluster_bonus = st.checkbox(
        "Bónus ao mesmo cluster",
        value=True
    )

    exclude_same_team = st.checkbox(
        "Excluir jogadores da mesma seleção/equipa",
        value=False
    )

    top_n = st.slider(
        "Número de recomendações",
        min_value=5,
        max_value=30,
        value=15,
        step=5
    )


# =========================================================
# 6. FILTRAR BASE DO PERFIL
# =========================================================

role_df = role_df[role_df[OFFICIAL_MINUTES_COL] >= min_minutes].copy()

if max_age is not None and "age" in role_df.columns:
    role_df = role_df[
        role_df["age"].isna() | (role_df["age"] <= max_age)
    ].copy()

if max_market_value is not None and "market_value_eur_2024" in role_df.columns:
    role_df = role_df[
        role_df["market_value_eur_2024"].isna() |
        (role_df["market_value_eur_2024"] <= max_market_value)
    ].copy()

role_df[fit_col] = pd.to_numeric(role_df[fit_col], errors="coerce").fillna(0)
role_df["market_opportunity_score"] = pd.to_numeric(
    role_df["market_opportunity_score"],
    errors="coerce"
).fillna(0.5)

role_df = compute_cluster_fit_score(role_df, fit_col)

if len(role_df) == 0:
    st.warning("Não existem jogadores para os filtros selecionados.")
    st.stop()

# Jogador modelo
model_options_df = (
    role_df
    .sort_values(fit_col, ascending=False)
    .copy()
)

model_options_df["player_option_label"] = model_options_df.apply(
    safe_player_label,
    axis=1
)

with st.sidebar:
    selected_model_label = st.selectbox(
        "Jogador modelo",
        model_options_df["player_option_label"].tolist()
    )

target_player = model_options_df[
    model_options_df["player_option_label"] == selected_model_label
].iloc[0]

target_player_id = int(target_player["statsbomb_player_id"])
target_player_name = target_player["player_name"]


# =========================================================
# 7. RECOMENDAÇÕES
# =========================================================

recommendations = compute_similar_players(
    role_df=role_df,
    target_player_id=target_player_id,
    features=TECHNICAL_FEATURES,
    fit_col=fit_col,
    same_cluster_only=same_cluster_only,
    same_cluster_bonus=same_cluster_bonus,
    exclude_same_team=exclude_same_team,
    min_minutes=min_minutes,
    max_age=max_age,
    max_market_value=max_market_value,
    top_n=top_n,
)

# Ranking geral por perfil, sem jogador modelo
profile_ranking = (
    role_df
    .sort_values(fit_col, ascending=False)
    .head(top_n)
    .copy()
)


# =========================================================
# 8. KPIS
# =========================================================

st.subheader(f"Perfil selecionado: {selected_profile_label}")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric("Jogadores elegíveis", len(role_df))
kpi2.metric("Jogador modelo", target_player_name)
kpi3.metric("Idade média", f"{role_df['age'].mean():.1f}" if "age" in role_df else "n.d.")
kpi4.metric(
    "Valor mediano",
    format_eur(role_df["market_value_eur_2024"].median())
    if "market_value_eur_2024" in role_df else "n.d."
)
kpi5.metric("Fit médio", f"{role_df[fit_col].mean():.3f}")


# =========================================================
# 9. LAYOUT PRINCIPAL
# =========================================================

tab_overview, tab_similarity, tab_radar, tab_market, tab_table = st.tabs(
    [
        "📌 Overview",
        "🧬 Similaridade",
        "📡 Radar",
        "💰 Mercado",
        "📋 Tabela"
    ]
)


with tab_overview:
    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.markdown("### Top jogadores por Fit Score")

        fig_fit = px.bar(
            profile_ranking.sort_values(fit_col, ascending=True),
            x=fit_col,
            y="player_name",
            orientation="h",
            hover_data=[
                col for col in [
                    "position",
                    "team_name",
                    "age",
                    "market_value_eur_2024",
                    OFFICIAL_MINUTES_COL,
                    "cluster_label"
                ]
                if col in profile_ranking.columns
            ],
            title=f"Top {top_n} — {selected_profile_label}"
        )

        fig_fit.update_layout(height=520, yaxis_title="", xaxis_title="Fit Score")
        st.plotly_chart(fig_fit, use_container_width=True)

    with col2:
        st.markdown("### Perfil do jogador modelo")

        target_display = target_player.to_dict()

        st.write(f"**Jogador:** {target_display.get('player_name', 'n.d.')}")
        st.write(f"**Posição:** {target_display.get('position', 'n.d.')}")
        st.write(f"**Role family:** {target_display.get('role_family', 'n.d.')}")
        st.write(f"**Equipa/Seleção:** {target_display.get('team_name', 'n.d.')}")
        st.write(f"**Clube:** {target_display.get('club', 'n.d.')}")
        st.write(f"**Liga:** {target_display.get('league', 'n.d.')}")
        st.write(f"**Idade:** {target_display.get('age', 'n.d.')}")
        st.write(f"**Valor de mercado:** {format_eur(target_display.get('market_value_eur_2024', np.nan))}")
        st.write(f"**Minutos:** {target_display.get(OFFICIAL_MINUTES_COL, 'n.d.')}")
        st.write(f"**Cluster:** {target_display.get('cluster_label', 'n.d.')}")
        st.write(f"**Fit Score:** {target_display.get(fit_col, 0):.3f}")


with tab_similarity:
    st.markdown("### Jogadores mais semelhantes ao jogador modelo")

    if len(recommendations) == 0:
        st.warning("Não foram encontrados jogadores similares com os filtros atuais.")
    else:
        fig_rec = px.bar(
            recommendations.sort_values("recommendation_score", ascending=True),
            x="recommendation_score",
            y="player_name",
            orientation="h",
            hover_data=[
                col for col in [
                    "similarity_score_adjusted",
                    fit_col,
                    "market_opportunity_score",
                    "cluster_fit_score",
                    "same_cluster",
                    "age",
                    "market_value_eur_2024",
                    OFFICIAL_MINUTES_COL
                ]
                if col in recommendations.columns
            ],
            title="Ranking de recomendação"
        )

        fig_rec.update_layout(height=560, yaxis_title="", xaxis_title="Recommendation Score")
        st.plotly_chart(fig_rec, use_container_width=True)

        scatter_x = "similarity_score_adjusted"
        scatter_y = fit_col

        fig_scatter = px.scatter(
            recommendations,
            x=scatter_x,
            y=scatter_y,
            size="market_value_eur_2024" if "market_value_eur_2024" in recommendations else None,
            color="recommendation_score",
            hover_name="player_name",
            hover_data=[
                col for col in [
                    "position",
                    "team_name",
                    "age",
                    "market_value_eur_2024",
                    "cluster_label",
                    "same_cluster"
                ]
                if col in recommendations.columns
            ],
            title="Similaridade técnica vs Fit Score"
        )

        fig_scatter.update_layout(height=520)
        st.plotly_chart(fig_scatter, use_container_width=True)


with tab_radar:
    st.markdown("### Radar comparativo")

    radar_candidates = [target_player_name]

    if len(recommendations) > 0:
        radar_candidates += recommendations["player_name"].head(4).tolist()

    selected_radar_players = st.multiselect(
        "Selecionar jogadores para comparar no radar",
        options=radar_candidates,
        default=radar_candidates[: min(3, len(radar_candidates))]
    )

    radar_base = pd.concat(
        [
            role_df[role_df["statsbomb_player_id"] == target_player_id],
            recommendations
        ],
        ignore_index=True
    ).drop_duplicates(subset=["statsbomb_player_id"])

    if len(selected_radar_players) > 0:
        fig_radar = radar_chart(
            players_df=radar_base,
            player_names=selected_radar_players,
            features=TECHNICAL_FEATURES
        )

        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("### Métricas técnicas")
    technical_cols = [
        "player_name",
        "position",
        OFFICIAL_MINUTES_COL,
    ] + [f for f in TECHNICAL_FEATURES if f in radar_base.columns]

    st.dataframe(
        radar_base[technical_cols].sort_values("player_name"),
        use_container_width=True
    )


with tab_market:
    st.markdown("### Mapa idade × valor de mercado")

    market_plot_df = role_df.copy()

    if "age" in market_plot_df.columns and "market_value_eur_2024" in market_plot_df.columns:
        fig_market = px.scatter(
            market_plot_df,
            x="age",
            y="market_value_eur_2024",
            color=fit_col,
            size=OFFICIAL_MINUTES_COL,
            hover_name="player_name",
            hover_data=[
                col for col in [
                    "position",
                    "team_name",
                    "club",
                    "league",
                    "market_opportunity_score",
                    "cluster_label"
                ]
                if col in market_plot_df.columns
            ],
            title="Idade vs Valor de mercado"
        )

        fig_market.update_layout(
            height=560,
            xaxis_title="Idade",
            yaxis_title="Valor de mercado (€)"
        )

        st.plotly_chart(fig_market, use_container_width=True)
    else:
        st.info("Não existem colunas suficientes para o gráfico de mercado.")

    st.markdown("### Top oportunidades de mercado")

    market_rank_cols = [
        "player_name",
        "position",
        "team_name",
        "age",
        "market_value_eur_2024",
        "market_opportunity_score",
        fit_col,
        OFFICIAL_MINUTES_COL,
        "cluster_label",
    ]

    market_rank_cols = [c for c in market_rank_cols if c in role_df.columns]

    st.dataframe(
        role_df[market_rank_cols]
        .sort_values(["market_opportunity_score", fit_col], ascending=False)
        .head(20),
        use_container_width=True
    )


with tab_table:
    st.markdown("### Tabela final de recomendações")

    if len(recommendations) == 0:
        st.warning("Sem recomendações para apresentar.")
    else:
        output_cols = [
            "player_name",
            "statsbomb_player_id",
            "team_name",
            "position",
            "role_family",
            OFFICIAL_MINUTES_COL,
            "age",
            "club",
            "league",
            "market_value_eur_2024",
            fit_col,
            "similarity_score",
            "similarity_score_adjusted",
            "same_cluster",
            "market_opportunity_score",
            "cluster_fit_score",
            "recommendation_score",
            "cluster_label",
        ]

        output_cols = [c for c in output_cols if c in recommendations.columns]

        st.dataframe(
            recommendations[output_cols],
            use_container_width=True
        )

        csv = recommendations[output_cols].to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Descarregar recomendações em CSV",
            data=csv,
            file_name="recommendations_dashboard.csv",
            mime="text/csv"
        )
