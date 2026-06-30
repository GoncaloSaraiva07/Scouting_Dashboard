
# =========================================================
# STREAMLIT DASHBOARD — SCOUTING NO FUTEBOL
# Dashboard-ready: Player DNA + Market + Clustering + Similarity + Recommendation
# =========================================================

import numpy as npF
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

@st.cache_data
def load_spatial_events(uploaded_file=None, default_path="outputs/dashboard_spatial_events.csv"):
    """
    Carrega eventos espaciais exportados do notebook.
    Espera colunas:
    statsbomb_player_id, player_name, event_type, x, y
    """

    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    candidate_paths = [
        Path(default_path),
        Path("dashboard_spatial_events.csv"),
        Path("outputs/dashboard_spatial_events.csv"),
    ]

    for path in candidate_paths:
        if path.exists():
            return pd.read_csv(path)

    return None

with st.sidebar:
    st.header("1. Dados")

    uploaded_file = st.file_uploader(
        "Carregar ficheiro agregado do notebook",
        type=["xlsx", "csv"]
    )

    uploaded_spatial_file = st.file_uploader(
        "Carregar eventos espaciais StatsBomb",
        type=["csv"]
    )

data = load_dashboard_data(uploaded_file=uploaded_file)

spatial_events = load_spatial_events(
    uploaded_file=uploaded_spatial_file
)

if data is None:
    st.warning(
        "Ainda não foi encontrado o ficheiro de dados. "
        "Exporta `outputs/dashboard_scouting_data.xlsx` no notebook ou carrega o ficheiro na sidebar."
    )
    st.stop()

df = data.copy()

# Carregar eventos espaciais do Notebook

@st.cache_data
def load_spatial_events(uploaded_file=None, default_path="outputs/dashboard_spatial_events.csv"):
    """
    Carrega eventos espaciais exportados do notebook.
    Espera colunas:
    statsbomb_player_id, player_name, event_type, x, y
    """

    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    candidate_paths = [
        Path(default_path),
        Path("dashboard_spatial_events.csv"),
        Path("outputs/dashboard_spatial_events.csv"),
    ]

    for path in candidate_paths:
        if path.exists():
            return pd.read_csv(path)

    return None

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

    # Se a coluna já existir, apenas garantir que não tem NaN
    if "cluster_fit_score" in data.columns:
        data["cluster_fit_score"] = pd.to_numeric(
            data["cluster_fit_score"],
            errors="coerce"
        ).fillna(0.5)
        return data

    # Se não existir cluster_id, usa valor neutro
    if "cluster_id" not in data.columns:
        data["cluster_fit_score"] = 0.5
        return data

    # Se não existir role_family, usa valor neutro
    if "role_family" not in data.columns:
        data["cluster_fit_score"] = 0.5
        return data

    # Se não existir a coluna de fit score, usa valor neutro
    if fit_col not in data.columns:
        data["cluster_fit_score"] = 0.5
        return data

    # Garantir que fit_col é numérica
    data[fit_col] = pd.to_numeric(
        data[fit_col],
        errors="coerce"
    ).fillna(0)

    # Calcular força média de cada cluster para o perfil escolhido
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

    # Normalizar a força do cluster dentro de cada role_family
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

    # Juntar score ao dataframe original
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

    data["cluster_fit_score"] = pd.to_numeric(
        data["cluster_fit_score"],
        errors="coerce"
    ).fillna(0.5)

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

def draw_pitch_layout(fig):
    """
    Desenha linhas básicas de um campo StatsBomb 120x80 em Plotly.
    """

    line_color = "rgba(40, 40, 40, 0.75)"

    shapes = []

    # Campo
    shapes.append(dict(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color=line_color, width=2)))

    # Linha do meio
    shapes.append(dict(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color=line_color, width=1)))

    # Grandes áreas
    shapes.append(dict(type="rect", x0=0, y0=18, x1=18, y1=62, line=dict(color=line_color, width=1)))
    shapes.append(dict(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color=line_color, width=1)))

    # Pequenas áreas
    shapes.append(dict(type="rect", x0=0, y0=30, x1=6, y1=50, line=dict(color=line_color, width=1)))
    shapes.append(dict(type="rect", x0=114, y0=30, x1=120, y1=50, line=dict(color=line_color, width=1)))

    # Círculo central
    shapes.append(dict(type="circle", x0=50, y0=30, x1=70, y1=50, line=dict(color=line_color, width=1)))

    fig.update_layout(shapes=shapes)

    fig.update_xaxes(
        range=[0, 120],
        showgrid=False,
        zeroline=False,
        showticklabels=False
    )

    fig.update_yaxes(
        range=[80, 0],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        scaleanchor="x",
        scaleratio=1
    )

    return fig


def plot_player_heatmap(spatial_events, player_id, player_name, event_group="Todas as ações"):
    """
    Cria mapa de manchas / heatmap do jogador modelo.
    Usa eventos StatsBomb com coordenadas x,y.
    """

    if spatial_events is None or len(spatial_events) == 0:
        return None, pd.DataFrame()

    data = spatial_events.copy()

    required_cols = [
        "statsbomb_player_id",
        "event_type",
        "x",
        "y"
    ]

    missing_cols = [
        col for col in required_cols
        if col not in data.columns
    ]

    if len(missing_cols) > 0:
        return None, pd.DataFrame()

    data["statsbomb_player_id"] = pd.to_numeric(
        data["statsbomb_player_id"],
        errors="coerce"
    )

    data["x"] = pd.to_numeric(data["x"], errors="coerce")
    data["y"] = pd.to_numeric(data["y"], errors="coerce")

    data = data.dropna(
        subset=["statsbomb_player_id", "x", "y"]
    ).copy()

    data["statsbomb_player_id"] = data["statsbomb_player_id"].astype(int)

    player_events = data[
        data["statsbomb_player_id"] == int(player_id)
    ].copy()

    event_groups = {
        "Todas as ações": None,
        "Passes": ["Pass"],
        "Conduções": ["Carry"],
        "Dribles": ["Dribble"],
        "Remates": ["Shot"],
        "Ações ofensivas": ["Pass", "Carry", "Dribble", "Shot", "Ball Receipt*"]
    }

    selected_events = event_groups.get(event_group)

    if selected_events is not None:
        player_events = player_events[
            player_events["event_type"].isin(selected_events)
        ].copy()

    if len(player_events) == 0:
        return None, player_events

    fig = go.Figure()

    fig.add_trace(
        go.Histogram2dContour(
            x=player_events["x"],
            y=player_events["y"],
            colorscale="Viridis",
            contours=dict(
                coloring="heatmap",
                showlabels=False
            ),
            opacity=0.80,
            ncontours=18,
            showscale=True,
            colorbar=dict(
                title="Densidade"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=player_events["x"],
            y=player_events["y"],
            mode="markers",
            marker=dict(
                size=4,
                color="white",
                opacity=0.35,
                line=dict(width=0)
            ),
            name="Ações"
        )
    )

    fig = draw_pitch_layout(fig)

    fig.update_layout(
        title=f"Mapa de manchas — {player_name} | {event_group}",
        height=560,
        plot_bgcolor="rgba(245, 247, 250, 1)",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False
    )

    return fig, player_events

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

# ---------------------------------------------------------
# Filtro de idade — range slider
# ---------------------------------------------------------

min_age = None
max_age = None

if "age" in role_df.columns:

    role_df["age"] = pd.to_numeric(
        role_df["age"],
        errors="coerce"
    )

    valid_ages = role_df["age"].dropna()

    if len(valid_ages) > 0:

        min_age_data = int(valid_ages.min())
        max_age_data = int(valid_ages.max())

        if min_age_data == max_age_data:
            min_age = min_age_data
            max_age = max_age_data

            st.info(
                f"Apenas existe uma idade disponível para este perfil: {min_age_data} anos."
            )

        else:
            default_min_age = min_age_data
            default_max_age = min(26, max_age_data)

            if default_max_age < default_min_age:
                default_max_age = max_age_data

            age_range = st.slider(
                "Intervalo de idade",
                min_value=min_age_data,
                max_value=max_age_data,
                value=(default_min_age, default_max_age),
                step=1
            )

            min_age = age_range[0]
            max_age = age_range[1]

    else:
        st.info(
            "Filtro de idade indisponível: não existem idades válidas para este perfil."
        )


# ---------------------------------------------------------
# Filtro de valor de mercado — range slider
# ---------------------------------------------------------

min_market_value = None
max_market_value = None

if "market_value_eur_2024" in role_df.columns:

    role_df["market_value_eur_2024"] = pd.to_numeric(
        role_df["market_value_eur_2024"],
        errors="coerce"
    )

    valid_market_values = role_df["market_value_eur_2024"].dropna()

    if len(valid_market_values) > 0:

        min_mv_m = round(float(valid_market_values.min()) / 1_000_000, 1)
        max_mv_m = round(float(valid_market_values.max()) / 1_000_000, 1)

        if min_mv_m == max_mv_m:
            min_market_value = min_mv_m * 1_000_000
            max_market_value = max_mv_m * 1_000_000

            st.info(
                f"Apenas existe um valor de mercado disponível para este perfil: €{max_mv_m:.1f}M."
            )

        else:
            default_min_mv_m = min_mv_m
            default_max_mv_m = min(35.0, max_mv_m)

            if default_max_mv_m < default_min_mv_m:
                default_max_mv_m = max_mv_m

            market_value_range_m = st.slider(
                "Intervalo de valor de mercado (€M)",
                min_value=min_mv_m,
                max_value=max_mv_m,
                value=(default_min_mv_m, default_max_mv_m),
                step=0.5
            )

            min_market_value = market_value_range_m[0] * 1_000_000
            max_market_value = market_value_range_m[1] * 1_000_000

    else:
        st.info(
            "Filtro de valor de mercado indisponível: não existem valores válidos para este perfil."
        )

# =========================================================
# 6. FILTRAR BASE DO PERFIL
# =========================================================

role_df = role_df[role_df[OFFICIAL_MINUTES_COL] >= min_minutes].copy()

# ---------------------------------------------------------
# Aplicar filtro de idade
# ---------------------------------------------------------

if min_age is not None and max_age is not None and "age" in role_df.columns:

    role_df["age"] = pd.to_numeric(
        role_df["age"],
        errors="coerce"
    )

    role_df = role_df[
        role_df["age"].between(min_age, max_age)
    ].copy()

# ---------------------------------------------------------
# Aplicar filtro de valor de mercado
# ---------------------------------------------------------

if (
    min_market_value is not None and
    max_market_value is not None and
    "market_value_eur_2024" in role_df.columns
):

    role_df["market_value_eur_2024"] = pd.to_numeric(
        role_df["market_value_eur_2024"],
        errors="coerce"
    )

    role_df = role_df[
        role_df["market_value_eur_2024"].between(
            min_market_value,
            max_market_value
        )
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

# =========================================================
# JOGADOR MODELO EM DESTAQUE
# =========================================================

model_age_value = target_player.get("age", np.nan)

if pd.isna(model_age_value):
    model_age_text = "n.d."
else:
    model_age_text = f"{int(model_age_value)} anos"


model_market_text = format_eur(
    target_player.get("market_value_eur_2024", np.nan)
)


model_minutes_value = target_player.get(OFFICIAL_MINUTES_COL, np.nan)

if pd.isna(model_minutes_value):
    model_minutes_text = "n.d."
else:
    model_minutes_text = f"{int(model_minutes_value)} min"


model_fit_value = pd.to_numeric(
    pd.Series([target_player.get(fit_col, np.nan)]),
    errors="coerce"
).iloc[0]

if pd.isna(model_fit_value):
    model_fit_text = "n.d."
else:
    model_fit_text = f"{model_fit_value:.3f}"


model_cluster_text = target_player.get("cluster_label", "n.d.")

if pd.isna(model_cluster_text):
    model_cluster_text = "n.d."


st.markdown(
    """
    <style>
    .model-player-card {
        background: linear-gradient(135deg, #101828 0%, #1D2939 55%, #344054 100%);
        color: white;
        border-radius: 18px;
        padding: 24px 28px;
        margin-top: 12px;
        margin-bottom: 24px;
        box-shadow: 0 10px 28px rgba(16, 24, 40, 0.18);
    }

    .model-player-title {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 4px;
        letter-spacing: 0.4px;
    }

    .model-player-subtitle {
        font-size: 15px;
        color: #D0D5DD;
        margin-bottom: 18px;
    }

    .model-player-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 14px;
    }

    .model-player-metric {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 12px 14px;
    }

    .model-player-label {
        font-size: 12px;
        color: #D0D5DD;
        margin-bottom: 5px;
    }

    .model-player-value {
        font-size: 18px;
        font-weight: 700;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    f"""
    <div class="model-player-card">
        <div class="model-player-title">🎯 Jogador modelo: {target_player_name}</div>
        <div class="model-player-subtitle">
            {target_player.get("position", "n.d.")} · {target_player.get("team_name", "n.d.")} · Perfil: {selected_profile_label}
        </div>

        <div class="model-player-grid">
            <div class="model-player-metric">
                <div class="model-player-label">Idade</div>
                <div class="model-player-value">{model_age_text}</div>
            </div>

            <div class="model-player-metric">
                <div class="model-player-label">Valor de mercado</div>
                <div class="model-player-value">{model_market_text}</div>
            </div>

            <div class="model-player-metric">
                <div class="model-player-label">Minutos</div>
                <div class="model-player-value">{model_minutes_text}</div>
            </div>

            <div class="model-player-metric">
                <div class="model-player-label">Fit Score</div>
                <div class="model-player-value">{model_fit_text}</div>
            </div>

            <div class="model-player-metric">
                <div class="model-player-label">Cluster</div>
                <div class="model-player-value">{model_cluster_text}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

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

tab_overview, tab_similarity, tab_radar, tab_heatmap, tab_market, tab_table = st.tabs(
    [
        "📌 Overview",
        "🧬 Similaridade",
        "📡 Radar",
        "🔥 Heat Map",
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

with tab_heatmap:

    st.markdown("### Mapa de manchas do jogador modelo")

    st.caption(
        "Visualização espacial das ações do jogador modelo com base nas coordenadas dos eventos StatsBomb."
    )

    if spatial_events is None:
        st.info(
            "Para visualizar o mapa de manchas, adiciona o ficheiro "
            "`outputs/dashboard_spatial_events.csv` exportado a partir do notebook "
            "ou carrega o ficheiro CSV na sidebar."
        )

    else:
        heatmap_event_group = st.selectbox(
            "Tipo de ações para o mapa",
            [
                "Todas as ações",
                "Ações ofensivas",
                "Passes",
                "Conduções",
                "Dribles",
                "Remates"
            ],
            index=0
        )

        fig_heatmap, player_spatial_events = plot_player_heatmap(
            spatial_events=spatial_events,
            player_id=target_player_id,
            player_name=target_player_name,
            event_group=heatmap_event_group
        )

        if fig_heatmap is None:
            st.warning(
                "Não existem eventos espaciais suficientes para este jogador "
                "com o filtro selecionado."
            )

        else:
            st.plotly_chart(
                fig_heatmap,
                use_container_width=True
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Ações no mapa",
                len(player_spatial_events)
            )

            c2.metric(
                "Tipo de ação",
                heatmap_event_group
            )

            c3.metric(
                "Jogador modelo",
                target_player_name
            )

            st.caption(
                "Nota: o mapa é construído com base nos eventos StatsBomb da Copa América 2024. "
                "A interpretação deve considerar os minutos jogados, o contexto tático e o número "
                "de jogos disponíveis."
            )

            if len(player_spatial_events) > 0:
                st.markdown("### Distribuição das ações usadas no mapa")

                event_summary = (
                    player_spatial_events["event_type"]
                    .value_counts()
                    .reset_index()
                )

                event_summary.columns = [
                    "Tipo de evento",
                    "Número de ações"
                ]

                st.dataframe(
                    event_summary,
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
