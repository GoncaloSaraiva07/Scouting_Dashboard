
# =========================================================
# STREAMLIT DASHBOARD — SCOUTING NO FUTEBOL
# Dashboard-ready: Player DNA + Market + Clustering + Similarity + Recommendation
# =========================================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import html as html_lib
import streamlit.components.v1 as components
import base64

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

# =========================================================
# HEADER VISUAL — DASHBOARD
# =========================================================

# Logo Copa America 2024

APP_DIR = Path(__file__).resolve().parent

COPA_LOGO_CANDIDATES = [
    APP_DIR / "assets" / "copa_america_2024_logo.png",
    APP_DIR / "assets" / "copa_america_2024_logo.jpg",
    APP_DIR / "assets" / "copa_america_2024_logo.jpeg",
    Path("assets") / "copa_america_2024_logo.png",
    Path("assets") / "copa_america_2024_logo.jpg",
    Path("assets") / "copa_america_2024_logo.jpeg",
]

def find_copa_logo_path():
    for path in COPA_LOGO_CANDIDATES:
        if path.exists():
            return path
    return None


def image_to_base64(image_path):
    if image_path is None:
        return None

    image_path = Path(image_path)

    if not image_path.exists():
        return None

    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    suffix = image_path.suffix.lower().replace(".", "")

    if suffix == "jpg":
        suffix = "jpeg"

    return f"data:image/{suffix};base64,{encoded}"


def render_dashboard_header():
    logo_path = find_copa_logo_path()
    logo_base64 = image_to_base64(logo_path) if logo_path is not None else None

    if logo_base64 is None:
        logo_html = """
        <div style="
            font-size: 15px;
            font-weight: 800;
            color: #98A2B3;
            text-align: right;
            white-space: nowrap;
        ">
            Copa América 2024
        </div>
        """
    else:
        logo_html = f"""
        <img src="{logo_base64}" style="
            width: 165px;
            max-height: 101px;
            object-fit: contain;
            display: block;
        ">
        """

    header_html = f"""
    <div style="
        width: 100%;
        box-sizing: border-box;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 28px;
        padding: 10px 4px 20px 4px;
        border-bottom: 1px solid rgba(208, 213, 221, 0.55);
        font-family: Inter, Segoe UI, Arial, sans-serif;
    ">

        <div style="
            display: flex;
            align-items: center;
            gap: 16px;
            min-width: 0;
        ">
            <div style="
                width: 56px;
                height: 56px;
                min-width: 56px;
                border-radius: 17px;
                background: linear-gradient(135deg, #1D4ED8, #06B6D4);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 31px;
                box-shadow: 0 9px 24px rgba(29, 78, 216, 0.28);
            ">
                ⚽
            </div>

            <div>
                <div style="
                    font-size: 39px;
                    font-weight: 950;
                    color: #1D2939;
                    letter-spacing: 1.2px;
                    line-height: 1.05;
                    white-space: nowrap;
                ">
                    Scouting Dashboard
                </div>

                <div style="
                    font-size: 16px;
                    color: #667085;
                    margin-top: 8px;
                    white-space: nowrap;
                ">
                    Copa América 2024 · Player DNA · Similaridade · Mercado · Recomendação
                </div>
            </div>
        </div>

        <div style="
            min-width: 170px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
        ">
            {logo_html}
        </div>

    </div>
    """

    components.html(
        header_html,
        height=120,
        scrolling=False
    )


render_dashboard_header()

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
    Desenha campo StatsBomb 120x80 alinhado com o heatmap.
    """

    line_color = "rgba(30, 30, 30, 0.85)"

    shapes = []

    # Campo completo
    shapes.append(
        dict(
            type="rect",
            x0=0, y0=0, x1=120, y1=80,
            line=dict(color=line_color, width=2),
            fillcolor="rgba(0,0,0,0)",
            layer="above"
        )
    )

    # Linha do meio
    shapes.append(
        dict(
            type="line",
            x0=60, y0=0, x1=60, y1=80,
            line=dict(color=line_color, width=1),
            layer="above"
        )
    )

    # Círculo central
    shapes.append(
        dict(
            type="circle",
            x0=50, y0=30, x1=70, y1=50,
            line=dict(color=line_color, width=1),
            fillcolor="rgba(0,0,0,0)",
            layer="above"
        )
    )

    # Grandes áreas
    shapes.append(
        dict(
            type="rect",
            x0=0, y0=18, x1=18, y1=62,
            line=dict(color=line_color, width=1),
            fillcolor="rgba(0,0,0,0)",
            layer="above"
        )
    )

    shapes.append(
        dict(
            type="rect",
            x0=102, y0=18, x1=120, y1=62,
            line=dict(color=line_color, width=1),
            fillcolor="rgba(0,0,0,0)",
            layer="above"
        )
    )

    # Pequenas áreas
    shapes.append(
        dict(
            type="rect",
            x0=0, y0=30, x1=6, y1=50,
            line=dict(color=line_color, width=1),
            fillcolor="rgba(0,0,0,0)",
            layer="above"
        )
    )

    shapes.append(
        dict(
            type="rect",
            x0=114, y0=30, x1=120, y1=50,
            line=dict(color=line_color, width=1),
            fillcolor="rgba(0,0,0,0)",
            layer="above"
        )
    )

    fig.update_layout(shapes=shapes)

    fig.update_xaxes(
        range=[0, 120],
        visible=False,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        constrain="domain"
    )

    fig.update_yaxes(
        range=[80, 0],
        visible=False,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        scaleanchor="x",
        scaleratio=1
    )

    return fig

def plot_player_heatmap(spatial_events, player_id, player_name, event_group="Todas as ações"):
    """
    Cria mapa de manchas do jogador modelo no campo StatsBomb 120x80.
    Versão corrigida para evitar que a densidade pareça menor que o campo.
    """

    if spatial_events is None or len(spatial_events) == 0:
        return None, pd.DataFrame()

    data = spatial_events.copy()

    required_cols = ["statsbomb_player_id", "event_type", "x", "y"]
    missing_cols = [col for col in required_cols if col not in data.columns]

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

    data = data[
        data["x"].between(0, 120) &
        data["y"].between(0, 80)
    ].copy()

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

    # Fundo completo do campo para garantir que a zona visual ocupa 120x80
    fig.add_trace(
        go.Heatmap(
            x=[0, 120],
            y=[0, 80],
            z=[[0, 0], [0, 0]],
            colorscale=[
                [0, "rgba(90, 55, 125, 0.72)"],
                [1, "rgba(90, 55, 125, 0.72)"]
            ],
            showscale=False,
            hoverinfo="skip",
            name="Fundo"
        )
    )

    # Densidade principal
    fig.add_trace(
        go.Histogram2d(
            x=player_events["x"],
            y=player_events["y"],
            xbins=dict(start=0, end=120, size=3),
            ybins=dict(start=0, end=80, size=3),
            colorscale=[
                [0.00, "rgba(90, 55, 125, 0.15)"],
                [0.20, "rgba(58, 82, 139, 0.45)"],
                [0.40, "rgba(32, 145, 140, 0.62)"],
                [0.65, "rgba(94, 201, 98, 0.78)"],
                [0.85, "rgba(253, 231, 37, 0.90)"],
                [1.00, "rgba(255, 255, 190, 0.96)"],
            ],
            zsmooth="best",
            showscale=True,
            colorbar=dict(title="Densidade"),
            hoverinfo="skip",
            name="Densidade"
        )
    )

    # Pontos das ações
    fig.add_trace(
        go.Scatter(
            x=player_events["x"],
            y=player_events["y"],
            mode="markers",
            marker=dict(
                size=4,
                color="white",
                opacity=0.36,
                line=dict(width=0)
            ),
            name="Ações",
            hovertemplate="x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>"
        )
    )

    fig = draw_pitch_layout(fig)

    fig.update_layout(
        title=f"Mapa de manchas — {player_name} | {event_group}",
        height=560,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False
    )

    return fig, player_events

# =========================================================
# 5. SIDEBAR — FILTROS DE SCOUTING
# =========================================================

with st.sidebar:
    st.header("2. Filtros de Scouting")

    profile_labels = {
        cfg["label"]: key
        for key, cfg in PROFILE_CONFIG.items()
    }

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

    # Base inicial do perfil selecionado
    role_df_raw = df[df["role_family"] == expected_role].copy()

    if len(role_df_raw) == 0:
        st.warning("Não existem jogadores disponíveis para este perfil.")
        st.stop()

    # Garantir minutos numéricos
    role_df_raw[OFFICIAL_MINUTES_COL] = pd.to_numeric(
        role_df_raw[OFFICIAL_MINUTES_COL],
        errors="coerce"
    ).fillna(0)

    # -----------------------------------------------------
    # Intervalo de idade
    # -----------------------------------------------------

    min_age = None
    max_age = None

    if "age" in role_df_raw.columns:

        role_df_raw["age"] = pd.to_numeric(
            role_df_raw["age"],
            errors="coerce"
        )

        valid_ages = role_df_raw["age"].dropna()

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

                age_range = st.slider(
                    "Intervalo de idade",
                    min_value=min_age_data,
                    max_value=max_age_data,
                    value=(min_age_data, max_age_data),
                    step=1
                )

                min_age = age_range[0]
                max_age = age_range[1]

        else:
            st.info(
                "Filtro de idade indisponível: não existem idades válidas para este perfil."
            )

    # -----------------------------------------------------
    # Intervalo de valor de mercado
    # -----------------------------------------------------

    min_market_value = None
    max_market_value = None

    if "market_value_eur_2024" in role_df_raw.columns:

        role_df_raw["market_value_eur_2024"] = pd.to_numeric(
            role_df_raw["market_value_eur_2024"],
            errors="coerce"
        )

        valid_market_values = role_df_raw["market_value_eur_2024"].dropna()

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

                market_value_range_m = st.slider(
                    "Intervalo de valor de mercado (€M)",
                    min_value=min_mv_m,
                    max_value=max_mv_m,
                    value=(min_mv_m, max_mv_m),
                    step=0.5
                )

                min_market_value = market_value_range_m[0] * 1_000_000
                max_market_value = market_value_range_m[1] * 1_000_000

        else:
            st.info(
                "Filtro de valor de mercado indisponível: não existem valores válidos para este perfil."
            )

    # -----------------------------------------------------
    # Minutos mínimos
    # -----------------------------------------------------

    valid_minutes = role_df_raw[OFFICIAL_MINUTES_COL].dropna()

    if len(valid_minutes) > 0:
        max_minutes_available = int(valid_minutes.max())
    else:
        max_minutes_available = 0

    max_minutes_available = max(1, max_minutes_available)
    default_min_minutes = min(180, max_minutes_available)

    min_minutes = int(
        st.slider(
            "Minutos mínimos",
            min_value=0,
            max_value=max_minutes_available,
            value=default_min_minutes,
            step=30
        )
    )

    # -----------------------------------------------------
    # Número de recomendações
    # -----------------------------------------------------

    top_n = st.slider(
        "Número de recomendações",
        min_value=5,
        max_value=30,
        value=15,
        step=5
    )


# =========================================================
# 5.1 CONFIGURAÇÕES INTERNAS DO CLUSTER
# =========================================================
# As opções visuais "Apenas mesmo cluster" e "Bónus ao mesmo cluster"
# foram removidas da sidebar. Mantêm-se valores fixos para não quebrar
# a função compute_similar_players.

same_cluster_only = False
same_cluster_bonus = False
exclude_same_team = False


# =========================================================
# 6. FILTRAR BASE DO PERFIL
# =========================================================

role_df = role_df_raw.copy()

# ---------------------------------------------------------
# Aplicar filtro de minutos
# ---------------------------------------------------------

role_df = role_df[
    role_df[OFFICIAL_MINUTES_COL] >= min_minutes
].copy()


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


if len(role_df) == 0:
    st.warning("Não existem jogadores para os filtros selecionados.")
    st.stop()


# ---------------------------------------------------------
# Garantir variáveis necessárias
# ---------------------------------------------------------

role_df[fit_col] = pd.to_numeric(
    role_df[fit_col],
    errors="coerce"
).fillna(0)

if "market_opportunity_score" not in role_df.columns:
    role_df["market_opportunity_score"] = 0.5

role_df["market_opportunity_score"] = pd.to_numeric(
    role_df["market_opportunity_score"],
    errors="coerce"
).fillna(0.5)

role_df = compute_cluster_fit_score(role_df, fit_col)

# =========================================================
# 6.1 JOGADOR MODELO — SELEÇÃO CENTRAL
# =========================================================

st.subheader(f"Perfil: {selected_profile_label}")

st.markdown("### Selecionar Jogador modelo")

model_options_df = (
    role_df
    .sort_values(fit_col, ascending=False)
    .copy()
)

model_options_df["player_option_label"] = model_options_df.apply(
    safe_player_label,
    axis=1
)

if len(model_options_df) == 0:
    st.warning(
        "Não existem jogadores disponíveis para os filtros selecionados. "
        "Alarga os filtros de idade, valor de mercado ou minutos."
    )
    st.stop()

selected_model_label = st.selectbox(
    "Selecionar jogador modelo",
    options=model_options_df["player_option_label"].tolist(),
    index=0,
    key="central_model_player"
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
# FUNÇÕES VISUAIS — BANDEIRA / PAÍS DO JOGADOR MODELO
# =========================================================

COUNTRY_CODE_MAP = {
    "Argentina": "ar",
    "Bolivia": "bo",
    "Brazil": "br",
    "Brasil": "br",
    "Canada": "ca",
    "Chile": "cl",
    "Colombia": "co",
    "Costa Rica": "cr",
    "Ecuador": "ec",
    "Jamaica": "jm",
    "Mexico": "mx",
    "México": "mx",
    "Panama": "pa",
    "Panamá": "pa",
    "Paraguay": "py",
    "Peru": "pe",
    "Perú": "pe",
    "United States": "us",
    "United States of America": "us",
    "USA": "us",
    "Uruguay": "uy",
    "Venezuela": "ve",
}

COUNTRY_THEME_MAP = {
    "ar": {"accent": "#75AADB", "accent2": "#F6B40E"},
    "bo": {"accent": "#D52B1E", "accent2": "#007934"},
    "br": {"accent": "#009B3A", "accent2": "#FFDF00"},
    "ca": {"accent": "#D80621", "accent2": "#FFFFFF"},
    "cl": {"accent": "#D52B1E", "accent2": "#0039A6"},
    "co": {"accent": "#FCD116", "accent2": "#CE1126"},
    "cr": {"accent": "#002B7F", "accent2": "#CE1126"},
    "ec": {"accent": "#FFD100", "accent2": "#EF3340"},
    "jm": {"accent": "#009B3A", "accent2": "#FED100"},
    "mx": {"accent": "#006847", "accent2": "#CE1126"},
    "pa": {"accent": "#005293", "accent2": "#D21034"},
    "py": {"accent": "#D52B1E", "accent2": "#0038A8"},
    "pe": {"accent": "#D91023", "accent2": "#FFFFFF"},
    "us": {"accent": "#3C3B6E", "accent2": "#B22234"},
    "uy": {"accent": "#0038A8", "accent2": "#FCD116"},
    "ve": {"accent": "#FCD116", "accent2": "#CF142B"},
}


def get_country_code_from_player(row):
    """
    Usa primeiro team_name, porque na Copa América representa a seleção.
    Se não existir, usa nationality.
    """
    possible_country = row.get("team_name", None)

    if possible_country is None or pd.isna(possible_country):
        possible_country = row.get("nationality", None)

    if possible_country is None or pd.isna(possible_country):
        return None, "n.d."

    country_name = str(possible_country).strip()

    country_code = COUNTRY_CODE_MAP.get(country_name)

    return country_code, country_name


def get_flag_url(country_code):
    """
    Usa FlagCDN para imagem da bandeira.
    Exemplo: https://flagcdn.com/w640/co.png
    """
    if country_code is None:
        return ""

    return f"https://flagcdn.com/w640/{country_code.lower()}.png"


def get_country_theme(country_code):
    """
    Define cores principais do cartão.
    """
    default_theme = {
        "accent": "#667085",
        "accent2": "#98A2B3"
    }

    if country_code is None:
        return default_theme

    return COUNTRY_THEME_MAP.get(
        country_code.lower(),
        default_theme
    )


def safe_text(value, default="n.d."):
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    return html_lib.escape(str(value))


def safe_numeric_text(value, decimals=3, default="n.d."):
    value = pd.to_numeric(
        pd.Series([value]),
        errors="coerce"
    ).iloc[0]

    if pd.isna(value):
        return default

    return f"{value:.{decimals}f}"

# =========================================================
# JOGADOR MODELO EM DESTAQUE — COMPONENT HTML
# =========================================================

def safe_text(value, default="n.d."):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return html_lib.escape(str(value))


def safe_numeric_text(value, decimals=3, default="n.d."):
    value = pd.to_numeric(
        pd.Series([value]),
        errors="coerce"
    ).iloc[0]

    if pd.isna(value):
        return default

    return f"{value:.{decimals}f}"


model_player_name = safe_text(target_player_name)
model_position = safe_text(target_player.get("position", "n.d."))
model_team = safe_text(target_player.get("team_name", "n.d."))
model_profile = safe_text(selected_profile_label)
model_club = safe_text(target_player.get("club", "n.d."))
model_league = safe_text(target_player.get("league", "n.d."))
model_cluster = safe_text(target_player.get("cluster_label", "Sem cluster"))

model_age_value = pd.to_numeric(
    pd.Series([target_player.get("age", np.nan)]),
    errors="coerce"
).iloc[0]

if pd.isna(model_age_value):
    model_age_text = "n.d."
else:
    model_age_text = f"{int(model_age_value)} anos"

model_market_text = format_eur(
    target_player.get("market_value_eur_2024", np.nan)
)

model_minutes_value = pd.to_numeric(
    pd.Series([target_player.get(OFFICIAL_MINUTES_COL, np.nan)]),
    errors="coerce"
).iloc[0]

if pd.isna(model_minutes_value):
    model_minutes_text = "n.d."
else:
    model_minutes_text = f"{int(model_minutes_value)} min"

model_fit_text = safe_numeric_text(
    target_player.get(fit_col, np.nan),
    decimals=3
)

# =========================================================
# JOGADOR MODELO EM DESTAQUE — CARTÃO COM BANDEIRA
# =========================================================

country_code, model_country_name = get_country_code_from_player(target_player)
flag_url = get_flag_url(country_code)
country_theme = get_country_theme(country_code)

accent_color = country_theme["accent"]
accent_color_2 = country_theme["accent2"]

model_player_name = safe_text(target_player_name)
model_position = safe_text(target_player.get("position", "n.d."))
model_team = safe_text(target_player.get("team_name", "n.d."))
model_country = safe_text(model_country_name)
model_profile = safe_text(selected_profile_label)
model_club = safe_text(target_player.get("club", "n.d."))
model_league = safe_text(target_player.get("league", "n.d."))
model_cluster = safe_text(target_player.get("cluster_label", "Sem cluster"))

model_age_value = pd.to_numeric(
    pd.Series([target_player.get("age", np.nan)]),
    errors="coerce"
).iloc[0]

if pd.isna(model_age_value):
    model_age_text = "n.d."
else:
    model_age_text = f"{int(model_age_value)} anos"

model_market_text = format_eur(
    target_player.get("market_value_eur_2024", np.nan)
)

model_minutes_value = pd.to_numeric(
    pd.Series([target_player.get(OFFICIAL_MINUTES_COL, np.nan)]),
    errors="coerce"
).iloc[0]

if pd.isna(model_minutes_value):
    model_minutes_text = "n.d."
else:
    model_minutes_text = f"{int(model_minutes_value)} min"

model_fit_text = safe_numeric_text(
    target_player.get(fit_col, np.nan),
    decimals=3
)

if flag_url != "":
    flag_background_css = f"""
        background-image:
            linear-gradient(135deg, rgba(11,18,32,0.94) 0%, rgba(23,32,51,0.90) 50%, rgba(36,52,77,0.86) 100%),
            url('{flag_url}');
        background-size: cover;
        background-position: center;
    """
else:
    flag_background_css = """
        background: linear-gradient(135deg, #0B1220 0%, #172033 55%, #25344D 100%);
    """


model_card_html = f"""
<div style="
    {flag_background_css}
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    padding: 30px 34px;
    color: white;
    font-family: Inter, Segoe UI, Arial, sans-serif;
    box-shadow: 0 16px 38px rgba(15, 23, 42, 0.30);
    border: 1px solid rgba(255,255,255,0.10);
    margin-bottom: 18px;
">

    <div style="
        position: absolute;
        top: -40px;
        right: -40px;
        width: 210px;
        height: 210px;
        border-radius: 50%;
        background: radial-gradient(circle, {accent_color}55 0%, rgba(255,255,255,0.00) 70%);
    "></div>

    <div style="
        position: absolute;
        bottom: -70px;
        left: -50px;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        background: radial-gradient(circle, {accent_color_2}38 0%, rgba(255,255,255,0.00) 72%);
    "></div>

    <div style="position: relative; z-index: 2;">

        <div style="display: flex; align-items: center; justify-content: space-between; gap: 16px;">
            <div style="display: flex; align-items: center; gap: 16px;">
                <div style="
                    width: 54px;
                    height: 54px;
                    border-radius: 16px;
                    background: linear-gradient(135deg, {accent_color}, {accent_color_2});
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 30px;
                    box-shadow: 0 8px 18px rgba(0,0,0,0.25);
                ">🎯</div>

                <div>
                    <div style="
                        font-size: 31px;
                        font-weight: 900;
                        letter-spacing: 0.3px;
                        line-height: 1.15;
                    ">
                        {model_player_name}
                    </div>

                    <div style="
                        font-size: 15px;
                        color: #E4E7EC;
                        margin-top: 7px;
                    ">
                        {model_position}
                    </div>
                </div>
            </div>

            <div style="
                text-align: right;
                min-width: 130px;
            ">
                <div style="
                    font-size: 12px;
                    color: #D0D5DD;
                    text-transform: uppercase;
                    letter-spacing: 0.7px;
                    margin-bottom: 6px;
                ">Seleção</div>

                <div style="
                    font-size: 20px;
                    font-weight: 850;
                ">{model_country}</div>
            </div>
        </div>

        <div style="margin-top: 20px; margin-bottom: 22px;">
            <span style="
                display: inline-block;
                background: rgba(255, 255, 255, 0.13);
                border: 1px solid rgba(255, 255, 255, 0.18);
                color: #F9FAFB;
                border-radius: 999px;
                padding: 7px 13px;
                font-size: 13px;
                font-weight: 700;
                margin-right: 8px;
                margin-bottom: 8px;
                backdrop-filter: blur(4px);
            ">Clube: {model_club}</span>

            <span style="
                display: inline-block;
                background: rgba(255, 255, 255, 0.13);
                border: 1px solid rgba(255, 255, 255, 0.18);
                color: #F9FAFB;
                border-radius: 999px;
                padding: 7px 13px;
                font-size: 13px;
                font-weight: 700;
                margin-right: 8px;
                margin-bottom: 8px;
                backdrop-filter: blur(4px);
            ">Liga: {model_league}</span>

            <span style="
                display: inline-block;
                background: rgba(255, 255, 255, 0.13);
                border: 1px solid rgba(255, 255, 255, 0.18);
                color: #F9FAFB;
                border-radius: 999px;
                padding: 7px 13px;
                font-size: 13px;
                font-weight: 700;
                margin-right: 8px;
                margin-bottom: 8px;
                backdrop-filter: blur(4px);
            ">Cluster: {model_cluster}</span>
        </div>

        <div style="
            display: grid;
            grid-template-columns: repeat(5, minmax(120px, 1fr));
            gap: 14px;
            margin-top: 12px;
        ">
            <div style="
                background: rgba(255,255,255,0.10);
                border: 1px solid rgba(255,255,255,0.16);
                border-left: 4px solid {accent_color};
                border-radius: 16px;
                padding: 15px 16px;
                backdrop-filter: blur(5px);
            ">
                <div style="font-size: 12px; color: #D0D5DD; margin-bottom: 7px; text-transform: uppercase; letter-spacing: 0.5px;">Idade</div>
                <div style="font-size: 22px; font-weight: 900;">{model_age_text}</div>
            </div>

            <div style="
                background: rgba(255,255,255,0.10);
                border: 1px solid rgba(255,255,255,0.16);
                border-left: 4px solid {accent_color_2};
                border-radius: 16px;
                padding: 15px 16px;
                backdrop-filter: blur(5px);
            ">
                <div style="font-size: 12px; color: #D0D5DD; margin-bottom: 7px; text-transform: uppercase; letter-spacing: 0.5px;">Valor mercado</div>
                <div style="font-size: 22px; font-weight: 900;">{model_market_text}</div>
            </div>

            <div style="
                background: rgba(255,255,255,0.10);
                border: 1px solid rgba(255,255,255,0.16);
                border-left: 4px solid {accent_color};
                border-radius: 16px;
                padding: 15px 16px;
                backdrop-filter: blur(5px);
            ">
                <div style="font-size: 12px; color: #D0D5DD; margin-bottom: 7px; text-transform: uppercase; letter-spacing: 0.5px;">Minutos</div>
                <div style="font-size: 22px; font-weight: 900;">{model_minutes_text}</div>
            </div>

            <div style="
                background: rgba(255,255,255,0.10);
                border: 1px solid rgba(255,255,255,0.16);
                border-left: 4px solid {accent_color_2};
                border-radius: 16px;
                padding: 15px 16px;
                backdrop-filter: blur(5px);
            ">
                <div style="font-size: 12px; color: #D0D5DD; margin-bottom: 7px; text-transform: uppercase; letter-spacing: 0.5px;">Fit Score</div>
                <div style="font-size: 22px; font-weight: 900;">{model_fit_text}</div>
            </div>

            <div style="
                background: rgba(255,255,255,0.10);
                border: 1px solid rgba(255,255,255,0.16);
                border-left: 4px solid {accent_color};
                border-radius: 16px;
                padding: 15px 16px;
                backdrop-filter: blur(5px);
            ">
                <div style="font-size: 12px; color: #D0D5DD; margin-bottom: 7px; text-transform: uppercase; letter-spacing: 0.5px;">Perfil DNA</div>
                <div style="font-size: 18px; font-weight: 900;">{model_profile}</div>
            </div>
        </div>
    </div>
</div>
"""

components.html(
    model_card_html,
    height=350,
    scrolling=False
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

    st.markdown("### Top jogadores por Fit Score")

    profile_ranking = (
        role_df
        .sort_values(fit_col, ascending=False)
        .head(top_n)
        .copy()
    )

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

    fig_fit.update_layout(
        height=520,
        yaxis_title="",
        xaxis_title="Fit Score",
        margin=dict(l=20, r=20, t=70, b=20)
    )

    st.plotly_chart(
        fig_fit,
        use_container_width=True
    )

    st.markdown("### Ranking do perfil selecionado")

    overview_cols = [
        "player_name",
        "position",
        "team_name",
        "age",
        "market_value_eur_2024",
        OFFICIAL_MINUTES_COL,
        fit_col,
        "cluster_label"
    ]

    overview_cols = [
        col for col in overview_cols
        if col in profile_ranking.columns
    ]

    st.dataframe(
        profile_ranking[overview_cols],
        use_container_width=True,
        hide_index=True
    )

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
