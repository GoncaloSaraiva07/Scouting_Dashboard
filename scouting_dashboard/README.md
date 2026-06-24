# Scouting Dashboard — Player DNA & Similarity Engine

Dashboard interativo em **Streamlit** para análise de scouting no futebol, com base em métricas criadas em Python a partir de eventos StatsBomb, dados de mercado e perfis de jogador definidos no módulo **Player DNA**.

O objetivo do dashboard é permitir selecionar um perfil de jogador, escolher um jogador-modelo e encontrar jogadores semelhantes, combinando performance técnica, idade, valor de mercado, cluster e score de recomendação.

---

## 1. Perfis disponíveis

O dashboard está preparado para trabalhar com cinco perfis criados no módulo **Player DNA**:

| Código do perfil | Nome do perfil |
|---|---|
| `extremo_1x1` | Extremo 1x1 |
| `extremo_finalizador` | Extremo Finalizador |
| `extremo_completo` | Extremo Completo |
| `medio_ofensivo_criativo` | Médio Ofensivo Criativo |
| `lateral_ofensivo` | Lateral Ofensivo |

Cada perfil tem um `fit_score` associado, criado previamente no notebook.

---

## 2. Estrutura recomendada do projeto

```text
scouting_dashboard/
│
├── app_scouting_dashboard.py
├── requirements_dashboard.txt
├── README.md
│
└── outputs/
    └── dashboard_scouting_data.xlsx
```

O ficheiro `dashboard_scouting_data.xlsx` deve ser exportado a partir do notebook principal, depois de estarem criados os módulos:

```text
Player DNA
Scoring Engine
Market Integration
Market Filters
Clustering
Similarity Engine / Recommendation Engine
```

---

## 3. Ficheiros necessários

### `app_scouting_dashboard.py`

Aplicação principal em Streamlit.

### `requirements_dashboard.txt`

Lista de bibliotecas Python necessárias para correr o dashboard.

### `outputs/dashboard_scouting_data.xlsx`

Base final exportada a partir do notebook. Deve conter, sempre que possível, colunas como:

```text
player_name
statsbomb_player_id
team_name
position
role_family
age
club
league
market_value_eur_2024
minutes_played
minutes_copa_america_2024
one_v_one_score
progression_score
xA_quality_score
xg_quality_score
gplus_proxy_score
cross_quality_score
extremo_1x1_fit_score
extremo_finalizador_fit_score
extremo_completo_fit_score
medio_ofensivo_criativo_fit_score
lateral_ofensivo_fit_score
market_opportunity_score
cluster_id
cluster_label
cluster_description
```

---

## 4. Como exportar os dados no notebook

No fim do notebook, depois de teres a base `scoring_market` pronta, corre uma célula semelhante a esta:

```python
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)

output_path = "outputs/dashboard_scouting_data.xlsx"

scoring_market.to_excel(
    output_path,
    sheet_name="Dashboard_Data",
    index=False
)

print(f"Dataset exportado para: {output_path}")
print(scoring_market.shape)
```

Depois copia o ficheiro `dashboard_scouting_data.xlsx` para a pasta `outputs/` do projeto Streamlit.

---

## 5. Instalação no VS Code

### 5.1 Criar ambiente virtual

No terminal do VS Code, dentro da pasta do projeto:

```bash
python -m venv .venv
```

Ativar no Windows:

```bash
.venv\Scripts\activate
```

Ativar no macOS/Linux:

```bash
source .venv/bin/activate
```

### 5.2 Instalar dependências

```bash
pip install -r requirements_dashboard.txt
```

Se não tiveres o ficheiro `requirements_dashboard.txt`, instala diretamente:

```bash
pip install streamlit pandas numpy plotly scikit-learn openpyxl
```

---

## 6. Como correr o dashboard

No terminal:

```bash
streamlit run app_scouting_dashboard.py
```

O Streamlit deverá abrir automaticamente no browser. Se não abrir, acede manualmente a:

```text
http://localhost:8501
```

---

## 7. Funcionalidades do dashboard

O dashboard permite:

- selecionar um dos cinco perfis Player DNA;
- filtrar por idade máxima;
- filtrar por valor máximo de mercado;
- filtrar por minutos mínimos;
- escolher um jogador-modelo;
- encontrar jogadores tecnicamente semelhantes;
- aplicar filtro ou bónus de cluster;
- analisar gráficos de radar;
- visualizar métricas de performance;
- comparar idade, valor de mercado e fit score;
- apoiar a tomada de decisão em scouting.

---

## 8. Lógica analítica usada

### Similarity Engine

A similaridade é calculada com base apenas em métricas técnicas:

```text
one_v_one_score
progression_score
xA_quality_score
xg_quality_score
gplus_proxy_score
cross_quality_score
```

A lógica usada é:

```text
MinMaxScaler
→ cosine similarity
→ comparação dentro da mesma role_family
→ mercado aplicado apenas como filtro posterior
→ cluster usado como filtro ou bónus opcional
```

Isto significa que a idade e o valor de mercado não entram diretamente no cálculo da similaridade técnica.

---

## 9. Recommendation Engine

O score final de recomendação combina:

```text
similarity_score_adjusted
fit_score do perfil escolhido
market_opportunity_score
cluster_fit_score
```

A recomendação não substitui a avaliação de scouting; serve como ranking inicial para priorizar jogadores a analisar.

---

## 10. Atualizar os dados

Sempre que alterares o notebook, deves:

1. voltar a correr a pipeline analítica;
2. exportar novamente `dashboard_scouting_data.xlsx`;
3. substituir o ficheiro antigo na pasta `outputs/`;
4. reiniciar o Streamlit.

---

## 11. Problemas comuns

### Erro: ficheiro não encontrado

Confirma se existe:

```text
outputs/dashboard_scouting_data.xlsx
```

### Erro: coluna não encontrada

Confirma se o notebook exportou todas as colunas necessárias, especialmente:

```text
role_family
market_value_eur_2024
market_opportunity_score
cluster_id
cluster_label
```

### Dashboard sem jogadores disponíveis

Verifica se os filtros estão demasiado restritivos, por exemplo:

```text
idade máxima muito baixa
valor de mercado muito baixo
minutos mínimos muito altos
```

---

## 12. Próximas melhorias possíveis

- adicionar página individual por jogador;
- incluir mapa de calor ou pitch map;
- adicionar comparação lado a lado entre dois jogadores;
- criar exportação automática de shortlist em Excel;
- adicionar filtros por clube, liga e nacionalidade;
- criar uma página específica para cada perfil Player DNA;
- integrar uma nota qualitativa de scouting manual.

---

## 13. Nota metodológica

Este dashboard utiliza dados quantitativos para apoiar decisões de scouting. Os rankings devem ser interpretados como uma primeira camada analítica e não como decisão final. A análise vídeo, contexto tático, modelo de jogo da equipa e disponibilidade real do jogador continuam a ser componentes essenciais na decisão de recrutamento.
