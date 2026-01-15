import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from contextlib import contextmanager
from typing import Optional
import re

GA_MEASUREMENT_ID = "G-NW1TCTKXM9"

def inject_ga4(measurement_id: str):
    st.components.v1.html(
        f"""
        <!-- Google tag (gtag.js) -->
        <script async src="https://www.googletagmanager.com/gtag/js?id={measurement_id}"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){{dataLayer.push(arguments);}}
          gtag('js', new Date());
          gtag('config', '{measurement_id}');
        </script>
        """,
        height=0,
    )

# Chama no início do app (1x)
inject_ga4(GA_MEASUREMENT_ID)

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Analisador de Planilhas", layout="wide")

# =========================
# TEMA / CSS
# =========================
st.markdown(
    """
    <style>
      .stApp {
        background: radial-gradient(circle at 20% 10%, rgba(99,102,241,0.10), transparent 35%),
                    radial-gradient(circle at 80% 0%, rgba(16,185,129,0.10), transparent 35%),
                    #0b1220;
        color: #e5e7eb;
      }

      .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem;
        max-width: 1100px;
      }

      .card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        backdrop-filter: blur(8px);
        margin-bottom: 14px;
      }

      .hero-title {
        font-size: 34px;
        font-weight: 800;
        margin: 0;
        line-height: 1.1;
      }
      .hero-sub {
        color: rgba(229,231,235,0.85);
        margin-top: 0.5rem;
        margin-bottom: 0;
      }

      .chip {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.10);
        font-size: 12px;
        margin-right: 8px;
      }

      section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
      }

      div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
      }

      div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 12px 12px 6px 12px;
      }

      div[data-testid="stFileUploader"] label {
        color: #e5e7eb !important;
        font-weight: 650;
      }

      div[data-testid="stFileUploader"] section {
        background: rgba(255,255,255,0.04) !important;
        border: 1px dashed rgba(255,255,255,0.20) !important;
        border-radius: 14px !important;
      }

      div[data-testid="stFileUploader"] section:hover {
        border-color: rgba(99,102,241,0.60) !important;
        background: rgba(99,102,241,0.08) !important;
      }

      div[data-testid="stFileUploader"] button {
        background: rgba(255,255,255,0.10) !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        color: #e5e7eb !important;
        border-radius: 14px !important;
      }

      div[data-testid="stFileUploader"] button:hover {
        border-color: rgba(16,185,129,0.55) !important;
        background: rgba(16,185,129,0.12) !important;
      }

      div[data-testid="stFileUploader"] small,
      div[data-testid="stFileUploader"] p,
      div[data-testid="stFileUploader"] span {
        color: rgba(229,231,235,0.85) !important;
      }

      input, textarea {
        color: #e5e7eb !important;
      }

      header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0px !important;
      }

      .stDownloadButton button,
      .stButton button {
        background: rgba(255,255,255,0.10) !important;
        color: #e5e7eb !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        border-radius: 14px !important;
        transition: all 0.15s ease-in-out;
      }

      .stDownloadButton button:hover,
      .stButton button:hover {
        background: rgba(255,255,255,0.14) !important;
        color: #ffffff !important;
        border-color: rgba(99,102,241,0.65) !important;
      }

      /* Loader circular (estilo Windows) */
      .win-loader-wrap{
        display:flex;
        align-items:center;
        gap:12px;
        padding: 10px 12px;
        border-radius: 14px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
      }
      .win-loader{
        width:18px;
        height:18px;
        border-radius:50%;
        border:2px solid rgba(255,255,255,0.22);
        border-top-color: rgba(99,102,241,0.95);
        animation: winspin 0.8s linear infinite;
      }
      @keyframes winspin{
        to { transform: rotate(360deg); }
      }

      /* Lista de relações (simples) */
      .rel-item{
        display:flex;
        gap:10px;
        align-items:flex-start;
        padding: 10px 12px;
        border-radius: 14px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 10px;
      }
      .rel-dot{
        width: 10px;
        height: 10px;
        border-radius: 999px;
        margin-top: 6px;
        background: rgba(99,102,241,0.9);
        box-shadow: 0 0 0 3px rgba(99,102,241,0.12);
        flex: 0 0 auto;
      }
      .rel-title{
        font-weight: 750;
        color: rgba(229,231,235,0.95);
        line-height: 1.2;
      }
      .rel-sub{
        color: rgba(229,231,235,0.78);
        margin-top: 2px;
        font-size: 13px;
      }
      .rel-badge{
        display:inline-block;
        padding: 3px 8px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.06);
        font-size: 12px;
        margin-left: 8px;
        color: rgba(229,231,235,0.9);
      }
    </style>
    """,
    unsafe_allow_html=True
)

@contextmanager
def card(title: str, subtitle: Optional[str] = None):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)
    yield
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# HERO
# =========================
st.markdown(
    """
    <p class="hero-title">📊 Analisador Inteligente de Planilhas</p>
    <p class="hero-sub">Envie um arquivo CSV ou XLSX e receba diagnóstico automático (nulos, outliers) e relatórios.</p>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <span class="chip">CSV</span>
    <span class="chip">XLSX</span>
    <span class="chip">Nulos</span>
    <span class="chip">Outliers</span>
    <span class="chip">Relações</span>
    <span class="chip">Métricas</span>
    <span class="chip">Export</span>
    <span class="chip">Período</span>
    """,
    unsafe_allow_html=True
)

# =========================
# HELPERS
# =========================
MESES_PT = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

DESCRIBE_PT = {
    "count": "contagem",
    "mean": "média",
    "std": "desvio padrão",
    "min": "mínimo",
    "25%": "25%",
    "50%": "mediana (50%)",
    "75%": "75%",
    "max": "máximo",
}

def ordenar_ano_mes_global(df: pd.DataFrame) -> pd.DataFrame:
    """
    Se existir Ano e Número Mês (qualquer variação de nome), ordena:
      - Ano desc (2025, 2024, 2023...)
      - Número Mês asc (1..12)
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    out = df.copy()

    candidatos_ano = ["Ano", "ANO", "ano"]
    candidatos_mes = ["Número Mês", "NUMERO_Mês", "NUMERO_MES", "Numero Mes", "numero_mes", "mes_num", "MES_NUM", "MES_NUMERO"]

    col_ano = next((c for c in candidatos_ano if c in out.columns), None)
    col_mes = next((c for c in candidatos_mes if c in out.columns), None)

    if not col_ano or not col_mes:
        return out

    out[col_ano] = pd.to_numeric(out[col_ano], errors="coerce").astype("Int64")
    out[col_mes] = pd.to_numeric(out[col_mes], errors="coerce").astype("Int64")

    out = out.sort_values([col_ano, col_mes], ascending=[False, True], na_position="last").reset_index(drop=True)
    return out

def show_df(df: pd.DataFrame, **kwargs):
    st.dataframe(ordenar_ano_mes_global(df), **kwargs)

def csv_bytes_sorted(df: pd.DataFrame) -> bytes:
    return ordenar_ano_mes_global(df).to_csv(index=False).encode("utf-8")

@st.cache_data(show_spinner=False)
def carregar_arquivo(bytes_data: bytes, nome: str) -> pd.DataFrame:
    nome = nome.lower()
    bio = BytesIO(bytes_data)
    if nome.endswith(".csv"):
        try:
            bio.seek(0)
            return pd.read_csv(bio, sep=None, engine="python", encoding="utf-8")
        except Exception:
            bio.seek(0)
            return pd.read_csv(bio, sep=None, engine="python", encoding="latin-1")
    else:
        bio.seek(0)
        return pd.read_excel(bio)

def tabela_nulos(df: pd.DataFrame) -> pd.DataFrame:
    nulos = df.isnull().sum()
    total = len(df)
    out = pd.DataFrame({
        "coluna": nulos.index,
        "nulos": nulos.values,
        "% nulos": (nulos.values / total * 100) if total else 0
    }).sort_values("nulos", ascending=False)
    return out

def outliers_iqr(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    if df.empty or not cols:
        return pd.DataFrame(columns=["coluna", "outliers", "% outliers", "limite_inf", "limite_sup", "q1", "q3", "iqr"])

    rows = []
    n = len(df)
    for c in cols:
        s = df[c].dropna()
        if s.empty:
            rows.append([c, 0, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan])
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        limite_inf = q1 - 1.5 * iqr
        limite_sup = q3 + 1.5 * iqr

        mask = (df[c] < limite_inf) | (df[c] > limite_sup)
        count = int(mask.sum(skipna=True))
        pct = (count / n * 100) if n else 0.0

        rows.append([c, count, pct, limite_inf, limite_sup, q1, q3, iqr])

    return pd.DataFrame(
        rows,
        columns=["coluna", "outliers", "% outliers", "limite_inf", "limite_sup", "q1", "q3", "iqr"]
    ).sort_values("outliers", ascending=False)

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def gerar_excel_relatorio(abas: dict) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for nome, tabela in abas.items():
            safe_name = str(nome)[:31]
            tabela_sorted = ordenar_ano_mes_global(tabela) if isinstance(tabela, pd.DataFrame) else tabela
            if isinstance(tabela_sorted, pd.DataFrame):
                tabela_sorted.to_excel(writer, sheet_name=safe_name, index=False)
            else:
                pd.DataFrame(tabela).to_excel(writer, sheet_name=safe_name, index=False)
    buffer.seek(0)
    return buffer.read()

_RE_NULOS_TEXTO = re.compile(r"^\s*(none|null|nan)\s*$", re.IGNORECASE)

def normalizar_texto_nulos(series: pd.Series) -> pd.Series:
    s = series.astype("string")
    s = s.where(~s.str.match(_RE_NULOS_TEXTO, na=False), other=pd.NA)
    return s

def _fmt_int_pt(n: float) -> str:
    try:
        return f"{int(round(float(n))):,}".replace(",", ".")
    except Exception:
        return str(n)

def _fmt_float_pt(x: float, casas: int = 2) -> str:
    try:
        s = f"{float(x):,.{casas}f}"
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
        return s
    except Exception:
        return str(x)

def parse_data_robusta(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if dt.isna().mean() > 0.70:
        sn = pd.to_numeric(series, errors="coerce")
        mask_excel = sn.between(20000, 60000)
        if mask_excel.any():
            dt2 = pd.to_datetime(sn, unit="D", origin="1899-12-30", errors="coerce")
            dt = dt.fillna(dt2)
    return dt

def _norm_colname(s: str) -> str:
    s = str(s).upper()
    s = s.replace("Á", "A").replace("À", "A").replace("Â", "A").replace("Ã", "A")
    s = s.replace("É", "E").replace("È", "E").replace("Ê", "E")
    s = s.replace("Í", "I").replace("Ì", "I").replace("Î", "I")
    s = s.replace("Ó", "O").replace("Ò", "O").replace("Ô", "O").replace("Õ", "O")
    s = s.replace("Ú", "U").replace("Ù", "U").replace("Û", "U")
    s = s.replace("Ç", "C")
    s = re.sub(r"[^A-Z0-9]+", "_", s).strip("_")
    return s

def _score_nome_data(col: str) -> int:
    c = _norm_colname(col)
    score = 0
    if "DATA_SOLICITACAO" in c or ("DATA" in c and "SOLIC" in c):
        score += 10
    if any(k in c for k in ["DATA", "DATE", "DT", "CREATED", "CRIADO", "ABERTURA", "REQUERIMENTO", "SOLICITACAO"]):
        score += 3
    if c.startswith("DATA"):
        score += 2
    return score

def detectar_colunas_data(df: pd.DataFrame) -> list:
    cols_dt = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    cols_nome = sorted(df.columns, key=lambda c: _score_nome_data(str(c)), reverse=True)
    cols_nome = [c for c in cols_nome if _score_nome_data(str(c)) > 0]

    out = []
    for c in cols_dt + cols_nome:
        if c not in out:
            out.append(c)
    return out

def agregar_por_periodo_metric(df: pd.DataFrame, col_data: str, granularidade: str,
                               modo: str, col_metrica: Optional[str] = None):
    sdt = parse_data_robusta(df[col_data])
    invalidas = int(sdt.isna().sum())

    if granularidade == "Dia":
        periodo = sdt.dt.to_period("D")
        label = "PERIODO"
    elif granularidade == "Semana":
        periodo = sdt.dt.to_period("W")
        label = "PERIODO"
    else:
        periodo = sdt.dt.to_period("M")
        label = "PERIODO"

    if modo == "Contagem":
        out = (
            pd.DataFrame({label: periodo})
            .dropna()
            .groupby(label)
            .size()
            .reset_index(name="VALOR")
        )
    else:
        if not col_metrica or col_metrica not in df.columns:
            return pd.DataFrame(columns=[label, "VALOR"]), invalidas, label, "VALOR", "Métrica"

        metric = pd.to_numeric(df[col_metrica], errors="coerce")
        tmp = pd.DataFrame({label: periodo, "metric": metric}).dropna(subset=[label])

        if modo == "Soma":
            out = tmp.groupby(label)["metric"].sum(min_count=1).reset_index(name="VALOR")
        else:
            out = tmp.groupby(label)["metric"].mean().reset_index(name="VALOR")

    out = out.sort_values(label).reset_index(drop=True)

    if modo == "Contagem":
        return out, invalidas, label, "VALOR", "Total de registros"
    if modo == "Soma":
        return out, invalidas, label, "VALOR", f"Soma de {col_metrica}"
    return out, invalidas, label, "VALOR", f"Média de {col_metrica}"

def formatar_visao_mes_generica(tabela: pd.DataFrame, col_periodo: str, col_valor: str, nome_valor: str) -> pd.DataFrame:
    if tabela.empty or col_periodo not in tabela.columns:
        return tabela

    tmp = tabela.copy()

    if not pd.api.types.is_period_dtype(tmp[col_periodo]):
        tmp[col_periodo] = pd.PeriodIndex(tmp[col_periodo].astype(str), freq="M")

    tmp["Ano"] = pd.to_numeric(tmp[col_periodo].dt.year, errors="coerce").astype("Int64")
    tmp["Número Mês"] = pd.to_numeric(tmp[col_periodo].dt.month, errors="coerce").astype("Int64")
    tmp["Mês"] = tmp["Número Mês"].map(lambda x: MESES_PT.get(int(x), "") if pd.notna(x) else "")
    tmp[nome_valor] = tmp[col_valor]

    tmp = tmp[["Ano", "Número Mês", "Mês", nome_valor]]
    tmp = tmp.sort_values(["Ano", "Número Mês"], ascending=[False, True], na_position="last").reset_index(drop=True)
    return tmp

def tipo_pt(df: pd.DataFrame, col: str) -> str:
    s = df[col]
    nome = str(col).lower()

    if pd.api.types.is_datetime64_any_dtype(s):
        try:
            times = pd.Series(s.dropna().dt.time)
            if not times.empty and (times != pd.to_datetime("00:00:00").time()).any():
                return "Data hora"
            return "Data"
        except Exception:
            return "Data hora"

    if "timestamp" in nome or "data_hora" in nome or "datahora" in nome:
        return "Data hora"

    if pd.api.types.is_numeric_dtype(s):
        return "Número"

    return "Texto"

def traduzir_estatisticas(df_describe: pd.DataFrame) -> pd.DataFrame:
    out = df_describe.copy()
    if "estatística" in out.columns:
        out["estatística"] = out["estatística"].astype(str).map(lambda x: DESCRIBE_PT.get(x, x))
    return out

def resumo_leigo_coluna(df: pd.DataFrame, col: str) -> list[str]:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return ["Não há valores numéricos válidos nessa coluna para interpretar."]

    n = int(s.shape[0])
    nunique = int(s.nunique(dropna=True))
    unique_ratio = nunique / n if n else 0

    vmin = float(s.min())
    vmax = float(s.max())
    q25 = float(s.quantile(0.25))
    q50 = float(s.quantile(0.50))
    q75 = float(s.quantile(0.75))
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if n > 1 else 0.0

    is_integerish = bool(((s % 1) == 0).mean() > 0.98)
    range_rel = (vmax - vmin) / (abs(mean) + 1e-9) if mean != 0 else (vmax - vmin)
    id_like = (unique_ratio > 0.90) and is_integerish and (range_rel > 0.5)

    cv = abs(std / mean) if mean != 0 else 0.0
    if cv < 0.15:
        variacao_txt = "pouca variação (valores bem parecidos)"
    elif cv < 0.40:
        variacao_txt = "variação moderada"
    else:
        variacao_txt = "muita variação (valores bem espalhados)"

    bullets = []
    bullets.append(f"• Existem **{_fmt_int_pt(n)}** valores válidos nessa coluna.")
    bullets.append(f"• Um valor típico (meio da lista) fica perto de **{_fmt_int_pt(q50)}**.")
    bullets.append(f"• A maioria dos registros está entre **{_fmt_int_pt(q25)}** e **{_fmt_int_pt(q75)}**.")
    bullets.append(f"• Menor valor: **{_fmt_int_pt(vmin)}** | Maior valor: **{_fmt_int_pt(vmax)}**.")
    bullets.append(f"• Os dados têm **{variacao_txt}**.")
    if id_like:
        bullets.append("• ⚠️ Essa coluna **parece um identificador (ID/CPF/protocolo)**. Nesse caso, **média/soma** normalmente não ajudam; o mais útil costuma ser **contagem** e **distintos (únicos)**.")
    else:
        bullets.append("• ✅ Se isso for um valor (ex.: dinheiro, quantidade), **média/soma** podem fazer sentido. Se for um ID, prefira **únicos**.")
    bullets.append(f"• Valores distintos (únicos): **{_fmt_int_pt(nunique)}** ({_fmt_float_pt(unique_ratio*100, 1)}%).")
    return bullets


def parece_identificador(df: pd.DataFrame, col: str) -> bool:
    """
    Heurística mínima:
    - nome da coluna sugere id/cpf/protocolo/codigo
    - ou valores quase todos inteiros e quase todos distintos
    """
    nome = str(col).lower()

    # 1) Pistas no nome
    chaves_nome = ["id", "cpf", "cnpj", "protoc", "codigo", "cod_", "matricula", "registro", "chave", "hash"]
    if any(k in nome for k in chaves_nome):
        return True

    # 2) Comportamento do dado
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return False

    n = int(len(s))
    nunique = int(s.nunique(dropna=True))
    unique_ratio = nunique / n if n else 0

    # "quase inteiro"
    try:
        integerish = ((s % 1) == 0).mean() > 0.98
    except Exception:
        integerish = False

    # IDs tendem a ser quase todos únicos
    if integerish and unique_ratio > 0.90:
        return True

    return False


def texto_outliers_explicacao(df: pd.DataFrame, cols: list[str]) -> str:
    """
    Texto simples que adapta a explicação de outliers conforme o tipo provável das colunas:
    - IDs / identificadores: outliers geralmente não são relevantes
    - Métricas / valores: outliers podem indicar erro ou casos excepcionais
    """
    if not cols:
        return (
            "**O que isso significa?**\n\n"
            "Aqui mostramos valores muito fora do padrão. Selecione ao menos uma coluna numérica para ver esta análise."
        )

    ids = []
    metrics = []

    for c in cols:
        if parece_identificador(df, c):
            ids.append(c)
        else:
            metrics.append(c)

    linhas = []
    linhas.append("**O que isso significa?**")
    linhas.append("Aqui mostramos valores que ficam muito fora do comportamento normal de cada coluna.")
    linhas.append("")
    linhas.append("Isso pode indicar:")
    linhas.append("- erro de preenchimento")
    linhas.append("- casos excepcionais (fora do padrão)")
    linhas.append("- registros que merecem conferência")
    linhas.append("")

    if metrics:
        linhas.append(f"✅ **Colunas que parecem métricas/valores** (vale analisar outliers): `{', '.join(metrics)}`")
    if ids:
        linhas.append(f"⚠️ **Colunas que parecem identificadores** (ID/CPF/código): `{', '.join(ids)}`")
        linhas.append("Para identificadores, “outliers” geralmente **não são um problema**; o mais útil costuma ser **contagem** e **distintos (únicos)**.")

    return "\n".join(linhas)


def classificar_intensidade(v: float):
    a = abs(float(v))
    if a >= 0.85:
        return ("muito forte", "Muito forte")
    if a >= 0.70:
        return ("forte", "Forte")
    if a >= 0.50:
        return ("moderada", "Moderada")
    if a >= 0.30:
        return ("fraca", "Fraca")
    return ("muito fraca", "Muito fraca")

def frase_relacao(col1: str, col2: str, v: float) -> str:
    sentido = "variam juntas" if v >= 0 else "tendem a ir em direções opostas"
    nivel, _ = classificar_intensidade(v)
    return f"**{col1}** e **{col2}** {sentido} (relação {nivel})."

def gerar_relacoes_simples(df: pd.DataFrame, cols: list, top_n: int, limiar_abs: float):
    if len(cols) < 2:
        return [], False

    corr = df[cols].corr(numeric_only=True)
    pares = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .reset_index()
    )
    pares.columns = ["coluna_1", "coluna_2", "valor"]
    pares["abs"] = pares["valor"].abs()
    pares = pares.sort_values("abs", ascending=False)

    fortes = pares[pares["abs"] >= limiar_abs].head(top_n)
    use = fortes
    fallback = False

    if use.empty and not pares.empty:
        use = pares.head(top_n)
        fallback = True

    out = []
    for _, r in use.iterrows():
        v = float(r["valor"])
        _, badge = classificar_intensidade(v)
        out.append({
            "c1": str(r["coluna_1"]),
            "c2": str(r["coluna_2"]),
            "valor": v,
            "badge": badge,
            "texto": frase_relacao(str(r["coluna_1"]), str(r["coluna_2"]), v),
        })
    return out, fallback

def adicionar_campos_data(df: pd.DataFrame, col_data: str):
    df_work = df.copy()
    s = parse_data_robusta(df_work[col_data])
    invalidas = int(s.isna().sum())

    df_work["Dia"] = s.dt.date.astype(str)
    df_work["Ano"] = s.dt.year.astype("Int64")
    df_work["Número Mês"] = s.dt.month.astype("Int64")
    df_work["Mês"] = df_work["Número Mês"].map(lambda x: MESES_PT.get(int(x), "") if pd.notna(x) else "")

    return df_work, invalidas

def construir_tabela_metricas(df: pd.DataFrame, group_cols: list, metric_cols: list, ops: list, preencher_vazios: bool = True) -> pd.DataFrame:
    df_work = df.copy()

    if preencher_vazios and group_cols:
        for c in group_cols:
            if c not in df_work.columns:
                continue
            if pd.api.types.is_object_dtype(df_work[c]) or pd.api.types.is_string_dtype(df_work[c]) or pd.api.types.is_categorical_dtype(df_work[c]):
                s = normalizar_texto_nulos(df_work[c])
                df_work[c] = s.fillna("Sem informação")

    if not group_cols:
        df_work["_grupo_unico"] = "Total"
        group_cols = ["_grupo_unico"]
        drop_group_unico = True
    else:
        drop_group_unico = False

    gb = df_work.groupby(group_cols, dropna=False)

    partes = []
    if "Total de registros" in ops:
        partes.append(gb.size().rename("Total_registros"))

    if "Distintos (únicos)" in ops and metric_cols:
        for c in metric_cols:
            partes.append(gb[c].nunique(dropna=True).rename(f"{c} - únicos"))

    if "Soma" in ops and metric_cols:
        for c in metric_cols:
            partes.append(gb[c].sum(min_count=1).rename(f"{c} - soma"))

    if "Média" in ops and metric_cols:
        for c in metric_cols:
            partes.append(gb[c].mean().rename(f"{c} - média"))

    if not partes:
        return pd.DataFrame()

    out = pd.concat(partes, axis=1).reset_index()
    if drop_group_unico:
        out = out.drop(columns=["_grupo_unico"], errors="ignore")
    return out

# =========================
# UPLOAD (CARD)
# =========================
with card("📥 Envie seu arquivo", "Arraste e solte ou clique no botão. Suporta .csv e .xlsx"):
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown(
            """
            <span class="chip">Dica</span> Se for CSV brasileiro, pode vir com <b>;</b> como separador.<br/>
            <span class="chip">Dica</span> Planilhas “visuais” (células mescladas, títulos no meio) podem dar resultados ruins.
            """,
            unsafe_allow_html=True
        )
    with right:
        uploaded_file = st.file_uploader("Escolha um arquivo", type=["csv", "xlsx"], label_visibility="collapsed")

st.divider()

if not uploaded_file:
    with card("✨ Pronto para analisar", "Envie um CSV/XLSX acima para gerar diagnóstico e relatórios automaticamente."):
        st.info("Dica: arquivos muito grandes podem demorar. Para versão free, recomendamos começar com uma amostra.")
    st.stop()

status_box = st.empty()
status_box.markdown(
    """
    <div class="win-loader-wrap">
      <div class="win-loader"></div>
      <div style="color: rgba(229,231,235,0.92); font-weight: 650;">
        Carregando arquivo e preparando análise...
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

try:
    df = carregar_arquivo(uploaded_file.getvalue(), uploaded_file.name)
    status_box.success("Arquivo carregado com sucesso!")
except Exception as e:
    status_box.empty()
    with card("❌ Não consegui ler o arquivo", "Confira se é CSV/XLSX válido e tente novamente."):
        st.exception(e)
    st.stop()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Controles")
preview_rows = st.sidebar.slider("Linhas na prévia", 5, 200, 50, 5)
top_nulls = st.sidebar.slider("Top colunas com nulos", 5, 50, 20, 5)

numeric_cols = df.select_dtypes(include="number").columns.tolist()

st.sidebar.header("🧲 Relações (simples)")
top_rel = st.sidebar.slider("Quantidade de relações", 3, 20, 7, 1)
sens = st.sidebar.selectbox(
    "Sensibilidade",
    ["Mostrar só fortes", "Mostrar moderadas e fortes", "Mostrar tudo (inclusive fracas)"],
    index=1
)
if sens == "Mostrar só fortes":
    limiar_abs = 0.70
elif sens == "Mostrar moderadas e fortes":
    limiar_abs = 0.50
else:
    limiar_abs = 0.10

date_cols = detectar_colunas_data(df)
auto_on = bool(date_cols)

st.sidebar.header("🕒 Análise por período")
habilitar_periodo = st.sidebar.checkbox("Ativar análise por período", value=auto_on)

col_data_escolhida = None
gran = "Mês"
modo_periodo = "Contagem"
col_metrica_periodo = None

if habilitar_periodo:
    if not date_cols:
        st.sidebar.warning("Não encontrei colunas de data para esta análise.")
        habilitar_periodo = False
    else:
        col_data_escolhida = st.sidebar.selectbox("Coluna de data", options=date_cols, index=0)
        gran = st.sidebar.selectbox("Granularidade", options=["Mês", "Semana", "Dia"], index=0)

        modo_ui = st.sidebar.selectbox(
            "Métrica do gráfico/tabela",
            options=["Contagem de registros", "Soma (coluna numérica)", "Média (coluna numérica)"],
            index=0
        )
        if modo_ui == "Contagem de registros":
            modo_periodo = "Contagem"
        elif modo_ui == "Soma (coluna numérica)":
            modo_periodo = "Soma"
        else:
            modo_periodo = "Média"

        if modo_periodo in ["Soma", "Média"]:
            if not numeric_cols:
                st.sidebar.warning("Não há colunas numéricas para Soma/Média.")
                modo_periodo = "Contagem"
            else:
                col_metrica_periodo = st.sidebar.selectbox("Coluna numérica", options=numeric_cols, index=0)

st.sidebar.subheader("⚡ Performance (período)")
max_pontos = st.sidebar.slider("Máx. pontos no gráfico", 6, 60, 24, 6)
mostrar_tabela_completa = st.sidebar.checkbox("Mostrar tabela completa (pode pesar)", value=False)

# =========================
# Info geral
# =========================
with card("📌 Informações Gerais", "Visão rápida do arquivo carregado"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linhas", df.shape[0])
    c2.metric("Colunas", df.shape[1])
    c3.metric("Nulos (total)", int(df.isnull().sum().sum()))
    c4.metric("Linhas duplicadas", int(df.duplicated().sum()))

# =========================
# Visão geral por período
# =========================
periodo_para_export = None

if habilitar_periodo and col_data_escolhida:
    tab_periodo, invalidas, label_periodo, col_valor, label_valor = agregar_por_periodo_metric(
        df, col_data_escolhida, gran, modo_periodo, col_metrica_periodo
    )

    subt = f"{label_valor} por {gran.lower()} usando '{col_data_escolhida}'"
    if modo_periodo in ["Soma", "Média"] and col_metrica_periodo:
        subt += f" e coluna '{col_metrica_periodo}'"

    with card(f"📅 Visão geral por {gran.lower()}", subt):
        if invalidas > 0:
            st.warning(f"⚠️ {invalidas} valores em '{col_data_escolhida}' não puderam ser interpretados como data e foram ignorados.")

        if tab_periodo.empty:
            st.info("Não foi possível gerar a agregação (datas inválidas ou métrica sem valores).")
        else:
            # ---------- TABELA (mantém completa; ordena por Ano desc + Mês asc quando for mês) ----------
            if gran == "Mês":
                tab_para_mostrar = formatar_visao_mes_generica(tab_periodo, label_periodo, col_valor, label_valor)
                periodo_para_export = tab_para_mostrar.copy()

                if mostrar_tabela_completa:
                    show_df(tab_para_mostrar, use_container_width=True)
                else:
                    show_df(tab_para_mostrar.head(24), use_container_width=True)

                # ---------- GRÁFICO (somente aqui: padrão último ano + filtro de anos) ----------
                plot_tbl = tab_para_mostrar.dropna(subset=["Ano", "Número Mês"]).copy()
                if plot_tbl.empty:
                    st.info("Sem dados suficientes para montar o gráfico mensal.")
                else:
                    anos_disp = sorted(plot_tbl["Ano"].dropna().astype(int).unique().tolist(), reverse=True)
                    ano_padrao = anos_disp[0] if anos_disp else None

                    colY1, colY2 = st.columns([1.2, 1])
                    with colY1:
                        anos_sel = st.multiselect(
                            "Anos no gráfico",
                            options=anos_disp,
                            default=[ano_padrao] if ano_padrao is not None else []
                        )
                    with colY2:
                        st.caption("Dica: por padrão mostramos só o último ano. Marque outros anos se precisar.")

                    if not anos_sel and ano_padrao is not None:
                        anos_sel = [ano_padrao]

                    plot_f = plot_tbl[plot_tbl["Ano"].isin(anos_sel)].copy()
                    plot_f = plot_f.sort_values(["Ano", "Número Mês"], ascending=[False, True]).reset_index(drop=True)

                    x_labels = plot_f.apply(lambda r: f"{int(r['Ano'])}-{int(r['Número Mês']):02d}", axis=1).tolist()
                    y = plot_f[label_valor].astype(float).tolist()

                    if len(y) == 0:
                        st.info("Sem dados para o(s) ano(s) selecionado(s).")
                    else:
                        # indicadores e insights do gráfico (baseados no que está sendo exibido)
                        media = float(np.mean(y))
                        idx_max = int(np.argmax(y))
                        idx_min = int(np.argmin(y))

                        st.caption(f"Mostrando **{len(y)}** pontos (anos: {', '.join(map(str, anos_sel))}).")

                        st.markdown(
                            f"""
                            **🔎 Insights (do gráfico)**
                            - Maior ponto: **{x_labels[idx_max]}** (**{_fmt_float_pt(y[idx_max], 2)}**)
                            - Menor ponto: **{x_labels[idx_min]}** (**{_fmt_float_pt(y[idx_min], 2)}**)
                            - Média do período exibido: **{_fmt_float_pt(media, 2)}**
                            """
                        )

                        fig, ax = plt.subplots(figsize=(9, 3.6))
                        ax.plot(range(len(x_labels)), y, marker="o", linewidth=2)

                        # tira eixo Y (fica mais clean)
                        ax.set_ylabel("")
                        ax.set_yticks([])
                        ax.spines["left"].set_visible(False)
                        ax.grid(axis="y", alpha=0.15)

                        ax.set_title(f"{label_valor} por mês")
                        ax.set_xlabel("Período")
                        ax.set_xticks(range(len(x_labels)))
                        ax.set_xticklabels(x_labels, rotation=45, ha="right")

                        # linha da média
                        ax.axhline(media, linewidth=1.5, alpha=0.6)
                        ax.annotate(
                            f"Média: {_fmt_float_pt(media, 2)}",
                            (len(x_labels) - 1, media),
                            textcoords="offset points",
                            xytext=(-10, -12),
                            ha="right",
                            fontsize=10,
                            alpha=0.9
                        )

                        # valores em cima dos pontos
                        for i, val in enumerate(y):
                            txt = f"{int(round(val)):,}".replace(",", ".") if modo_periodo == "Contagem" else _fmt_float_pt(val, 2)
                            ax.annotate(txt, (i, val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=10, alpha=0.95)

                        # destaca max/min
                        ax.plot(idx_max, y[idx_max], marker="o", markersize=10, linestyle="None")
                        ax.annotate("Máx", (idx_max, y[idx_max]), textcoords="offset points", xytext=(0, 18), ha="center", fontsize=10, weight="bold")

                        ax.plot(idx_min, y[idx_min], marker="o", markersize=10, linestyle="None")
                        ax.annotate("Mín", (idx_min, y[idx_min]), textcoords="offset points", xytext=(0, 18), ha="center", fontsize=10, weight="bold")

                        # margem visual
                        ymin = min(y)
                        ymax = max(y)
                        margem = (ymax - ymin) * 0.15 if ymax != ymin else (abs(ymax) * 0.15 + 1)
                        ax.set_ylim(ymin - margem, ymax + margem)

                        st.pyplot(fig, clear_figure=True)

            else:
                # não-mensal: mantém o comportamento simples atual (tabela + gráfico)
                tab_para_mostrar = tab_periodo.copy()
                tab_para_mostrar["Período"] = tab_para_mostrar[label_periodo].astype(str)
                tab_para_mostrar = tab_para_mostrar[["Período", "VALOR"]].rename(columns={"VALOR": label_valor})
                periodo_para_export = tab_para_mostrar.copy()

                if mostrar_tabela_completa:
                    show_df(tab_para_mostrar, use_container_width=True)
                else:
                    show_df(tab_para_mostrar.head(max_pontos), use_container_width=True)

                plot_tbl = tab_periodo.head(max_pontos).copy()
                x_labels = plot_tbl[label_periodo].astype(str).tolist()
                y = plot_tbl[col_valor].astype(float).tolist()

                if len(y) > 0:
                    media = float(np.mean(y))
                    idx_max = int(np.argmax(y))
                    idx_min = int(np.argmin(y))

                    st.caption(f"Mostrando últimos **{len(y)}** pontos.")

                    fig, ax = plt.subplots(figsize=(9, 3.6))
                    ax.plot(range(len(x_labels)), y, marker="o", linewidth=2)

                    ax.set_ylabel("")
                    ax.set_yticks([])
                    ax.spines["left"].set_visible(False)
                    ax.grid(axis="y", alpha=0.15)

                    ax.set_title(f"{label_valor} por {gran.lower()}")
                    ax.set_xlabel("Período")
                    ax.set_xticks(range(len(x_labels)))
                    ax.set_xticklabels(x_labels, rotation=45, ha="right")

                    ax.axhline(media, linewidth=1.5, alpha=0.6)

                    for i, val in enumerate(y):
                        txt = f"{int(round(val)):,}".replace(",", ".") if modo_periodo == "Contagem" else _fmt_float_pt(val, 2)
                        ax.annotate(txt, (i, val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=10, alpha=0.95)

                    ax.plot(idx_max, y[idx_max], marker="o", markersize=10, linestyle="None")
                    ax.annotate("Máx", (idx_max, y[idx_max]), textcoords="offset points", xytext=(0, 18), ha="center", fontsize=10, weight="bold")

                    ax.plot(idx_min, y[idx_min], marker="o", markersize=10, linestyle="None")
                    ax.annotate("Mín", (idx_min, y[idx_min]), textcoords="offset points", xytext=(0, 18), ha="center", fontsize=10, weight="bold")

                    ymin = min(y)
                    ymax = max(y)
                    margem = (ymax - ymin) * 0.15 if ymax != ymin else (abs(ymax) * 0.15 + 1)
                    ax.set_ylim(ymin - margem, ymax + margem)

                    st.pyplot(fig, clear_figure=True)
# =========================
# Pré-visualização
# =========================
with card("🔍 Pré-visualização", "Amostra inicial do seu dataset"):
    show_df(df.head(preview_rows), use_container_width=True)

# =========================
# Tipos
# =========================
dtypes_df = pd.DataFrame({"coluna": df.columns, "tipo": [tipo_pt(df, c) for c in df.columns]})
with card("🧬 Tipos de dados", "Tipos normalizados (Número, Data, Data hora, Texto)"):
    show_df(dtypes_df, use_container_width=True)

# =========================
# A) Nulos
# =========================
nulos_df = tabela_nulos(df)
with card("🧻 A) Nulos por coluna", "Onde estão as “lacunas” no seu arquivo"):
    if nulos_df["nulos"].sum() == 0:
        st.info("Nenhuma célula nula encontrada. Quase raro de ver. 🧐")
    else:
        show_df(nulos_df.head(top_nulls), use_container_width=True)

# =========================
# Numéricas
# =========================
with card("📈 Colunas numéricas", "Selecione o que será analisado (estatísticas, outliers e relações)"):
    if not numeric_cols:
        st.warning("Nenhuma coluna numérica encontrada. Sem números, sem estatística.")
        st.stop()

    selecionadas = st.multiselect("Selecione colunas numéricas para análise", numeric_cols, default=numeric_cols)
    if not selecionadas:
        st.warning("Selecione ao menos uma coluna numérica.")
        st.stop()

# =========================
# Resumo estatístico
# =========================
describe_df = df[selecionadas].describe().reset_index().rename(columns={"index": "estatística"})
describe_df = traduzir_estatisticas(describe_df)

with card("📊 Resumo estatístico", "Leitura rápida (simples) + tabela detalhada"):
    col_escolhida = st.selectbox("Escolha a coluna para interpretar", options=selecionadas, index=0)

    st.markdown("**🧠 Interpretação automática (linguagem simples)**")
    st.markdown("\n".join(resumo_leigo_coluna(df, col_escolhida)))

    with st.expander("Ver tabela estatística completa (detalhada)", expanded=False):
        show_df(describe_df, use_container_width=True)

# =========================
# B) Outliers
# =========================
outliers_df = outliers_iqr(df, selecionadas)
with card("🚨 B) Valores fora do padrão (IQR)", "Conta valores muito abaixo/acima do esperado por coluna"):
    st.markdown(texto_outliers_explicacao(df, selecionadas))
    show_df(outliers_df, use_container_width=True)

# =========================
# C) Relações
# =========================
relacoes, usou_fallback = gerar_relacoes_simples(df, selecionadas, top_n=top_rel, limiar_abs=limiar_abs)

with card("🧲 C) Relações encontradas nos dados", "Sem números e sem termos técnicos: só o que vale a pena olhar"):
    if len(selecionadas) < 2:
        st.info("Para encontrar relações, selecione pelo menos **2 colunas numéricas**.")
    elif not relacoes:
        st.info("Não foi possível calcular relações com as colunas selecionadas.")
    else:
        if usou_fallback:
            st.warning("Não encontramos relações fortes. Abaixo estão as **mais próximas** (podem ser fracas).")

        for r in relacoes:
            st.markdown(
                f"""
                <div class="rel-item">
                  <div class="rel-dot"></div>
                  <div>
                    <div class="rel-title">{r["c1"]} ↔ {r["c2"]}
                      <span class="rel-badge">{r["badge"]}</span>
                    </div>
                    <div class="rel-sub">{r["texto"]}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# =========================
# E) Métricas e Categorizações
# =========================
tabela_metricas_final = pd.DataFrame()

with card("🧮 E) Métricas e Categorizações (exploratória)",
          "Escolha o que contar/somar e como agrupar. Ideal para IDs, CPFs, status, programas e períodos."):

    colA, colB = st.columns([1.2, 1])
    with colA:
        group_cols_user = st.multiselect("Agrupar por (uma ou mais colunas)", options=df.columns.tolist(), default=[])

    with colB:
        usar_data = st.checkbox("Adicionar opções de agrupamento por data", value=False)
        col_data_metricas = None
        campos_periodo_escolhidos = []
        df_work = df

        if usar_data:
            date_cols = detectar_colunas_data(df)
            if not date_cols:
                st.warning("Não encontrei colunas de data automaticamente.")
                usar_data = False
            else:
                col_data_metricas = st.selectbox("Coluna de data", options=date_cols, index=0)
                df_work, invalidas_e = adicionar_campos_data(df_work, col_data_metricas)

                campos_periodo_escolhidos = st.multiselect(
                    "Campos de período para agrupar",
                    options=["Ano", "Número Mês", "Mês", "Dia"],
                    default=["Ano", "Número Mês", "Mês"]
                )

    colC, colD = st.columns([1.2, 1])
    with colC:
        metric_cols_user = st.multiselect("Colunas numéricas para métricas (IDs, valores, etc.)", options=numeric_cols, default=[])

    with colD:
        ops_user = st.multiselect(
            "Métricas desejadas",
            options=["Total de registros", "Distintos (únicos)", "Soma", "Média"],
            default=["Total de registros"]
        )

    preencher_vazios = st.checkbox("Trocar vazios/None por 'Sem informação' nas categorias", value=True)

    group_cols_final = list(dict.fromkeys(group_cols_user))
    if usar_data and campos_periodo_escolhidos:
        prioridade = ["Ano", "Número Mês", "Mês", "Dia"]
        campos_periodo_ordenados = [c for c in prioridade if c in campos_periodo_escolhidos]
        for c in campos_periodo_ordenados:
            if c in group_cols_final:
                group_cols_final.remove(c)
        group_cols_final = campos_periodo_ordenados + group_cols_final

    if any(op in ops_user for op in ["Distintos (únicos)", "Soma", "Média"]) and not metric_cols_user:
        ops_user = ["Total de registros"]

    if ops_user:
        tabela_metricas_final = construir_tabela_metricas(
            df=df_work if usar_data else df,
            group_cols=group_cols_final,
            metric_cols=metric_cols_user,
            ops=ops_user,
            preencher_vazios=preencher_vazios
        )

        if not tabela_metricas_final.empty:
            tabela_metricas_final = ordenar_ano_mes_global(tabela_metricas_final)
            show_df(tabela_metricas_final, use_container_width=True)

            st.download_button(
                "Baixar esta tabela (CSV)",
                data=csv_bytes_sorted(tabela_metricas_final),
                file_name="metricas_categorizacoes.csv",
                mime="text/csv"
            )

# =========================
# D) Exportar relatórios
# =========================
info_df = pd.DataFrame([{
    "arquivo": uploaded_file.name,
    "linhas": df.shape[0],
    "colunas": df.shape[1],
    "nulos_total": int(df.isnull().sum().sum()),
    "duplicadas": int(df.duplicated().sum())
}])

relacoes_export = pd.DataFrame([
    {"coluna_1": r["c1"], "coluna_2": r["c2"], "intensidade": r["badge"], "descricao": re.sub(r"\*\*", "", r["texto"])}
    for r in relacoes
])

rel_abas = {
    "info_geral": info_df,
    "tipos": dtypes_df,
    "nulos": nulos_df,
    "describe": describe_df,
    "outliers_iqr": outliers_df,
    "relacoes": relacoes_export
}

if isinstance(periodo_para_export, pd.DataFrame) and not periodo_para_export.empty:
    rel_abas[f"periodo_{gran.lower()}"] = periodo_para_export

if isinstance(tabela_metricas_final, pd.DataFrame) and not tabela_metricas_final.empty:
    rel_abas["metricas_categorizacoes"] = tabela_metricas_final

with card("⬇️ D) Exportar relatórios", "Baixe resultados em CSV (rápido) ou Excel (com abas)"):
    cA, cB, cC, cD = st.columns(4)

    with cA:
        st.download_button("describe.csv", data=to_csv_bytes(describe_df), file_name="describe.csv", mime="text/csv")
    with cB:
        st.download_button("nulos.csv", data=to_csv_bytes(nulos_df), file_name="nulos_por_coluna.csv", mime="text/csv")
    with cC:
        st.download_button("outliers.csv", data=to_csv_bytes(outliers_df), file_name="outliers_iqr.csv", mime="text/csv")
    with cD:
        try:
            excel_bytes = gerar_excel_relatorio(rel_abas)
            st.download_button(
                "relatorio.xlsx",
                data=excel_bytes,
                file_name="relatorio.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.warning("Não consegui gerar Excel. Verifique se o openpyxl está instalado (pip install openpyxl).")
            st.exception(e)

