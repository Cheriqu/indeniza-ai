import streamlit as st
import pandas as pd
import pickle
import numpy as np
import gc
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from datetime import datetime
import json

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Indeniza Aí",
    page_icon="✈️",
    layout="centered"
)

# Estilo CSS 
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #00C853; 
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #009624;
        color: white;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
    h1, h2, h3 {
        color: #1E1E1E;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# CARREGAMENTO COM CACHE (Bi-Encoder + Cross-Encoder)
# ==============================================================================
@st.cache_resource
def carregar_ia():
    # 1. Carrega Bi-Encoder (Busca Inicial)
    # Usa o Large para bater com o seu arquivo .pkl gerado
    bi_encoder = SentenceTransformer("intfloat/multilingual-e5-large")
    
    # 2. Carrega Cross-Encoder (O Juiz Digital)
    # Usando Unicamp-DL para evitar erros de repositório
    cross_encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    # 3. Carrega Banco de Dados
    with open("banco_vetorial_tjpr_e5.pkl", "rb") as f:
        dados = pickle.load(f)
    
    df = dados["dataframe"]
    vetores = dados["vetores"]

    del dados
    gc.collect()

    # Tratamentos de Dados
    df['valor_dano_moral'] = pd.to_numeric(df['valor_dano_moral'], errors='coerce').fillna(0)
    df['valor_dano_material'] = pd.to_numeric(df.get('valor_dano_material', 0), errors='coerce').fillna(0)
    df['valor_total'] = df['valor_dano_moral'] + df['valor_dano_material']
    df['quem_ganhou'] = df['quem_ganhou'].astype(str).str.lower()
    df['resultado_decisao'] = df['resultado_decisao'].astype(str).str.lower()
    df['data_julgamento'] = pd.to_datetime(df['data_julgamento'], format='%d/%m/%Y', errors='coerce')
    
    return bi_encoder, cross_encoder, df, vetores

# Carregamento seguro na interface
try:
    with st.spinner("Inicializando Inteligência Jurídica (Isso pode levar alguns segundos)..."):
        bi_encoder, cross_encoder, df, vetores_banco = carregar_ia()
except Exception as e:
    st.error(f"Erro técnico ao carregar base: {e}")
    st.stop()

# Configuração da API (Segredo)
api_key = st.secrets.get("OPENROUTER_API_KEY")
client = None
if api_key:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key, 
    )

ANO_ATUAL = datetime.now().year

# ==============================================================================
# FUNÇÕES LÓGICAS (IGUAIS AO SIMULADOR)
# ==============================================================================
def calcular_peso_temporal(data_julg):
    if pd.isna(data_julg): return 0.5
    
    delta = max(0, ANO_ATUAL - data_julg.year)
    peso = 1 / (delta + 1)
    if delta == 0: peso = 1.2
    return peso

def classificar_desfecho(row):
    # Olha para o texto enriquecido pelo embedding, que é mais confiável
    texto_rico = str(row.get('texto_para_embedding', '')).lower()
    val_total = float(row.get('valor_total', 0))
    
    if "concedida/mantida" in texto_rico: return "VITORIA_OCULTA"
    if "improcedente/negado" in texto_rico: return "DERROTA"
    if val_total > 100: return "VITORIA_COM_VALOR"
    
    return "DERROTA"

# ==============================================================================
# INTERFACE
# ==============================================================================
st.markdown("<h1 style='text-align: center;'>✈️ Indeniza Aí</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Descubra a chance de êxito e o valor da sua indenização em segundos.</p>", unsafe_allow_html=True)

queixa = st.text_area(
    "O que aconteceu?",
    placeholder="Ex: Tive minha mala extraviada em voo internacional, fiquei 3 dias sem roupas...",
    height=140
)

if st.button("CALCULAR INDENIZAÇÃO"):
    if len(queixa) < 15:
        st.warning("⚠️ Conta mais detalhes pra gente! Pelo menos uma frase completa.")
    else:
        # --- 1. VALIDAÇÃO (PORTEIRO) ---
        analise_permitida = True
        motivo_bloqueio = ""

        if client:
            with st.spinner("Validando seu relato..."):
                try:
                    prompt = f"""Responda APENAS JSON. Texto: "{queixa[:400]}". 
                    É um relato de problema de consumo/aéreo/jurídico? 
                    {{"valido": true}} ou {{"valido": false}}"""
                    
                    check = client.chat.completions.create(
                        model="xiaomi/mimo-v2-flash:free",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0
                    )
                    content = check.choices[0].message.content.replace("```json","").replace("```","").strip()
                    resp = json.loads(content)
                    if not resp['valido']:
                        analise_permitida = False
                        motivo_bloqueio = "O texto não parece ser um relato sobre problemas aéreos ou de consumo."
                except:
                    pass # Fail-open

        if not analise_permitida:
            st.error(f"❌ Ops! {motivo_bloqueio}")
        else:
            with st.spinner("Consultando jurisprudência e calculando probabilidades..."):
                
                # --- 2. BUSCA VETORIAL (TOP 60) ---
                texto_formatado = f"query: {queixa}"
                vetor_queixa = bi_encoder.encode([texto_formatado])
                similaridades = cosine_similarity(vetor_queixa, vetores_banco)[0]
                
                N_BUSCA = 60
                indices_top = np.argsort(similaridades)[-N_BUSCA:][::-1]
                candidatos = df.iloc[indices_top].copy()
                
                # --- 3. RERANKING (TOP 20) ---
                pares = [[queixa, txt.replace("passage:", "").strip()] for txt in candidatos['texto_para_embedding']]
                scores = cross_encoder.predict(pares)
                candidatos['score_ia'] = scores
                
                N_FINAL = 20
                finais = candidatos.sort_values('score_ia', ascending=False).head(N_FINAL).copy()
                
                # --- 4. CÁLCULO PONDERADO ---
                finais['peso_tempo'] = finais['data_julgamento'].apply(calcular_peso_temporal)
                finais['peso_final'] = np.exp(finais['score_ia']) * finais['peso_tempo']
                finais['categoria'] = finais.apply(classificar_desfecho, axis=1)
                
                # Estatísticas
                total = len(finais)
                vitorias = len(finais[finais['categoria'].str.contains("VITORIA")])
                prob = (vitorias / total) * 100
                
                # TRAVA LEGAL: Máximo 95%
                if prob > 95: prob = 95.0
                
                financeiro = finais[finais['categoria'].isin(["VITORIA_COM_VALOR", "DERROTA"])].copy()
                val_esperado = 0
                teto = 0
                if len(financeiro) > 0:
                    val_esperado = np.average(financeiro['valor_total'], weights=financeiro['peso_final'])
                    pagos = financeiro[financeiro['valor_total'] > 0]
                    if len(pagos) > 0:
                        teto = pagos['valor_total'].quantile(0.85) # Teto Otimista

                # --- 5. EXIBIÇÃO ---
                col1, col2 = st.columns(2)
                with col1:
                    cor_txt = "#2ecc71" if prob > 70 else "#e74c3c" if prob < 40 else "#f1c40f"
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>Probabilidade</h3>
                            <h1 style="color:{cor_txt}">{prob:.0f}%</h1>
                        </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>Valor Estimado</h3>
                            <h1 style="color:#2c3e50">R$ {val_esperado:,.0f}</h1>
                        </div>
                    """, unsafe_allow_html=True)
                
                if teto > 0 and prob > 50:
                    st.info(f"🚀 **Potencial Otimista:** Em casos de sucesso total, o valor pode chegar a **R$ {teto:,.2f}**.")
                
                st.divider()
                st.subheader("📚 Casos Reais Parecidos")
                st.caption(f"Baseado nos {N_FINAL} casos mais relevantes e recentes encontrados.")

                for i, row in finais.head(3).iterrows():
                    cat = row['categoria']
                    val = row['valor_total']
                    
                    # Lógica de Exibição Visual
                    if cat == "VITORIA_COM_VALOR":
                        icon = "✅"
                        msg_val = f"R$ {val:,.2f}"
                    elif cat == "VITORIA_OCULTA":
                        icon = "✅"
                        msg_val = "Ganhou (Valor não informado)"
                    elif cat == "DERROTA":
                        icon = "❌"
                        msg_val = "Perdeu (R$ 0,00)"
                    else:
                        icon = "❓"
                        msg_val = "Indefinido"
                    
                    data_fmt = row['data_julgamento'].strftime('%d/%m/%Y') if not pd.isna(row['data_julgamento']) else ""
                    
                    with st.expander(f"{icon} {msg_val} | Data: {data_fmt}"):
                        st.caption(f"Relevância IA: {row['score_ia']:.2f}")
                        st.write(row['resumo'])

st.markdown("<div style='text-align: center; margin-top: 50px; color: #888;'>Indeniza Aí © 2025 - Beta</div>", unsafe_allow_html=True)
