import streamlit as st
import pandas as pd
import pickle
import numpy as np
import gc
import json
import time
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from datetime import datetime
import streamlit.components.v1 as components

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Indeniza Aí",
    page_icon="⚖️",
    layout="centered"
)

# --- CSS E JAVASCRIPT PARA ANIMAÇÃO (O GIF FALSO) ---
# Atualizado para trocar entre Avião, Dinheiro e Balança a cada 5 segundos
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .stButton>button {
        width: 100%;
        background-color: #00C853; 
        color: white; 
        font-weight: bold;
        border-radius: 10px; height: 50px; font-size: 18px;
    }
    .stButton>button:hover { background-color: #009624; color: white; }
    .metric-card {
        background-color: #ffffff; padding: 20px; border-radius: 10px;
        border: 1px solid #e0e0e0; box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
    h1, h2, h3 { color: #1E1E1E; }
    
    /* ID para o título animado */
    #titulo-animado { transition: all 0.5s ease; }
    </style>
    
    <script>
    // Função que roda no navegador do cliente
    function animarIcone() {
        const icones = ["✈️", "💰", "⚖️"]; // Adicionada a Balança
        let i = 0;
        const target = window.parent.document.querySelector('h1'); 
        if (target) {
            setInterval(() => {
                let texto = target.innerText;
                // Verifica se tem algum dos ícones para substituir
                if (texto.includes("✈️") || texto.includes("💰") || texto.includes("⚖️")) {
                     target.innerText = icones[i] + " Indeniza Aí";
                     i = (i + 1) % icones.length;
                }
            }, 5000); // Alterado para 5 segundos (5000ms)
        }
    }
    setTimeout(animarIcone, 1000);
    </script>
""", unsafe_allow_html=True)

# ==============================================================================
# CARREGAMENTO INTELIGENTE (CARREGA SÓ O NECESSÁRIO)
# ==============================================================================
@st.cache_resource
def carregar_modelos_ia():
    # 1. Carrega Bi-Encoder (Busca Inicial)
    bi_encoder = SentenceTransformer("intfloat/multilingual-e5-large")
    NOME_CROSS_ENCODER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    cross_encoder = CrossEncoder(NOME_CROSS_ENCODER)
    
    return bi_encoder, cross_encoder

@st.cache_resource
def carregar_banco_dados(arquivo_pkl):
    """Carrega o banco específico baseado no tema (Aéreo ou Nome Sujo)"""
    with open(arquivo_pkl, "rb") as f:
        dados = pickle.load(f)
    
    df = dados["dataframe"]
    vetores = dados["vetores"]
    
    # Limpeza de memória imediata
    del dados
    gc.collect()

    # Tratamentos
    df['valor_dano_moral'] = pd.to_numeric(df['valor_dano_moral'], errors='coerce').fillna(0)
    df['valor_dano_material'] = pd.to_numeric(df.get('valor_dano_material', 0), errors='coerce').fillna(0)
    df['valor_total'] = df['valor_dano_moral'] + df['valor_dano_material']
    df['quem_ganhou'] = df['quem_ganhou'].astype(str).str.lower()
    df['resultado_decisao'] = df['resultado_decisao'].astype(str).str.lower()
    df['data_julgamento'] = pd.to_datetime(df['data_julgamento'], format='%d/%m/%Y', errors='coerce')
    
    return df, vetores

# Inicializa Modelos (Ficam na RAM)
try:
    with st.spinner("Carregando Cérebro Jurídico..."):
        bi_encoder, cross_encoder = carregar_modelos_ia()
except Exception as e:
    st.error(f"Erro ao carregar IA: {e}")
    st.stop()

# Configuração da API
api_key = st.secrets.get("OPENROUTER_API_KEY")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key) if api_key else None
ANO_ATUAL = datetime.now().year

# ==============================================================================
# LÓGICA JURIMÉTRICA
# ==============================================================================
def calcular_peso_temporal(data_julg):
    if pd.isna(data_julg): return 0.5
    delta = max(0, ANO_ATUAL - data_julg.year)
    peso = 1 / (delta + 1)
    if delta == 0: peso = 1.2
    return peso

def classificar_desfecho(row):
    texto_rico = str(row.get('texto_para_embedding', '')).lower()
    val_total = float(row.get('valor_total', 0))
    
    if "concedida/mantida" in texto_rico: return "VITORIA_OCULTA"
    if "improcedente/negado" in texto_rico: return "DERROTA"
    if val_total > 100: return "VITORIA_COM_VALOR"
    return "DERROTA"

# ==============================================================================
# INTERFACE
# ==============================================================================
# Título com ID para o JavaScript achar (Começa com a balança)
st.markdown("<h1 id='titulo-animado' style='text-align: center;'>⚖️ Indeniza Aí</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Descubra suas chances em casos de <b>Voo</b>, <b>Nome Negativado</b> e outros.</p>", unsafe_allow_html=True)

# Placeholder dinâmico
exemplos = "Ex: 'Voo cancelado em Guarulhos...' OU 'Meu nome foi para o Serasa indevidamente...'"
queixa = st.text_area("Relate seu caso:", placeholder=exemplos, height=140)

if st.button("CALCULAR INDENIZAÇÃO"):
    if len(queixa) < 15:
        st.warning("⚠️ Descreva melhor o ocorrido (mínimo 15 caracteres).")
    elif not client:
        st.error("Erro de configuração: API Key não encontrada.")
    else:
        # --- 1. CLASSIFICAÇÃO E ROTEAMENTO (ROUTER) ---
        categoria = None
        arquivo_alvo = None
        analise_permitida = False
        motivo_bloqueio = ""

        with st.spinner("🤖 A IA está analisando seu caso..."):
            try:
                # Prompt Router
                prompt = f"""
                Analise o texto: "{queixa[:500]}".
                Classifique em JSON:
                {{
                    "categoria": "AEREO" (voo, bagagem, atraso) ou "NOMESUJO" (serasa, spc, cobranca indevida, banco) ou "OUTROS",
                    "valido": true/false (false se for spam, oi, teste, receita)
                }}
                """
                check = client.chat.completions.create(
                    model="xiaomi/mimo-v2-flash:free",
                    messages=[{"role": "user", "content": prompt}], temperature=0
                )
                resp = json.loads(check.choices[0].message.content.replace("```json","").replace("```","").strip())
                
                if not resp['valido']:
                    motivo_bloqueio = "Texto inválido ou fora da área jurídica."
                elif resp['categoria'] == 'AEREO':
                    categoria = "AEREO"
                    arquivo_alvo = "banco_vetorial_tjpr_e5.pkl" 
                    analise_permitida = True
                elif resp['categoria'] == 'NOMESUJO':
                    categoria = "NOMESUJO"
                    arquivo_alvo = "banco_nome_sujo.pkl"
                    analise_permitida = True
                else:
                    motivo_bloqueio = "No momento só analisamos casos Aéreos ou de Negativação Indevida."
            
            except Exception as e:
                st.error(f"Erro na classificação: {e}")

        # --- 2. EXECUÇÃO DA ANÁLISE ---
        if not analise_permitida:
            st.error(f"❌ {motivo_bloqueio}")
        else:
            try:
                # Carrega o banco correto dinamicamente
                nome_base = "Direito Aéreo" if categoria == "AEREO" else "Direito Bancário/Consumidor"
                st.toast(f"📂 Consultando base: {nome_base}")
                
                with st.spinner(f"Pesquisando jurisprudência para {nome_base}..."):
                    df, vetores_banco = carregar_banco_dados(arquivo_alvo)
                    
                    # Busca Vetorial
                    vetor_queixa = bi_encoder.encode([f"query: {queixa}"])
                    simil = cosine_similarity(vetor_queixa, vetores_banco)[0]
                    indices = np.argsort(simil)[-60:][::-1]
                    candidatos = df.iloc[indices].copy()
                    
                    # Reranking
                    pares = [[queixa, txt.replace("passage:", "").strip()] for txt in candidatos['texto_para_embedding']]
                    candidatos['score_ia'] = cross_encoder.predict(pares)
                    finais = candidatos.sort_values('score_ia', ascending=False).head(20).copy()
                    
                    # Cálculos
                    finais['peso_tempo'] = finais['data_julgamento'].apply(calcular_peso_temporal)
                    finais['peso_final'] = np.exp(finais['score_ia']) * finais['peso_tempo']
                    finais['categoria'] = finais.apply(classificar_desfecho, axis=1)
                    
                    # Estatística Final
                    vitorias = len(finais[finais['categoria'].str.contains("VITORIA")])
                    prob = min((vitorias / 20) * 100, 95.0) # Trava legal 95%
                    
                    financeiro = finais[finais['categoria'].isin(["VITORIA_COM_VALOR", "DERROTA"])].copy()
                    val_esperado = 0
                    teto = 0
                    if len(financeiro) > 0:
                        val_esperado = np.average(financeiro['valor_total'], weights=financeiro['peso_final'])
                        pagos = financeiro[financeiro['valor_total'] > 0]
                        if len(pagos) > 0: teto = pagos['valor_total'].quantile(0.85)

                    # --- EXIBIÇÃO DE RESULTADOS ---
                    col1, col2 = st.columns(2)
                    with col1:
                        cor = "#2ecc71" if prob > 70 else "#e74c3c" if prob < 40 else "#f1c40f"
                        st.markdown(f"<div class='metric-card'><h3>Probabilidade</h3><h1 style='color:{cor}'>{prob:.0f}%</h1></div>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<div class='metric-card'><h3>Valor Estimado</h3><h1 style='color:#2c3e50'>R$ {val_esperado:,.0f}</h1></div>", unsafe_allow_html=True)
                    
                    if teto > 0 and prob > 50:
                        st.info(f"🚀 **Potencial Otimista:** Em casos de sucesso total, o valor pode chegar a **R$ {teto:,.2f}**.")
                    
                    st.divider()
                    st.subheader(f"📚 Casos Semelhantes ({nome_base})")
                    
                    for i, row in finais.head(3).iterrows():
                        cat = row['categoria']
                        val = row['valor_total']
                        
                        if cat == "VITORIA_COM_VALOR": icon, msg = "✅", f"R$ {val:,.2f}"
                        elif cat == "VITORIA_OCULTA": icon, msg = "✅", "Ganhou (Valor ñ informado)"
                        elif cat == "DERROTA": icon, msg = "❌", "Perdeu (R$ 0,00)"
                        else: icon, msg = "❓", "Indefinido"
                        
                        data_fmt = row['data_julgamento'].strftime('%d/%m/%Y') if pd.notnull(row['data_julgamento']) else ""
                        
                        # Link do Acórdão (Se existir na base nova)
                        link_html = ""
                        if 'link_acordao' in row and str(row['link_acordao']).startswith("http"):
                            link_html = f" | <a href='{row['link_acordao']}' target='_blank'>📄 Ler Acórdão</a>"

                        with st.expander(f"{icon} {msg} | Data: {data_fmt}"):
                            st.markdown(f"**Decisão:** {row['resultado_decisao'].title()}{link_html}", unsafe_allow_html=True)
                            st.caption(f"Relevância IA: {row['score_ia']:.2f}")
                            st.write(row['resumo'])

            except FileNotFoundError:
                st.error(f"⚠️ O Banco de Dados '{arquivo_alvo}' não foi encontrado. Você gerou ele com a fábrica?")
            except Exception as e:
                st.error(f"Erro técnico: {e}")

st.markdown("<div style='text-align: center; margin-top: 50px; color: #888;'>Indeniza Aí © 2025 - Beta</div>", unsafe_allow_html=True)