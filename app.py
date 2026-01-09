import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Indeniza Aí",
    page_icon="✈️",
    layout="centered"
)

# Estilo CSS para dar a cara do "Indeniza Aí"
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #00C853; /* Verde Indeniza Aí */
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
# CARREGAMENTO COM CACHE
# ==============================================================================
@st.cache_resource
def carregar_ia():
    # Usando o modelo BASE para caber na memória do servidor grátis
    # Se você gerou o PKL com o LARGE, idealmente gere de novo com o BASE.
    # Mas o código tenta carregar mesmo assim.
    model_emb = SentenceTransformer("intfloat/multilingual-e5-large")
        
    # Carrega Banco de Dados
    # O arquivo precisa estar na mesma pasta no GitHub
    with open("banco_vetorial_tjpr_e5.pkl", "rb") as f:
        dados = pickle.load(f)
    
    df = dados["dataframe"]
    vetores = dados["vetores"]
    
    # Tratamentos de segurança
    df['valor_dano_moral'] = pd.to_numeric(df['valor_dano_moral'], errors='coerce').fillna(0)
    df['valor_dano_material'] = pd.to_numeric(df.get('valor_dano_material', 0), errors='coerce').fillna(0)
    df['valor_total'] = df['valor_dano_moral'] + df['valor_dano_material']
    df['quem_ganhou'] = df['quem_ganhou'].astype(str).str.lower()
    df['resultado_decisao'] = df['resultado_decisao'].astype(str).str.lower()
    
    return model_emb, df, vetores

# Carregamento seguro
try:
    with st.spinner("Carregando inteligência jurídica..."):
        model_emb, df, vetores_banco = carregar_ia()
except Exception as e:
    st.error(f"Erro técnico ao carregar base: {e}")
    st.stop()

# Configuração da API (Pega dos Segredos do Streamlit Cloud)
api_key = st.secrets.get("OPENROUTER_API_KEY")
client = None
if api_key:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key, 
    )

# ==============================================================================
# INTERFACE
# ==============================================================================
st.markdown("<h1 style='text-align: center;'>✈️ Indeniza Aí</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Descubra a chance de êxito e o valor da sua indenização em segundos.</p>", unsafe_allow_html=True)

queixa = st.text_area(
    "O que aconteceu?",
    placeholder="Ex: Meu voo de Curitiba para SP foi cancelado, perdi uma reunião importante e fiquei 8 horas no aeroporto sem assistência...",
    height=140
)

# Botão de Ação
if st.button("CALCULAR INDENIZAÇÃO"):
    if len(queixa) < 15:
        st.warning("⚠️ Conta mais detalhes pra gente! Pelo menos uma frase completa.")
    else:
        # Validação com LLM (Se a chave estiver configurada)
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
                        messages=[{"role": "user", "content": prompt}]
                    )
                    import json
                    resp = json.loads(check.choices[0].message.content.replace("```json","").replace("```",""))
                    if not resp['valido']:
                        analise_permitida = False
                        motivo_bloqueio = "O texto não parece ser um relato sobre problemas aéreos ou de consumo."
                except:
                    pass # Se a IA falhar, deixa passar (Fail-open)

        if not analise_permitida:
            st.error(f"❌ Ops! {motivo_bloqueio}")
        else:
            with st.spinner("Consultando jurisprudência..."):
                # 1. Busca Semântica
                texto_formatado = f"query: {queixa}"
                vetor_queixa = model_emb.encode([texto_formatado])
                similaridades = cosine_similarity(vetor_queixa, vetores_banco)[0]
                
                indices_top = np.argsort(similaridades)[-30:][::-1]
                amostra = df.iloc[indices_top].copy()
                amostra['similaridade'] = similaridades[indices_top]
                amostra = amostra[amostra['similaridade'] > 0.72]
                
                if len(amostra) < 3:
                    st.warning("⚠️ Caso muito específico. Não encontramos precedentes suficientes.")
                else:
                    # 2. Lógica Advogado do Diabo
                    def classificar(row):
                        ganhador = str(row['quem_ganhou'])
                        decisao = str(row['resultado_decisao'])
                        valor = float(row['valor_total'])
                        
                        termos_derrota = ['improcedente', 'não provido', 'negado', 'indefiro', 'extinto']
                        if 'empresa' in ganhador or 'réu' in ganhador or any(t in decisao for t in termos_derrota):
                            return "DERROTA"
                        
                        termos_vitoria = ['procedente', 'parcialmente', 'provido', 'acolhido']
                        if 'consumidor' in ganhador or 'autor' in ganhador or any(t in decisao for t in termos_vitoria):
                            return "VITORIA_COM_VALOR" if valor > 100 else "VITORIA_OCULTA"
                        
                        return "INDEFINIDO"

                    amostra['categoria'] = amostra.apply(classificar, axis=1)
                    
                    total = len(amostra)
                    vitorias = len(amostra[amostra['categoria'].str.contains("VITORIA")])
                    prob = (vitorias / total) * 100
                    
                    financeiro = amostra[amostra['categoria'].isin(["VITORIA_COM_VALOR", "DERROTA"])].copy()
                    val_esperado = 0
                    teto = 0
                    if len(financeiro) > 0:
                        val_esperado = np.average(financeiro['valor_total'], weights=financeiro['similaridade'])
                        pagos = financeiro[financeiro['valor_total'] > 0]
                        if len(pagos) > 0:
                            teto = pagos['valor_total'].quantile(0.80)

                    # 3. Exibição
                    col1, col2 = st.columns(2)
                    with col1:
                        cor_txt = "#2ecc71" if prob > 70 else "#e74c3c"
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
                    for i, row in amostra.head(3).iterrows():
                        icon = "✅" if "VITORIA" in row['categoria'] else "❌"
                        valor_txt = f"R$ {row['valor_total']:,.2f}" if row['valor_total'] > 0 else "R$ 0,00"
                        with st.expander(f"{icon} {valor_txt} - {row['resultado_decisao'].title()}"):
                            st.caption(f"Semelhança: {row['similaridade']*100:.1f}%")
                            st.write(row['resumo'])

st.markdown("<div style='text-align: center; margin-top: 50px; color: #888;'>Indeniza Aí © 2025 - Beta</div>", unsafe_allow_html=True)
