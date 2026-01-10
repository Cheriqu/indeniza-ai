import streamlit as st
import pandas as pd
import pickle
import numpy as np
import gc
import json
import time
import sqlite3
import hashlib
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from datetime import datetime

# ==============================================================================
# CONFIGURAÇÃO GERAL E CSS
# ==============================================================================
st.set_page_config(
    page_title="Indeniza Aí",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Estilo Profissional (Paywall + Blur + Cards)
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    
    /* Cards de Métricas */
    .metric-card {
        background-color: #ffffff; padding: 20px; border-radius: 12px;
        border: 1px solid #e0e0e0; box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
        text-align: center; margin-bottom: 20px;
    }
    
    /* Efeito de Borrão (Blur) para o Paywall */
    .blur-content {
        filter: blur(5px);
        pointer-events: none;
        user-select: none;
        opacity: 0.6;
    }
    
    /* Caixa do Paywall */
    .paywall-overlay {
        background: linear-gradient(135deg, #ffffff 0%, #f0f2f5 100%);
        border: 2px solid #00C853;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin-top: -20px;
        position: relative;
        z-index: 999;
        box-shadow: 0px 10px 30px rgba(0, 200, 83, 0.15);
    }
    
    /* Botões */
    .stButton>button {
        width: 100%; border-radius: 8px; height: 50px; font-weight: bold;
    }
    
    h1, h2, h3 { color: #2c3e50; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# BANCO DE DADOS (SQLite Local)
# ==============================================================================
def init_db():
    """Cria o banco de dados e tabelas se não existirem"""
    conn = sqlite3.connect('indeniza.db')
    c = conn.cursor()
    
    # Tabela de Clientes (Leads)
    c.execute('''CREATE TABLE IF NOT EXISTS leads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_registro DATETIME DEFAULT CURRENT_TIMESTAMP,
                    nome TEXT,
                    whatsapp TEXT,
                    cidade TEXT,
                    resumo_caso TEXT,
                    categoria TEXT,
                    probabilidade REAL,
                    valor_estimado REAL,
                    pagou BOOLEAN DEFAULT 0,
                    aceita_advogado BOOLEAN DEFAULT 0
                )''')
    
    # Tabela de Advogados (Login Simples)
    c.execute('''CREATE TABLE IF NOT EXISTS advogados (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE,
                    senha TEXT,
                    nome TEXT,
                    ativo BOOLEAN DEFAULT 1
                )''')
    
    # Cria um advogado admin padrão se não existir (Email: admin / Senha: admin)
    try:
        c.execute("INSERT INTO advogados (email, senha, nome) VALUES (?, ?, ?)", 
                 ('admin', 'admin', 'Administrador'))
    except sqlite3.IntegrityError:
        pass
        
    conn.commit()
    conn.close()

def salvar_lead(dados):
    conn = sqlite3.connect('indeniza.db')
    c = conn.cursor()
    c.execute('''INSERT INTO leads (nome, whatsapp, cidade, resumo_caso, categoria, probabilidade, valor_estimado, pagou, aceita_advogado)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                 (dados['nome'], dados['whatsapp'], dados['cidade'], dados['resumo'], 
                  dados['categoria'], dados['prob'], dados['valor'], dados['pagou'], dados['aceita_advogado']))
    conn.commit()
    conn.close()

def listar_leads():
    conn = sqlite3.connect('indeniza.db')
    df = pd.read_sql_query("SELECT * FROM leads ORDER BY data_registro DESC", conn)
    conn.close()
    return df

def verificar_login_advogado(email, senha):
    conn = sqlite3.connect('indeniza.db')
    c = conn.cursor()
    c.execute("SELECT nome FROM advogados WHERE email = ? AND senha = ?", (email, senha))
    result = c.fetchone()
    conn.close()
    return result

# Inicializa o banco ao abrir
init_db()

# ==============================================================================
# CARREGAMENTO DE IA (Igual ao anterior)
# ==============================================================================
@st.cache_resource
def carregar_modelos_ia():
    bi_encoder = SentenceTransformer("intfloat/multilingual-e5-large")
    cross_encoder = CrossEncoder("cross-encoder/mmarco-mMiniLM-v2-L12-H384-v1")
    return bi_encoder, cross_encoder

@st.cache_resource
def carregar_banco_dados(arquivo_pkl):
    try:
        with open(arquivo_pkl, "rb") as f:
            dados = pickle.load(f)
        df = dados["dataframe"]
        vetores = dados["vetores"]
        del dados
        gc.collect()
        
        # Tratamento de dados
        df['valor_dano_moral'] = pd.to_numeric(df['valor_dano_moral'], errors='coerce').fillna(0)
        df['valor_dano_material'] = pd.to_numeric(df.get('valor_dano_material', 0), errors='coerce').fillna(0)
        df['valor_total'] = df['valor_dano_moral'] + df['valor_dano_material']
        df['quem_ganhou'] = df['quem_ganhou'].astype(str).str.lower()
        df['resultado_decisao'] = df['resultado_decisao'].astype(str).str.lower()
        df['data_julgamento'] = pd.to_datetime(df['data_julgamento'], format='%d/%m/%Y', errors='coerce')
        return df, vetores
    except FileNotFoundError:
        return None, None

try:
    with st.spinner("Carregando Cérebro Jurídico..."):
        bi_encoder, cross_encoder = carregar_modelos_ia()
except Exception as e:
    st.error(f"Erro crítico na IA: {e}")
    st.stop()

api_key = st.secrets.get("OPENROUTER_API_KEY")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key) if api_key else None
ANO_ATUAL = datetime.now().year

# ==============================================================================
# FUNÇÕES DE CÁLCULO
# ==============================================================================
def calcular_peso_temporal(data_julg):
    if pd.isna(data_julg): return 0.5
    delta = max(0, ANO_ATUAL - data_julg.year)
    return 1 / (delta + 1) if delta > 0 else 1.2

def classificar_desfecho(row):
    quem = str(row.get('quem_ganhou', '')).lower()
    if 'consumidor' in quem or 'autor' in quem:
        return "VITORIA_COM_VALOR" if float(row.get('valor_total', 0)) > 100 else "VITORIA_OCULTA"
    if 'empresa' in quem or 'réu' in quem: return "DERROTA"
    
    txt = str(row.get('texto_para_embedding', '')).lower()
    if "improcedente" in txt: return "DERROTA"
    return "VITORIA_COM_VALOR" if float(row.get('valor_total', 0)) > 100 else "DERROTA"

# ==============================================================================
# INTERFACE PRINCIPAL
# ==============================================================================
# Menu Lateral para Advogados
with st.sidebar:
    st.title("Área Restrita 🔒")
    menu = st.radio("Navegação", ["Sou Cliente", "Sou Advogado"])

# --- PÁGINA DO CLIENTE ---
if menu == "Sou Cliente":
    st.markdown("<h1 style='text-align: center;'>⚖️ Indeniza Aí</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Analise seu caso com IA e descubra quanto você pode ganhar.</p>", unsafe_allow_html=True)

    # Controle de Estado da Sessão (Para manter dados após clicar em pagar)
    if 'analise_concluida' not in st.session_state: st.session_state.analise_concluida = False
    if 'dados_finais' not in st.session_state: st.session_state.dados_finais = {}
    if 'pagamento_aprovado' not in st.session_state: st.session_state.pagamento_aprovado = False

    queixa = st.text_area("O que aconteceu?", placeholder="Ex: Voo cancelado em Guarulhos, meu nome foi negativado indevidamente...", height=120)

    if st.button("CALCULAR PROBABILIDADE GRATUITAMENTE"):
        if len(queixa) < 15:
            st.warning("⚠️ Descreva melhor o caso.")
        elif not client:
            st.error("Erro: API Key não configurada.")
        else:
            with st.spinner("🤖 A IA está lendo jurisprudências..."):
                # 1. Classificação
                try:
                    prompt = f'Analise: "{queixa[:500]}". JSON output: {{"categoria": "AEREO" ou "NOMESUJO" ou "OUTROS", "valido": true/false}}'
                    check = client.chat.completions.create(model="xiaomi/mimo-v2-flash:free", messages=[{"role": "user", "content": prompt}], temperature=0)
                    resp = json.loads(check.choices[0].message.content.replace("```json","").replace("```","").strip())
                    
                    if not resp['valido']:
                        st.error("Texto inválido.")
                        st.stop()
                    
                    arquivo_alvo = "banco_vetorial_tjpr_e5.pkl" if resp['categoria'] == 'AEREO' else "banco_nome_sujo.pkl"
                    
                    # 2. Busca e Cálculo
                    df, vetores = carregar_banco_dados(arquivo_alvo)
                    if df is None:
                        st.error(f"Banco de dados '{arquivo_alvo}' não encontrado.")
                        st.stop()
                        
                    vetor = bi_encoder.encode([f"query: {queixa}"])
                    simil = cosine_similarity(vetor, vetores)[0]
                    candidatos = df.iloc[np.argsort(simil)[-60:][::-1]].copy()
                    
                    pares = [[queixa, txt.replace("passage:", "").strip()] for txt in candidatos['texto_para_embedding']]
                    candidatos['score_ia'] = cross_encoder.predict(pares)
                    finais = candidatos.sort_values('score_ia', ascending=False).head(20).copy()
                    
                    finais['categoria'] = finais.apply(classificar_desfecho, axis=1)
                    finais['peso_final'] = np.exp(finais['score_ia']) * finais['data_julgamento'].apply(calcular_peso_temporal)
                    
                    vitorias = len(finais[finais['categoria'].str.contains("VITORIA")])
                    prob = min((vitorias / 20) * 100, 95.0)
                    
                    financeiro = finais[finais['categoria'].isin(["VITORIA_COM_VALOR", "DERROTA"])].copy()
                    val_esperado = 0
                    if len(financeiro) > 0:
                        val_esperado = np.average(financeiro['valor_total'], weights=financeiro['peso_final'])
                    
                    # Salva na sessão
                    st.session_state.dados_finais = {
                        "prob": prob, "valor": val_esperado, "df_finais": finais, 
                        "categoria": resp['categoria'], "resumo": queixa
                    }
                    st.session_state.analise_concluida = True
                    st.rerun() # Recarrega para mostrar resultados
                    
                except Exception as e:
                    st.error(f"Erro: {e}")

    # --- TELA DE RESULTADOS ---
    if st.session_state.analise_concluida:
        dados = st.session_state.dados_finais
        
        # 1. Probabilidade (Grátis - Always Visible)
        cor_prob = "#2ecc71" if dados['prob'] > 70 else "#f1c40f"
        st.markdown(f"""
            <div class='metric-card' style='border-left: 5px solid {cor_prob};'>
                <h3>Probabilidade de Vitória</h3>
                <h1 style='color: {cor_prob}; font-size: 3rem;'>{dados['prob']:.0f}%</h1>
                <p>Baseado em {len(dados['df_finais'])} casos analisados</p>
            </div>
        """, unsafe_allow_html=True)

        # 2. Área Protegida (Paywall ou Desbloqueado)
        if not st.session_state.pagamento_aprovado:
            # === MODO BLOQUEADO ===
            st.markdown("<div class='blur-content'>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1: st.markdown("<div class='metric-card'><h3>Valor Estimado</h3><h1>R$ 5.800,00</h1></div>", unsafe_allow_html=True)
            with col2: st.markdown("<div class='metric-card'><h3>Tempo Médio</h3><h1>12 Meses</h1></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True) # Fim do Blur
            
            # Caixa de Conversão
            with st.container():
                st.markdown("""
                <div class='paywall-overlay'>
                    <h2>🔓 Desbloqueie seu Relatório Completo</h2>
                    <p>Veja o valor exato da indenização, casos semelhantes e receba contato de advogados parceiros.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Formulário de Lead
                with st.form("form_pagamento"):
                    c_nome = st.text_input("Seu Nome Completo")
                    c_whats = st.text_input("Seu WhatsApp (com DDD)")
                    c_cidade = st.text_input("Sua Cidade/Estado")
                    c_check = st.checkbox("Aceito que advogados parceiros entrem em contato comigo.")
                    
                    # Opções de Pagamento
                    st.divider()
                    st.write("Pagamento Único: **R$ 9,90**")
                    tipo_pag = st.radio("Forma de Pagamento:", ["PIX (Instantâneo)", "Bitcoin (Lightning)"])
                    
                    if st.form_submit_button("LIBERAR RELATÓRIO AGORA"):
                        if len(c_nome) > 3 and len(c_whats) > 8:
                            # Salva Lead no SQLite
                            salvar_lead({
                                "nome": c_nome, "whatsapp": c_whats, "cidade": c_cidade,
                                "resumo": dados['resumo'], "categoria": dados['categoria'],
                                "prob": dados['prob'], "valor": dados['valor'],
                                "pagou": False, "aceita_advogado": c_check
                            })
                            st.success("Dados salvos! Redirecionando para pagamento...")
                            time.sleep(1)
                            st.session_state.aguardando_pagamento = True
                            st.rerun()
                        else:
                            st.warning("Preencha nome e WhatsApp corretamente.")
                
                # Simulação de Pagamento (Isso sairia na versão final)
                if st.session_state.get('aguardando_pagamento'):
                    st.info("ℹ️ Aqui apareceria o QR Code do PIX ou Bitcoin.")
                    if st.button("SIMULAR: Pagamento Aprovado pelo Banco"):
                        st.session_state.pagamento_aprovado = True
                        # Atualiza no banco que pagou (pegando o ultimo ID seria o ideal, aqui simplificado)
                        st.rerun()

        else:
            # === MODO DESBLOQUEADO (PREMIUM) ===
            st.balloons()
            st.success("✅ Relatório Desbloqueado com Sucesso!")
            
            # Mostra o Valor Real
            st.markdown(f"""
                <div class='metric-card' style='background-color: #e8f5e9; border: 2px solid #2ecc71;'>
                    <h3>💰 Valor Estimado da Indenização</h3>
                    <h1 style='color: #1b5e20; font-size: 3.5rem;'>R$ {dados['valor']:,.2f}</h1>
                    <p>Valores baseados na jurisprudência recente.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            st.subheader("📚 Jurisprudência Encontrada")
            
            for i, row in dados['df_finais'].head(5).iterrows():
                icon = "✅" if "VITORIA" in row['categoria'] else "❌"
                val_txt = f"R$ {row['valor_total']:,.2f}" if row['valor_total'] > 0 else "Valor não inf."
                
                with st.expander(f"{icon} {val_txt} | {row['quem_ganhou'].title()}"):
                    st.write(f"**Resumo:** {row['resumo']}")
                    st.caption(f"Decisão: {row['resultado_decisao']} | Data: {row['data_julgamento']}")
                    if row.get('link_acordao'):
                        st.markdown(f"[📄 Ler Documento Original]({row['link_acordao']})")

# --- PÁGINA DO ADVOGADO ---
elif menu == "Sou Advogado":
    st.header("Painel do Advogado Parceiro ⚖️")
    
    if 'advogado_logado' not in st.session_state:
        # Login
        with st.form("login_adv"):
            email = st.text_input("E-mail")
            senha = st.text_input("Senha", type="password")
            if st.form_submit_button("Entrar"):
                user = verificar_login_advogado(email, senha)
                if user:
                    st.session_state.advogado_logado = user[0]
                    st.rerun()
                else:
                    st.error("Acesso negado. Tente email: admin / senha: admin")
    else:
        # Dashboard
        st.write(f"Bem-vindo, Dr(a). {st.session_state.advogado_logado}!")
        if st.button("Sair"):
            del st.session_state.advogado_logado
            st.rerun()
            
        st.divider()
        st.subheader("🔥 Leads Recentes (Clientes Potenciais)")
        
        df_leads = listar_leads()
        if not df_leads.empty:
            # Filtros
            cidade_filter = st.multiselect("Filtrar por Cidade", df_leads['cidade'].unique())
            if cidade_filter:
                df_leads = df_leads[df_leads['cidade'].isin(cidade_filter)]
            
            for i, row in df_leads.iterrows():
                with st.container(border=True):
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c1:
                        st.markdown(f"**{row['nome']}**")
                        st.caption(f"{row['cidade']} | {row['data_registro']}")
                    with c2:
                        st.info(f"Prob: {row['probabilidade']:.0f}% | Est: R$ {row['valor_estimado']:,.0f}")
                        st.write(f"Caso: {row['resumo_caso'][:100]}...")
                    with c3:
                        if row['aceita_advogado']:
                            st.link_button("Chamar no WhatsApp", f"https://wa.me/55{row['whatsapp'].replace(' ','').replace('-','')}")
                        else:
                            st.caption("Contato não autorizado")
        else:
            st.info("Nenhum lead cadastrado ainda.")

# Rodapé
st.markdown("<br><hr><center style='color:#ccc'>Indeniza Aí © 2026</center>", unsafe_allow_html=True)