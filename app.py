import streamlit as st
import pandas as pd
import pickle
import numpy as np
import gc
import json
import time
import sqlite3
import mercadopago  # BIBLIOTECA DE PAGAMENTO
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
        filter: blur(8px);
        pointer-events: none;
        user-select: none;
        opacity: 0.5;
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
                    email TEXT,
                    whatsapp TEXT,
                    cidade TEXT,
                    resumo_caso TEXT,
                    categoria TEXT,
                    probabilidade REAL,
                    valor_estimado REAL,
                    pagou BOOLEAN DEFAULT 0,
                    payment_id TEXT,
                    aceita_advogado BOOLEAN DEFAULT 0
                )''')
    
    # Tabela de Advogados
    c.execute('''CREATE TABLE IF NOT EXISTS advogados (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE,
                    senha TEXT,
                    nome TEXT,
                    ativo BOOLEAN DEFAULT 1
                )''')
    
    # Cria Admin padrão
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
    c.execute('''INSERT INTO leads (nome, email, whatsapp, cidade, resumo_caso, categoria, probabilidade, valor_estimado, pagou, payment_id, aceita_advogado)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                 (dados['nome'], dados['email'], dados['whatsapp'], dados['cidade'], dados['resumo'], 
                  dados['categoria'], dados['prob'], dados['valor'], dados['pagou'], dados.get('payment_id'), dados['aceita_advogado']))
    conn.commit()
    conn.close()

def atualizar_pagamento_lead(payment_id):
    conn = sqlite3.connect('indeniza.db')
    c = conn.cursor()
    c.execute("UPDATE leads SET pagou = 1 WHERE payment_id = ?", (str(payment_id),))
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

# Inicializa Banco
init_db()

# ==============================================================================
# CARREGAMENTO DE IA (CORRIGIDO)
# ==============================================================================
@st.cache_resource
def carregar_modelos_ia():
    bi_encoder = SentenceTransformer("intfloat/multilingual-e5-large")
    
    # --- SUA REGRA DE OURO ---
    NOME_CROSS_ENCODER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    cross_encoder = CrossEncoder(NOME_CROSS_ENCODER)
    # -------------------------
    
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
        
        # Tratamento
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

# Configuração de APIs (Compatível com seus Secrets)
api_key_openai = st.secrets.get("OPENROUTER_API_KEY") 
# Se a chave estiver dentro de [openai] no secrets, use: st.secrets["openai"]["api_key"]
# Vou deixar genérico para tentar pegar direto, ajuste no secrets se precisar.

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key_openai) if api_key_openai else None

ANO_ATUAL = datetime.now().year

# ==============================================================================
# LÓGICA DE NEGÓCIO
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
# INTERFACE
# ==============================================================================
with st.sidebar:
    st.title("Área Restrita 🔒")
    menu = st.radio("Navegação", ["Sou Cliente", "Sou Advogado"])

# --- CLIENTE ---
if menu == "Sou Cliente":
    st.markdown("<h1 style='text-align: center;'>⚖️ Indeniza Aí</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Analise seu caso com IA e descubra quanto você pode ganhar.</p>", unsafe_allow_html=True)

    if 'analise_concluida' not in st.session_state: st.session_state.analise_concluida = False
    if 'dados_finais' not in st.session_state: st.session_state.dados_finais = {}
    if 'pagamento_aprovado' not in st.session_state: st.session_state.pagamento_aprovado = False

    queixa = st.text_area("O que aconteceu?", placeholder="Ex: Voo cancelado em Guarulhos, meu nome foi negativado indevidamente...", height=120)

    if st.button("CALCULAR PROBABILIDADE GRATUITAMENTE"):
        if len(queixa) < 15:
            st.warning("⚠️ Descreva melhor o caso.")
        elif not client:
            st.error("Erro: API Key OpenAI não configurada.")
        else:
            with st.spinner("🤖 A IA está lendo jurisprudências..."):
                try:
                    # 1. Router
                    prompt = f'Analise: "{queixa[:500]}". JSON output: {{"categoria": "AEREO" ou "NOMESUJO" ou "OUTROS", "valido": true/false}}'
                    check = client.chat.completions.create(model="xiaomi/mimo-v2-flash:free", messages=[{"role": "user", "content": prompt}], temperature=0)
                    resp = json.loads(check.choices[0].message.content.replace("```json","").replace("```","").strip())
                    
                    if not resp['valido']:
                        st.error("Texto inválido.")
                        st.stop()
                    
                    arquivo_alvo = "banco_vetorial_tjpr_e5.pkl" if resp['categoria'] == 'AEREO' else "banco_nome_sujo.pkl"
                    
                    # 2. Busca
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
                    
                    st.session_state.dados_finais = {
                        "prob": prob, "valor": val_esperado, "df_finais": finais, 
                        "categoria": resp['categoria'], "resumo": queixa
                    }
                    st.session_state.analise_concluida = True
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Erro: {e}")

    # --- RESULTADOS ---
    if st.session_state.analise_concluida:
        dados = st.session_state.dados_finais
        
        # Probabilidade (Sempre visível)
        cor_prob = "#2ecc71" if dados['prob'] > 70 else "#f1c40f"
        st.markdown(f"""
            <div class='metric-card' style='border-left: 5px solid {cor_prob};'>
                <h3>Probabilidade de Vitória</h3>
                <h1 style='color: {cor_prob}; font-size: 3rem;'>{dados['prob']:.0f}%</h1>
                <p>Baseado em {len(dados['df_finais'])} casos analisados</p>
            </div>
        """, unsafe_allow_html=True)

        # === BLOQUEIO DE PAGAMENTO (PAYWALL) ===
        if not st.session_state.pagamento_aprovado:
            # Conteúdo Borrado (Blur)
            st.markdown("<div class='blur-content'>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1: st.markdown("<div class='metric-card'><h3>Valor Estimado</h3><h1>R$ 5.800,00</h1></div>", unsafe_allow_html=True)
            with col2: st.markdown("<div class='metric-card'><h3>Tempo Médio</h3><h1>12 Meses</h1></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Caixa do Paywall
            with st.container():
                st.markdown("""
                <div class='paywall-overlay'>
                    <h2>🔓 Desbloqueie seu Relatório Completo</h2>
                    <p>Veja o valor exato, casos semelhantes e fale com advogados.</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.form("form_pagamento"):
                    c_nome = st.text_input("Nome Completo")
                    c_email = st.text_input("E-mail (Para receber o relatório)")
                    c_whats = st.text_input("WhatsApp (com DDD)")
                    c_cidade = st.text_input("Cidade/Estado")
                    c_check = st.checkbox("Aceito que advogados entrem em contato.")
                    
                    st.divider()
                    st.write("Valor do Relatório: **R$ 9,90**")
                    submitted = st.form_submit_button("GERAR PIX PARA LIBERAR")

                    if submitted:
                        if len(c_nome) > 3 and len(c_email) > 5:
                            # INTEGRAÇÃO MERCADO PAGO
                            try:
                                sdk = mercadopago.SDK(st.secrets["pagamento"]["mp_token"])
                                payment_data = {
                                    "transaction_amount": 9.90,
                                    "description": "Relatório IndenizaAí",
                                    "payment_method_id": "pix",
                                    "payer": {"email": c_email, "first_name": c_nome}
                                }
                                payment_response = sdk.payment().create(payment_data)
                                pagamento = payment_response["response"]
                                
                                st.session_state.pagamento_id = pagamento['id']
                                st.session_state.qr_code_copia = pagamento['point_of_interaction']['transaction_data']['qr_code']
                                st.session_state.aguardando_pagamento = True
                                
                                # Salva Lead Pendente
                                salvar_lead({
                                    "nome": c_nome, "email": c_email, "whatsapp": c_whats, 
                                    "cidade": c_cidade, "resumo": dados['resumo'], 
                                    "categoria": dados['categoria'], "prob": dados['prob'], 
                                    "valor": dados['valor'], "pagou": False, 
                                    "payment_id": str(pagamento['id']), "aceita_advogado": c_check
                                })
                                st.rerun()
                            except Exception as e:
                                st.error(f"Erro ao gerar PIX: {e}")
                        else:
                            st.warning("Preencha os dados corretamente.")
                
                # Exibe PIX se gerado
                if st.session_state.get('aguardando_pagamento'):
                    with st.container(border=True):
                        st.info("📱 Pague via PIX (Copia e Cola)")
                        st.code(st.session_state.qr_code_copia, language="text")
                        
                        if st.button("JÁ PAGUEI! LIBERAR RELATÓRIO"):
                            try:
                                sdk = mercadopago.SDK(st.secrets["pagamento"]["mp_token"])
                                check = sdk.payment().get(st.session_state.pagamento_id)
                                status = check["response"]["status"]
                                
                                if status == "approved":
                                    st.session_state.pagamento_aprovado = True
                                    atualizar_pagamento_lead(st.session_state.pagamento_id)
                                    st.balloons()
                                    st.rerun()
                                else:
                                    st.warning(f"Status atual: {status}. Aguarde alguns segundos...")
                            except:
                                st.error("Erro ao verificar pagamento.")

        else:
            # === DESBLOQUEADO (PREMIUM) ===
            st.success("✅ Relatório Desbloqueado com Sucesso!")
            
            # Valor Real
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

# --- ADVOGADO ---
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
                    st.error("Acesso negado. (Teste: admin/admin)")
    else:
        # Dashboard
        st.write(f"Bem-vindo, Dr(a). {st.session_state.advogado_logado}!")
        if st.button("Sair"):
            del st.session_state.advogado_logado
            st.rerun()
            
        st.divider()
        st.subheader("🔥 Leads Recentes")
        
        df_leads = listar_leads()
        if not df_leads.empty:
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
                        status_pag = "✅ PAGO" if row['pagou'] else "❌ PENDENTE"
                        st.info(f"Prob: {row['probabilidade']:.0f}% | {status_pag}")
                        st.write(f"Caso: {row['resumo_caso'][:100]}...")
                    with c3:
                        if row['aceita_advogado']:
                            st.link_button("Chamar no WhatsApp", f"https://wa.me/55{str(row['whatsapp']).replace(' ','').replace('-','')}")
                        else:
                            st.caption("Contato não autorizado")
        else:
            st.info("Nenhum lead cadastrado ainda.")

# Rodapé
st.markdown("<br><hr><center style='color:#ccc'>Indeniza Aí © 2026</center>", unsafe_allow_html=True)