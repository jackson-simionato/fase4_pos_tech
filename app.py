#Importação das bibliotecas
import streamlit as st 
from utils import DropFeatures, OneHotEncoding, OrdinalFeature, OverSample, MinMax
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
from joblib import load

# Dicionário para converter Sim e Não para binário
dict_binario = {'Sim':1, 'Não':0}

# Carregando dataframe limpo 
dados = pd.read_csv('./dados/credit_score/df_clientes_variavel_target.csv')

############################# Streamlit ############################
st.markdown('<style>div[role="listbox"] ul{background-color: #6e42ad}; </style>', unsafe_allow_html=True)

st.markdown("<h1 style = 'text-align : center; '> Modelo de análise de crédito </h1>", unsafe_allow_html=True)

st.warning('Preencha o formulário com todos os seus dados pessoais e clique no botão **ENVIAR** no final da página.')

st.divider()

st.write('### Idade')
input_idade = int(st.slider('Selecione sua idade', 18, 100))

st.write('\n')
st.write('### Grau de escolaridade')
input_escolaridade = st.selectbox(label='Selecione seu grau de escolaridade',options=list(dados['Grau_escolaridade'].unique()),
                                  placeholder="Escolha uma opção")

st.write('\n')
st.write('### Estado civil')
input_estado_civil = st.selectbox(label='Selecione seu estado civil',options=list(dados['Estado_civil'].unique()),
                                  placeholder="Escolha uma opção")
st.write('\n')
st.write('### Tamanho da família')
input_tamanho_familia = int(st.slider('Indique quantos membros sua família possui', 1, 20))

st.write('\n')
st.write('### Carro próprio')
input_carro = st.radio('Possui carro próprio?', ['Sim','Não'])
input_carro = dict_binario[input_carro]

st.write('\n')
st.write('### Casa própria')
input_casa = st.radio('Possui casa própria?', ['Sim','Não'])
input_casa = dict_binario[input_casa]

st.write('\n')
st.write('### Moradia')
input_moradia = st.selectbox(label='Selecione seu tipo de residência',options=list(dados['Moradia'].unique()),
                                  placeholder="Escolha uma opção")

st.write('\n')
st.write('### Categoria de renda')
input_categoria_renda = st.selectbox(label='Selecione sua categoria de renda',options=list(dados['Categoria_de_renda'].unique()),
                                  placeholder="Escolha uma opção")

st.write('\n')
st.write('### Ocupação')
input_ocupacao = st.selectbox(label='Selecione sua ocupação',options=list(dados['Ocupacao'].unique()),
                                  placeholder="Escolha uma opção")

st.write('\n')
st.write('### Anos empregado')
input_anos_empregado = int(st.slider('Indique há quantos anos está empregado', 0, 60))

st.write('\n')
st.write('### Rendimento anual')
input_rendimento_anual = float(st.number_input('Informe seu rendimento anual médio em R$', 0))

st.write('\n')
st.write('### Email')
input_email = st.radio('Possui email?', ['Sim','Não'])
input_email = dict_binario[input_email]

st.write('\n')
st.write('### Telefone fixo')
input_telefone_fixo = st.radio('Possui telefone fixo pessoal?', ['Sim','Não'])
input_telefone_fixo = dict_binario[input_telefone_fixo]

st.write('\n')
st.write('### Telefone corporativo')
input_telefone_trabalho = st.radio('Possui telefone corporativo?', ['Sim','Não'])
input_telefone_trabalho = dict_binario[input_telefone_trabalho]

new_client = [0, input_carro, input_casa, input_telefone_trabalho, input_telefone_fixo, 
              input_email, input_idade, input_anos_empregado, input_tamanho_familia,
              input_rendimento_anual, input_categoria_renda, input_escolaridade,
              input_estado_civil, input_moradia, input_ocupacao, 0]