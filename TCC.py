from surprise import Dataset, Reader, KNNBasic, KNNWithMeans, accuracy
from surprise.model_selection import train_test_split
from collections import defaultdict
import pandas as pd
import numpy as np

def carregar_filmes(path):
    filmes = pd.read_csv(path)
    return filmes

def carregar_avaliacoes(path):
    avaliacoes = pd.read_csv(path)
    return avaliacoes

#carregar um arquivo padrão Movielens definido na variavel PATH
def carregar_dataset(path):
    reader = Reader (line_format = 'user item rating timestamp', sep = ',', rating_scale=(0.5,5), skip_lines=1)
    data = Dataset.load_from_file(path, reader = reader)
    return data

#Configura um modelo de recomendação, executa o treinamento e realiza a validação
def rodar_modelo(data, teste_tamanho, sim_opcoes, k):
    treina, testa = train_test_split(data, teste_tamanho)
    knn = KNNBasic(k=k, sim_options=sim_opcoes)
    knn.fit(treina)
    knn_predicoes = knn.test(testa)
    accuracy.rmse(knn_predicoes)
    return knn

#Configura um modelo de recomendação, executa o treinamento e realiza a validação
def rodar_modelo_sem_teste(data, sim_opcoes, k):
    treina = data.build_full_trainset()
    knn = KNNBasic(k=k, sim_options=sim_opcoes)
    knn.fit(treina)
    return knn

#Realiza uma previsão de avaliação para um usuario com base no modelo treinado 
def prever_avaliacao(modelo, id_usuario, id_filme, mostrar_tela):
    return modelo.predict(str(id_usuario), str(id_filme), verbose = mostrar_tela)

#Retorna os n vizinhos de um usuário com base no modele treinado
def encontrar_vizinhos(modelo, id_usuario, n):
    return modelo.get_neighbors(id_usuario, n)

#Realiza a recomendação dos filmes não vistos que tem as maiores previsoes de avaliação com base no modelo treinado
def recomendar_filmes(modelo, data, id_usuario, n):
    print('Montando base...')
    data_treina_prever = data.build_full_trainset()
    print('Encontrando filmes não avaliados...')
    data_treina_prever = data_treina_prever.build_anti_testset()
    lista_filmes_nao_assistidos = []
    for id_usu, id_filme, estimativa_default in data_treina_prever:
        if str(id_usu) == str(id_usuario):
            lista_filmes_nao_assistidos.append(id_filme)
    #print (sorted(lista_filmes))
    #return 1
    top_filmes = []
    print('Avaliando filmes...')
    for id_filme in lista_filmes_nao_assistidos:
       top_filmes.append([id_filme, prever_avaliacao(modelo, id_usuario, id_filme, False).est]) 
    top_filmes = sorted(top_filmes, key=lambda x: x[1], reverse=True)[:n]
    print('Previsões realizadas. Recomendados '+str(n)+' filmes.')
    filmes_recomendados = np.array(top_filmes)
    return filmes_recomendados

#retorna um dataframe com os dados de uma lista de filmes
def encontrar_detalhes_filmes(lista_filmes, id_filmes):
    detalhes_filmes = lista_filmes[lista_filmes['movieId'].isin(id_filmes)]
    return detalhes_filmes
    

    
