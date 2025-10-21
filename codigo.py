import numpy as np
import random
from collections import deque, Counter
import pandas as pd
import time
import matplotlib.pyplot as plt

def carregar_matriz_C(caminho_csv):
    """
    Carrega a matriz de coancestralidade a partir de um arquivo CSV.
    """ 
    df = pd.read_csv(caminho_csv)
    todos_pares = list(set(df['Animal_1']).union(set(df['Animal_2'])))
    machos = sorted({p.split('_')[0] for p in todos_pares})
    femeas = sorted({p.split('_')[1] for p in todos_pares})
    NF = len(femeas)
    NM = len(machos)
    tamanho = NF * NM
    C = np.zeros((tamanho, tamanho))
    par_para_indice = {f"{m}_{f}": i for i, (f, m) in enumerate([(f, m) for f in femeas for m in machos])}

    for _, linha in df.iterrows():
        a1, a2, coef = linha['Animal_1'], linha['Animal_2'], linha['Coef']
        if a1 in par_para_indice and a2 in par_para_indice:
            i = par_para_indice[a1]
            j = par_para_indice[a2]
            C[i, j] = coef
            C[j, i] = coef

    np.fill_diagonal(C, 1) # Garante que a diagonal principal seja 1
    return C, femeas, machos, par_para_indice

def avaliar(P, C, NF, NM):
    """
    Calcula o custo de uma solução (média da coancestralidade).
    """
    soma = 0.0
    total = 0
    for i in range(NF):
        for j in range(i + 1, NF):
            idx1 = i * NM + P[i]
            idx2 = j * NM + P[j]
            soma += C[idx1, idx2]
            total += 1
    return soma / total if total > 0 else float('inf')


def gerar_vizinhos(P, NM, max_uso):
    """
    Gera soluções vizinhas através de trocas simples e duplas.
    """
    vizinhos = []
    NF = len(P)
    uso = Counter(P)
    # Troca simples
    for i in range(NF):
        for m in range(NM):
            # Usamos .get() para o caso de um macho não estar no dicionário max_uso
            if m != P[i] and uso[m] < max_uso.get(m, float('inf')):
                nova = P.copy()
                nova[i] = m
                vizinhos.append(nova)
    # Troca dupla
    for i in range(NF):
        for j in range(i + 1, NF):
            if P[i] != P[j]:
                nova = P.copy()
                nova[i], nova[j] = nova[j], nova[i]
                vizinhos.append(nova)
    return vizinhos

def solucao_inicial(NF, NM, max_uso):
    """
    Cria uma solução inicial válida que respeita as restrições de uso.
    """
    P = [-1] * NF
    machos_disponiveis = list(range(NM))
    random.shuffle(machos_disponiveis)
    contagem = Counter()

    for i in range(NF):
        macho_encontrado = False
        # Tenta atribuir um macho que não exceda o limite
        for m in machos_disponiveis:
            if contagem[m] < max_uso.get(m, float('inf')):
                P[i] = m
                contagem[m] += 1
                macho_encontrado = True
                break
        # Se todos os machos atingiram o limite, atribui um aleatório
        if not macho_encontrado:
             P[i] = random.choice(machos_disponiveis)
             print(f"Aviso: macho {m} excedeu o limite de uso na solução inicial.")
    return P

def busca_tabu(C, NF, NM, max_uso, iter_max, tabu_tam, limite_sem_melhora=300):
    """
    Executa o algoritmo de Busca Tabu com Critério de Aspiração e Diversificação.
    """
    P = solucao_inicial(NF, NM, max_uso)
    melhor_P = P.copy()
    custo_atual = avaliar(P, C, NF, NM)
    melhor_custo = custo_atual

    lista_tabu = deque(maxlen=tabu_tam)
    lista_tabu.append(tuple(P))

    custo_historico = [melhor_custo]
    iter_sem_melhora = 0

    for it in range(iter_max):
        vizinhos = gerar_vizinhos(P, NM, max_uso)
        random.shuffle(vizinhos)

        melhor_vizinho = None
        melhor_custo_vizinho = float('inf')

        for vizinho in vizinhos:
            custo_v = avaliar(vizinho, C, NF, NM)
            
            eh_tabu = tuple(vizinho) in lista_tabu
            eh_aspiracao = custo_v < melhor_custo
          
            if not eh_tabu or eh_aspiracao:
                if custo_v < melhor_custo_vizinho:
                    melhor_vizinho = vizinho
                    melhor_custo_vizinho = custo_v
            if eh_tabu and eh_aspiracao:
             print(f"[{it}] Solução tabu aceita por aspiração! Custo: {custo_v:.4f}")

     
        if melhor_vizinho is None:
            print(f"[{it}] Não foram encontrados vizinhos válidos. Encerrando.")
            break

        P = melhor_vizinho
        custo_atual = melhor_custo_vizinho
        lista_tabu.append(tuple(P))

        if custo_atual < melhor_custo:
            melhor_P = P.copy()
            melhor_custo = custo_atual
            iter_sem_melhora = 0
        else:
            iter_sem_melhora += 1

        custo_historico.append(melhor_custo)

        if it % 100 == 0:
            print(f"[{it}] Custo atual: {custo_atual:.4f} | Melhor global: {melhor_custo:.4f}")

        if iter_sem_melhora >= limite_sem_melhora:
            print(f"[{it}] ESTAGNAÇÃO! Resetando a solução para diversificar a busca.")
            P = solucao_inicial(NF, NM, max_uso)
            iter_sem_melhora = 0
            lista_tabu.clear()
            lista_tabu.append(tuple(P))

    return melhor_P, melhor_custo, custo_historico

if __name__ == "__main__":
    try:
        inicio = time.time()
        
        caminho_csv = "parentesco_produtos.csv"
        C, femeas, machos, par_para_indice = carregar_matriz_C(caminho_csv)
        NF = len(femeas)
        NM = len(machos)

        print(f"Número de fêmeas: {NF}, Número de machos: {NM}")
        
        max_uso = {i: 12 for i in range(NM)}

        solucao, custo, historico = busca_tabu(
            C, NF, NM, max_uso, 
            iter_max=5000, 
            tabu_tam=30, 
            limite_sem_melhora=400
        )

        fim = time.time()
        
        print("\nCruzamentos finais:")
        for i, macho_idx in enumerate(solucao):
            print(f" {machos[macho_idx]} × {femeas[i]}")

        print(f"\nCusto total de coancestralidade: {custo:.4f}")
        print(f"Tempo de execução: {fim - inicio:.2f} segundos")

        # --- ALTERAÇÃO AQUI ---
        # Salvar o gráfico em um arquivo PNG em vez de exibi-lo
        nome_arquivo_grafico = 'grafico_convergencia_coancestralidade.png'
        plt.figure(figsize=(12, 6))
        plt.plot(historico)
        plt.title('Convergência do Custo da Busca Tabu')
        plt.xlabel('Iteração')
        plt.ylabel('Melhor Custo de Coancestralidade')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(nome_arquivo_grafico, dpi=300) # dpi=300 para alta resolução
            
        print(f"\nGráfico de convergência salvo como '{nome_arquivo_grafico}'")

    except FileNotFoundError:
        print("\nERRO: O arquivo 'parentesco_produtos.csv' não foi encontrado.")
        print("Por favor, carregue o arquivo e execute o script novamente.")