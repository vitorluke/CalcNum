from src.classes.rede_hidraulica import RedeHidraulica
from src.graphs_utils.gera_grafo import gera_grafo
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import lu_factor, lu_solve

def plot_pressao_maxima(vetor_tempo, vetor_pressao, titulo, caminho_salvar=None):
    plt.figure(figsize=(8, 5))
    plt.plot(vetor_tempo, vetor_pressao, label="Pressão Máxima", color="darkred")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Pressão (Pa)")
    plt.title(titulo)
    plt.grid(True)
    plt.legend()
    if caminho_salvar:
        plt.savefig(caminho_salvar, dpi=300)
    plt.show()

def resolver_base_superposicao(rede:RedeHidraulica, nos_atm, no_injecao):
    bombas_unitarias = {no_injecao: 1.0}
    pressao_base = rede.resolver(nos_atm, bombas_unitarias)
    return pressao_base.copy()

def exercicio_4(omega=3, n_passos=1000, tempo_final=10):
    rede = RedeHidraulica(levels=3, A_k=2.5e-7)

    tempo = np.linspace(0, tempo_final, n_passos)
    q0_t = np.add(0.1e-6 * np.sin(omega * tempo), 1e-6)
    
    pressao_base_0 = resolver_base_superposicao(rede, {6: 0.0}, 1)
    
    pressoes_maximas = []
    
    for q in q0_t:
        pressao_t = q * pressao_base_0
        pressoes_maximas.append(np.max(pressao_t))
    
    plot_pressao_maxima(tempo, pressoes_maximas, "Ex 4: Pressão Máxima na Rede ao Longo do Tempo", "imagens/rede hidraulica/ex4.png")
    return tempo, pressoes_maximas

def exercicio_5(omega=4, n_passos=1000, tempo_final=10):
    rede = RedeHidraulica(levels=3, A_k=2.5e-7)

    tempo = np.linspace(0, tempo_final, n_passos)
    q0_t = np.add(0.1e-6 * np.sin(3 * tempo), 1e-6)
    q175_t = np.add(0.01e-6 * np.cos(omega * tempo), 0.1e-6)
    
    pressao_base_0 = resolver_base_superposicao(rede, {6: 0.0}, 1)

    last_node = rede.numero_nos if rede.numero_nos < 176 else 176

    pressao_base_175 = resolver_base_superposicao(rede, {6: 0.0}, 176)
    
    pressoes_maximas = []
    
    for q0, q175 in zip(q0_t, q175_t):
        pressao_t = (q0 * pressao_base_0) + (q175 * pressao_base_175)
        pressoes_maximas.append(np.max(pressao_t))
        
    plot_pressao_maxima(tempo, pressoes_maximas, "Ex 5: Pressão Máxima com Múltiplas Injeções", "imagens/rede hidraulica/ex5.png")
    return tempo, pressoes_maximas

def calcular_temperatura(t):
    return 20.0 + 0.9 * (t ** 2)

def calcular_viscosidade(temp):
    return 0.001791 / (1 + 0.03368 * temp + 0.000221 * (temp ** 2))

def exercicio_6(n_passos=1000, tempo_final=10):
    rede = RedeHidraulica(levels=3, A_k=2.5e-7)

    tempo = np.linspace(0, tempo_final, n_passos)
    q0_constante = 0.1e-6
    
    pressao_base_0 = resolver_base_superposicao(rede, {6: 0.0}, 1)
    
    temperatura_inicial = calcular_temperatura(0)
    viscosidade_inicial = calcular_viscosidade(temperatura_inicial)
    
    pressoes_maximas = []
    
    for t in tempo:
        temp_atual = calcular_temperatura(t)
        visc_atual = calcular_viscosidade(temp_atual)
        fator_escala = visc_atual / viscosidade_inicial
        pressao_t = (q0_constante * pressao_base_0) * fator_escala
        pressoes_maximas.append(np.max(pressao_t))
        
    plot_pressao_maxima(tempo, pressoes_maximas, "Ex 6: Pressão Máxima com Viscosidade Variável", "imagens/rede hidraulica/ex6.png")
    return tempo, pressoes_maximas

def montar_matriz_fatorada(rede, nos_atm):
    rede.assembly()
    matriz = rede.matriz_global.copy()
    
    for no in nos_atm:
        index_atm = no - 1
        matriz[index_atm, :] = 0
        matriz[index_atm, index_atm] = 1
        
    return lu_factor(matriz)

def resolver_sistema_fatorado(lu_e_piv, numero_nos, nos_atm, bombas):
    vazao = np.zeros(numero_nos)
    
    for no, q in bombas.items():
        index = no - 1
        vazao[index] += q
        
    for no in nos_atm:
        index_atm = no - 1
        vazao[index_atm] = 0
        
    return lu_solve(lu_e_piv, vazao)

def avaliar_desempenho_rede(niveis, num_execucoes=10):
    tempos_montagem = []
    tempos_resolucao = []
    
    for _ in range(num_execucoes):
        inicio_montagem = time.perf_counter()
        rede = RedeHidraulica(levels=niveis, A_k=2.5e-7)
        rede.assembly()
        fim_montagem = time.perf_counter()
        tempos_montagem.append(fim_montagem - inicio_montagem)
        
        inicio_resolucao = time.perf_counter()
        rede.resolver({6: 0.0}, {1: 1.0})
        fim_resolucao = time.perf_counter()
        tempos_resolucao.append(fim_resolucao - inicio_resolucao)
        
    tempo_medio_montagem = np.mean(tempos_montagem)
    tempo_medio_resolucao = np.mean(tempos_resolucao)
    
    return rede.numero_nos, tempo_medio_montagem, tempo_medio_resolucao

def exercicio_7(niveis):
    print(f"{'Nível':<10} | {'Nº de Nós':<15} | {'Tempo Médio Montagem (s)':<25} | {'Tempo Médio Resolução (s)':<25}")
    print("-" * 80)
    
    for nivel in niveis:
        # nos, arestas = gera_grafo(levels=nivel)
        # numero_nos = len(nos)
        # condutancias = np.ones(len(arestas))
        
        # rede = RedeHidraulica(numero_nos=numero_nos, conectividade=arestas, condutancias=condutancias, coordenadas=nos)

        numero_nos, t_montagem, t_resolucao = avaliar_desempenho_rede(nivel)
        print(f"{nivel:<10} | {numero_nos:<15} | {t_montagem:<25.6e} | {t_resolucao:<25.6e}")

def main():
    exercicio_4()
    exercicio_5()
    exercicio_6()
    exercicio_7(range(1,11))

if __name__ == "__main__":
    main()