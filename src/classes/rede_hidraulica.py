import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time

from src.graphs_utils.gera_grafo import gera_grafo

class RedeHidraulica:
    def __init__(self, levels: int = 3, A_k=None, H_k=None):

        assert (A_k is not None) or (H_k is not None)

        self.Lx = 0.03
        self.Ly = 0.015

        if A_k is not None:
            self.A_k = A_k
            self.D_k = np.sqrt(
                4.0 * A_k / np.pi
            )
        else:
            self.D_k = H_k

        self.temperatura_referencia = 20.0

        coordenadas, conectividade = gera_grafo(levels)
        coordenadas *= 1e-3

        self.numero_nos = len(coordenadas)

        self.conec = np.asarray(
            conectividade,
            dtype=np.int32
        )

        self.xnos = np.asarray(
            coordenadas,
            dtype=np.float64
        )

        self.i_edges = self.conec[:, 0]
        self.j_edges = self.conec[:, 1]

        dx = (
            self.xnos[self.i_edges, 0]
            - self.xnos[self.j_edges, 0]
        )

        dy = (
            self.xnos[self.i_edges, 1]
            - self.xnos[self.j_edges, 1]
        )

        self.comprimentos = np.sqrt(
            dx * dx + dy * dy
        )

        self.numero_canos = len(self.conec)
        self.D = np.zeros((self.numero_canos, self.numero_nos))
        for k, (i, j) in enumerate(self.conec):
            self.D[k, i] = 1
            self.D[k, j] = -1

        self.cond = self._calcular_condutancias(np.full(self.numero_nos, self.temperatura_referencia))

        self.vazoes_por_no = None
        self.pressao_por_no = None

        self.A = None

        self.p = None
        self.Q = None

        self.historico_pressao = []
        self.historico_vazao = []

    def viscosidade(self, T):
        return 0.001791 / (1 + 0.03368 * T + 0.000221 * T*T)

    def _calcular_condutancias(self, temperaturas):
        T_media = 0.5 * (temperaturas[self.i_edges] + temperaturas[self.j_edges])

        mu = self.viscosidade(T_media)

        kappa = np.pi * self.D_k**4 / (128.0 * mu)

        return kappa/ self.comprimentos

    def assembly(self):
        ne = self.numero_canos

        rows = np.empty(4 * ne, dtype=np.int32)
        cols = np.empty(4 * ne, dtype=np.int32)

        data = np.empty(4 * ne,dtype=np.float64)

        rows[0::4] = self.i_edges
        rows[1::4] = self.j_edges
        rows[2::4] = self.i_edges
        rows[3::4] = self.j_edges

        cols[0::4] = self.i_edges
        cols[1::4] = self.j_edges
        cols[2::4] = self.j_edges
        cols[3::4] = self.i_edges

        data[0::4] = self.cond
        data[1::4] = self.cond
        data[2::4] = -self.cond
        data[3::4] = -self.cond

        self.A = sp.coo_matrix((data, (rows, cols)), shape=(self.numero_nos, self.numero_nos)).tocsr()

    def resolver(self, pressao_imposta=None, vazao_imposta=None):
        if pressao_imposta is None:
            pressao_imposta = {}

        if vazao_imposta is None:
            vazao_imposta = {}

        if self.A is None:
            self.assembly()

        matriz_modificada = self.A.tolil(copy=True)

        b = np.zeros(self.numero_nos)

        for k, vazao in vazao_imposta.items():
            b[k] = vazao

        for k, pressao in pressao_imposta.items():
            matriz_modificada[k, :] = 0.0
            matriz_modificada[k, k] = 1.0

            b[k] = pressao

        self.p = spla.spsolve(matriz_modificada.tocsc(), b)

        self.calcular_vazoes()

        self.historico_pressao.append(self.p.copy())
        self.historico_vazao.append(self.Q.copy())

        return self.p

    def calcular_vazoes(self):
        dp = self.D @ self.p
        self.Q = self.cond * dp

        return self.Q

    def calcular_potencia(self):
        if self.p is None:
            print("Erro: Resolva a rede antes de plotar.")
            return None

        p = self.p
        D = self.D
        K = np.diag(self.cond)

        W = p.T @ D.T @ K @ D @ p

        return W

    def plotaRede(self, scale=1.0, save_path=None):
        if self.p is None or self.Q is None:
            print("Erro: Resolva a rede antes de plotar.")
            return

        coord = self.xnos * scale

        edges = self.conec

        p = self.p
        q = self.Q

        segs = []
        mids = []

        for (i, j) in edges:
            x1, y1 = coord[i]
            x2, y2 = coord[j]

            segs.append(((x1, y1), (x2, y2)))

            mids.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

        segs = np.asarray(segs)
        mids = np.asarray(mids)

        fig, ax = plt.subplots(figsize=(10, 10))

        cmap = plt.get_cmap("coolwarm")

        norm = plt.Normalize(vmin=float(p.min()), vmax=float(p.max()))

        colors = [cmap(norm(pi)) for pi in p]

        ax.scatter(
            coord[:, 0],
            coord[:, 1],
            s=500,
            c=colors,
            zorder=3,
            edgecolors="black"
        )

        arrow_scale = 0.05

        for idx, ((x1, y1), (x2, y2)) in enumerate(segs):
            ax.plot(
                [x1, x2],
                [y1, y2],
                color="black",
                linewidth=2.0,
                zorder=1
            )

            xm, ym = mids[idx]

            dx = x2 - x1
            dy = y2 - y1

            L = np.hypot(dx, dy)

            if L == 0:
                continue

            dxn = dx / L
            dyn = dy / L

            nx = -dyn
            ny = dxn

            q_dir = 1 if p[edges[idx, 0]]> p[edges[idx, 1]]else -1

            ax.annotate(
                "",
                xy=(
                    xm + q_dir * 1.5 * arrow_scale * dxn,
                    ym + q_dir * 1.5 * arrow_scale * dyn
                ),
                xytext=(
                    xm - q_dir * 1.5 * arrow_scale * dxn,
                    ym - q_dir * 1.5 * arrow_scale * dyn
                ),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="black",
                    lw=1.5,
                    mutation_scale=20
                ),
                zorder=5
            )

            ax.text(
                xm + nx * 0.1,
                ym + ny * 0.1,
                f"q={q[idx]:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                zorder=6,
                bbox=dict(
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none"
                )
            )

        for i, (x, y) in enumerate(coord):

            ax.text(
                x,
                y,
                str(i + 1),
                ha="center",
                va="center",
                fontweight="bold",
                zorder=4
            )

            ax.text(
                x,
                y - 0.15,
                f"p={p[i]:.2f}",
                ha="center",
                va="top",
                fontsize=9,
                color="blue"
            )

        ax.set_aspect("equal")
        ax.axis("off")

        ax.set_xlim(coord[:, 0].min() - 0.5, coord[:, 0].max() + 0.5)

        ax.set_ylim(coord[:, 1].min() - 0.5, coord[:, 1].max() + 0.5)

        sm = cm.ScalarMappable(cmap=cmap,norm=norm)

        plt.colorbar(sm, ax=ax,label="Pressão (p)")

        if save_path:
            plt.savefig(save_path, dpi=300)

        plt.show()

    def atualizar_condutancias(self, temperaturas):

        self.cond = self._calcular_condutancias(temperaturas)
        self.A = None

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
    
    pressao_base_0 = resolver_base_superposicao(rede, {5: 0.0}, 0)
    
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
    
    pressao_base_0 = resolver_base_superposicao(rede, {5: 0.0}, 0)

    last_node = rede.numero_nos if rede.numero_nos < 176 else 176

    pressao_base_175 = resolver_base_superposicao(rede, {5: 0.0}, 175)
    
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
    
    pressao_base_0 = resolver_base_superposicao(rede, {5: 0.0}, 0)
    
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
        rede.resolver({5: 0.0}, {0: 1.0})
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