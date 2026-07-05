from src.classes.rede_hidraulica import RedeHidraulica
from src.classes.placa_termica import PlacaTermica
from src.graphs_utils.utils import gerar_grafo_aleatorio, plotar_grafo_alternativo
from src.graphs_utils.utils import gera_rede
from src.graph_benchmarking.benchmark import *

from src.classes.hidraulico_termico import HidraulicoTermico, ex_2_acoplamento, ex_3_acoplamento, ex_4_acoplamento, ex_5_acoplamento, ex_1_especial_acoplamento, ex_2_extra
    
def main():
    #ex_2_acoplamento()
    #ex_3_acoplamento()
    #ex_4_acoplamento()
    #ex_5_acoplamento()
    #ex_2_extra()
    ex_1_especial_acoplamento()

if __name__ == "__main__":
    main()