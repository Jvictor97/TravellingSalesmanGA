# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import math
import random
from matplotlib import pyplot as plt

# Define o número de cidades
numCidades = 10

# Define a matriz de distâncias
matriz = np.zeros(shape=(numCidades, numCidades))

# Número de indivíduos por geração (DEVE SER ÍMPAR)
populacaoMaxima = 201

# Define a probabilidade de ocorrer mutação
probabilidadeMutacao = 0.01


def main():
    mapa = np.loadtxt("MAPA10.txt")
    geracao = np.zeros(shape=(populacaoMaxima, numCidades), dtype=int)
    aptidoes = np.zeros(shape=(populacaoMaxima))

    xMapa = mapa[:, 0]
    yMapa = mapa[:, 1]

    # Cálculo da Matriz de Distâncias
    for iIdx, i in enumerate(mapa):
        for jIdx, j in enumerate(mapa):
            matriz[iIdx][jIdx] = math.sqrt(
                pow(i[0] - j[0], 2) + pow(i[1] - j[1], 2))

    # Permuta cidades de 0 a 10 (ou de 0 a 200) e armazena no vetor
    for i in range(0, populacaoMaxima):
        geracao[i] = np.random.permutation(numCidades)
        aptidoes[i] = fitness(geracao[i])

    # Inicia o ciclo de seleção -> reprodução -> mutação
    strike = 0
    menorDistancia = None
    numGeracoes = 1
    while strike < 700:
        sobrevivente, pares, ultimaMenorDistancia = selecao(geracao, aptidoes)
        geracao, aptidoes = reproducao(sobrevivente, geracao, pares)
        geracao = mutacao(geracao)
        numGeracoes += 1
        # Avaliação do resultado da última geração
        if ultimaMenorDistancia < menorDistancia or menorDistancia == None:
            menorDistancia = ultimaMenorDistancia

            caminho = np.zeros(shape=(numCidades + 1, 2))
            for idx, cidade in enumerate(sobrevivente):
                caminho[idx] = mapa[cidade]

            caminho[idx + 1] = caminho[0]

            xCaminho = caminho[:, 0]
            yCaminho = caminho[:, 1]

            plt.clf()
            plt.title('Caixeiro Viajante')
            plt.plot(xCaminho, yCaminho, color="g")
            plt.scatter(xMapa, yMapa, color="r")
            plt.pause(0.05)

            print('Menor Distancia: %f' % (menorDistancia))
            strike = 0
        else:
            strike += 1

    print('\nDistancia Final: %f' % (menorDistancia))
    print('Numero de Geracoes Total: %d' % numGeracoes)
    print('Numero de Geracoes Ate o Melhor: %d' % (numGeracoes - 700))
    print('Individuos por Geracao: %d' % populacaoMaxima)
    print('Taxa de Mutacao: %.2f' % probabilidadeMutacao)
    plt.show()


def fitness(individuo):
    # Calcula a função fitness para um dado indivíduo
    distancia = 0.0
    idx = 0
    while idx < len(individuo) - 1:
        distancia += matriz[individuo[idx]][individuo[idx + 1]]
        idx += 1

    distancia += matriz[individuo[idx]][individuo[0]]

    return distancia


def selecao(geracao, aptidoes):
    # Define o melhor indivíduo e os pares para reprodução
    menorDistancia = None
    sobrevivente = None
    numPares = 0
    roleta = np.zeros(shape=(populacaoMaxima))
    probabilidadeAcumulada = 0.0
    somaAptidoes = aptidoes.sum()
    numAptidoes = len(aptidoes)
    total = 0

    geracao = [g for _, g in sorted(
        zip(aptidoes, geracao), key=lambda pair: pair[0])]
    aptidoes.sort()

    # Define a Roleta e o Sobrevivente
    for idx, aptidao in enumerate(aptidoes):
        probabilidade = aptidao / somaAptidoes
        aptidaoAjustada = (2 / numAptidoes) - probabilidade

        roleta[idx] = aptidaoAjustada
        if menorDistancia == None or aptidao < menorDistancia:
            menorDistancia = aptidao
            sobrevivente = geracao[idx]

    # Define todos os 37 pares para reprodução
    pares = np.zeros(shape=(int(populacaoMaxima / 2), 2), dtype=int)
    while numPares < int(populacaoMaxima / 2):
        par = np.zeros(shape=(2), dtype=int)
        pais = 0
        while pais < 2:
            # Número randômico entre 0 e 1 para selecionar na roleta
            rand = random.uniform(0, 1)
            for idx, probabilidade in enumerate(roleta):
                rand -= probabilidade
                if rand <= 0:
                    par[pais] = idx
                    pais += 1
                    break

        pares[numPares] = par
        numPares += 1

    return sobrevivente, pares, menorDistancia


def reproducao(sobrevivente, geracao, pares):
    novaGeracao = np.zeros(shape=(populacaoMaxima, numCidades), dtype=int)
    aptidoes = np.zeros(shape=(populacaoMaxima))
    novaGeracao[0] = sobrevivente
    x = 0
    for i in xrange(1, populacaoMaxima, 2):
        novaGeracao[i], novaGeracao[i+1] = crossover(geracao, pares[x])
        x += 1

    for i in range(0, populacaoMaxima):
        aptidoes[i] = fitness(novaGeracao[i])

    return novaGeracao, aptidoes


def crossover(geracao, par):
    # Coleta os cromossomos dos pais
    mae = geracao[par[0]]
    pai = geracao[par[1]]

    # Define o índice até o qual será feita
    # a cópia de cromossomos para cada filho
    idxDivisao = random.randint(1, numCidades - 1)

    # Definindo cromossomos dos filhos
    filhos = np.zeros(shape=(2, numCidades), dtype=int)
    filhosGerados = 0
    while filhosGerados < 2:
        idx = 0
        cromossomosCopiados = np.zeros(shape=(idxDivisao), dtype=int)
        # Copia os cromossomos da mãe ou do pai até o índice máximo
        while idx < idxDivisao:
            filhos[filhosGerados][idx] = mae[idx] if filhosGerados == 0 else pai[idx]
            cromossomosCopiados[idx] = filhos[filhosGerados][idx]
            idx += 1

        # Então copia os cromossomos faltantes na ordem em que aparecem
        # no outro progenitor
        idxParada = idxDivisao
        if filhosGerados == 0:
            for cromossomo in pai:
                if not np.any(cromossomosCopiados[:] == cromossomo):
                    filhos[filhosGerados][idxParada] = cromossomo
                    idxParada += 1
        else:
            for cromossomo in mae:
                if not np.any(cromossomosCopiados[:] == cromossomo):
                    filhos[filhosGerados][idxParada] = cromossomo
                    idxParada += 1

        filhosGerados += 1

    return filhos[0], filhos[1]


def mutacao(geracao):
    for individuo in geracao:
        r = random.uniform(0, 1)
        if r < probabilidadeMutacao:
            # Escolhe randomicamente dois elementos para fazer swap
            idxPrimeiro = random.randint(0, numCidades - 1)
            idxSegundo = random.randint(0, numCidades - 1)

            # A mutação faz o swap dos elementos
            aux = individuo[idxPrimeiro]
            individuo[idxPrimeiro] = individuo[idxSegundo]
            individuo[idxSegundo] = aux
    return geracao


if __name__ == "__main__":
    main()
