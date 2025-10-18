# Laboratório 09

<div style="text-align:center;">

**Universidade de São Paulo**  
**Faculdade de Filosofia, Letras e Ciências Humanas**  
**Departamento de Ciência Política**

**FLP0478 - Processamento de Língua Natural Aplicada para Ciência Política e Análise de Políticas Públicas (2025)**

**2° semestre / 2025**

**Laboratório 9**

**Professora Lorena G. Barberia**

**Prazo: 20 de outubro, 2025**


</div>

---

A tarefa deste laboratório será a de montar o seu script em Python do zero, desde a descrição dos seus dados até a execução de um modelo logístico multinomial, classificando o posicionamento dos tweets (todas as 3 classes) com relação às vacinas de COVID-19. O trabalho deve estar dividido nas seguintes seções. Vocês devem entregar um relatório de até 08 páginas (10 com desafio) em .doc ou .pdf, junto do seu script em Python (.ipynb), no moodle na caixa do "Laboratório 9" de seu respectivo turno.

## 1. Introdução

## 2. Descrição dos Dados

- Tweets Relevantes/Irrelevantes
- Distribuição das Classes em Posicionamento
- Possíveis consequências da distribuição

## 3. Modelo Multinomial + K-fold (10 folds)

a. Divisão Treino/Teste;

b. Quais os hiperparâmetros? (e.g. n-gram, solver, penalty, and regularization strength);

c. Distribuição do Precision, Recall e F1-score ao longo dos 10 folds de validação, para cada classe;

d. Acurácia, Matriz de Confusão e Relatório de Classificação no Banco de Teste;

e. Conclusão sobre os resultados.

## 4. Tuning Dos Hiperparâmetros

a. Estabeleça um espaço de hiperparâmetros para o modelo;

b. Faça o tuning usando grid search (10 folds). Quais foram os melhores hiperparâmetros?

c. Treine o modelo usando os melhores hiperparâmetros com uma validação cruzada com 10 folds. Quais foram os resultados de validação e teste?

d. O resultado mudou com relação à seção 3? O que você conclui?

## 5. DESAFIO (opcional) - Escolha outro modelo de aprendizado de máquina (tudo menos Deep Learning):

a. Apresente o modelo e os principais hiperparâmetros relacionados;

b. Escolha um método de busca exaustiva (menos o manual) e tente obter os melhores hiperparâmetros. Treine o melhor modelo e apresente os resultados de validação/teste;

c. Escolha um método com otimização Bayesiana e faça o mesmo.

d. Compare os resultados obtidos nos dois métodos e tire uma conclusão.
