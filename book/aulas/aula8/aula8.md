# Sobreajuste, Reamostragem e Validação dos Resultados

Na última aula, vimos que a transição da classificação tradicional com variáveis numéricas para a classificação de textos exige primeiro **converter a linguagem em números**, e foi aí que aprofundamos os modelos de linguagem *n-gram*. Discutimos como unigramas, bigramas e trigramas capturam diferentes níveis de contexto e por que a suposição de Markov e a Regra da Cadeia permitem estimar probabilidades de sequências mesmo em corpus limitados. Essa base probabilística ficou clara quando mostramos que **classificadores como a regressão logística**, apresentados esta semana, continuam seguindo o mesmo princípio: atribuir pesos às features—neste caso, contagens ou frequências de *n-grams*—para modelar $Pr(Y\! =\! 1\mid X)$ por meio da função sigmoide, ou do softmax quando temos mais de duas classes. Finalmente, amarramos essas ideias ao processo de treinamento: definimos a perda em cross-entropy, vimos como o gradiente descende essa função e por que hiperparâmetros como a learning rate influenciam a convergência do modelo. Em suma, o encontro uniu teoria de modelos de linguagem e prática de classificação supervisionada, mostrando que **entender *n-grams* é o primeiro passo para aplicar algoritmos como regressão logística a tarefas de sentimento, spam ou tópicos**. 

Na aula de hoje, iremos discutir um problema que não é único ao texto em Aprendizado de Máquina: o Sobreajuste (ou ***overfitting***), e como identificá-lo e evitá-lo. O problema decorre do fato de que muitos métodos estatísticos/de Aprendizado de Máquina são facilmente adaptáveis e capazes de modelar relações complexa (Mesmo antes da difusão do *Deep Learning*!). No entanto, podem acabar enfatizando demais características dos dados de treinamento que não estão no teste (ou no mundo), o que vai impactar sua capacidade de generalização para outros casos[^1]. Por isso, precisamos garantir que nossos modelos são confiáveis e podem ser usados para previsão. Sem essa confiança, os modelos são inúteis.


## Sobreajuste (*Overfitting*)

Existem técnicas de aprendizado estatístico que são capazes de "aprender" tão bem as características de um banco de treinamento que podem fazer previsões corretas para todas as observações. Esse tipo de modelo é conhecido como um modelo sobreajustado, ou *over-fit*, e provavelmente não vai se sair muito bem em novas amostras. Vamos imaginar uma tarefa de classificação bem simples em que há somente dois preditores: A e B, e um modelo deve classificá-la entre a classe azul e a classe vermelha (simulação).

```{figure} ../aula8/images/kkfig.4.2.png
---
width: 100%
name: figkkclass
align: center
---
Exemplo de um banco de treinamento com duas classes e preditores. Os painéis mostram dois modelos de classificação e seus limites de classificação. Fonte Kuhn and Kjell (p.63, {cite}`kuhn2018applied`.)
```

Como mostra A {numref}`Figura {number} <figkkclass>`, o modelo 1 seria um modelo sobreajustado, pois tenta definir os limites para a classificação muito próximos também aos casos atípicos. Isto é, os casos azuis que estão para valores dos preditores maiores do que 0.5 (ambos preditores). Com esse sobreajuste, com certeza o modelo apresentaria ótimos resultados no banco de treinamento, mas performaria muito mal em novas observações, justamente por tentar se aproximar muito de como as observações se comportaram no treinamento. O segundo modelo, apesar de admitir mais erro durante o treinamento, provavelmente apresentaria melhores resultados em observações não vistas e, portanto, melhor capacidade de generalização. Esse é o ponto principal do problema: Queremos conseguir identificar quando nosso modelo está sobreajustado, e qual modelo terá melhores resultados em novas observações. Nós vimos isso acontecer anteriormente na quinta aula, quando avaliamos o resultado de treinamento e teste de alguns modelos com base na sua flexibilidade, na {numref}`Figura {number} <flexteste>`:

```{figure} ../aula5/images/fig2.9.png
---
width: 100%
name: flexteste
align: center
---
À esquerda: dados simulados a partir de f, mostrados em preto. Três estimativas de f são exibidas: a linha de regressão linear (curva laranja) e dois ajustes por splines de suavização (curvas azul e verde).

À direita: Erro Quadrático Médio de treinamento (curva cinza), EQM de teste (curva vermelha) e EQM mínimo possível de teste entre todos os métodos (linha tracejada). Os quadrados representam os EQMs de treinamento e de teste para os três ajustes mostrados no painel da esquerda. Fonte: James et al. ({cite}`james2023introduction`., p. 29).
```

Nesse caso, temos um modelo sobreajustado, que é representado na linha verde: Ele segue bem de perto a variação estocástica de cada observação, e consegue ótimos resultados no treinamento (quadrado verde, linha cinza). No entanto, vemos que seu resultado é ruim para o banco de teste (linha vermelha), só não sendo pior do que a regressão linear, que nesse caso foi um modelo subajustado (*under-fit*). Nas nossas aplicações, sempre almejamos o modelo azul (corretamente ajustado), e, apesar de não sabermos o verdadeiro erro mínimo possível, podemos comparar diversos modelos e suas estimativas com métodos de reamostragem, escolhendo o melhor modelo dentre os testados. 


Vamos tentar imaginar como isso ocorreria com texto em um exemplo simples. Considere um mini‐conjunto de tweets sobre um filme. Temos o seguinte banco de Treino fictítico para uma classificação de sentimento (6 frases, rótulos entre parênteses):

1. “Amei muito este filme!” (+)
2. “Filme incrível, recomendo!” (+)
3. “Obra-prima total.” (+)
4. “Odiei esse filme demais.” (−)
5. “Péssimo enredo, detestei.” (−)
6. “Terrível do começo ao fim.” (−)

Para transformar texto em números aplicamos um modelo *bag-of-words* de **trigramas**: cada sequência de três palavras vira uma feature, gerando dezenas de colunas muito específicas (“amei muito este”, “esse filme demais” etc.). Treinamos uma **regressão logística** sem regularização; como há exatamente um trigrama exclusivo para cada frase, o algoritmo pode facilmente encontrar pesos que se encaixam perfeitamente e atingir 100% de acurácia no conjunto de treino. No entanto, imaginemos o seguinte banco de teste:

Teste (2 frases nunca vistas):

7. “Adorei esse filme.” (+)
8. “Horrível, não gostei.” (−)

Nenhum trigama do teste aparece no treino, logo o modelo poderá atribuir probabilidades aleatórias e classificar ambas como negativas ou positivas, resultando em apenas 50% de acerto (ou errar ambas). O sistema **memoriza ruído em vez de aprender padrões gerais**, o que é uma evidência de sobreajuste. Poderíamos trocar para unigramas ou aplicar regularização para reduzir a complexidade e recuperar a capacidade de generalização. Mas precisamos saber identificar quando estamos com sobreajuste.



```{admonition} 💬 Com a palavra, os autores:
:class: quote
"Ao fazer o processamento computacional de textos escritos, a definição de que tipo de unidade de processamento se quer buscar/estudar parece estar atrelada às necessidades da tarefa ou trabalho pretendidos. Geralmente, considera-se que uma palavra é, simplesmente, uma unidade grafológica delimitada, nas línguas europeias, entre espaços em branco na representação gráfica, ou entre um espaço em branco e um sinal de pontuação. Essa é uma definição bastante concreta, e bastante prática. No entanto, ao pensarmos em nossos modelos computacionais e suas aplicações no mundo, é importante nos aprofundarmos um pouco mais na conceituação do que é uma palavra e nas possibilidades de processamento e implicações das decisões tomadas no pré-processamento dos corpora."
({cite}`caseli_nunes_pln_2024`., p. 68, tradução nossa)
```

## Reamostragem e Validação Cruzada

```{video} https://www.youtube.com/embed/fSytzGwwBVw?si=1si6a5JK_COrLkHa
```

---

Justamente para que sejamos capazes de identificar quando estamos na situação de sobreajuste que surgiram os métodos de Reamostragem. **Métodos de Reamostragem**, ou *Resampling*, são ferramentas indispensáveis na estatística e no aprendizado de máquina, e consistem em sortear amostras repetidas de um determinado conjunto de dados de treinamento, reajustando o modelo em cada nova amostra e obtendo informações adicionais sobre seu ajuste e incerteza. 

Métodos de reamostragem podem ser utilizados para estimar o erro de um modelo ou para selecionar o nível de flexibilidade (como vimos na {numref}`Figura {number} <flexteste>`). O processo de avaliar a performance de um modelo é conhecido como **Validação do Modelo**, e o processo de escolher hiperparâmetros e/ou flexibilidade do modelo é conhecido como **Seleção do modelo** (ou *hyperparameter tuning*).

Durante o curso, discutimos muito o valor de erro do modelo no banco de teste. o Erro de Teste é a média dos erros que resultam das previsões do modelo em uma nova observação, que não foi vista durante o treinamento. No entanto, a distinção entre banco de treino e teste pode ser um pouco ilusória: Na verdade, não temos um banco anotado separado só para teste, mas um banco de treinamento que foi repartido para treinamento e teste. Na ausência do banco de teste ideal, usamos métodos de validação para verificar a capacidade de generalização de um modelo.

## Abordagem do Conjunto de Validação/Teste


```{video} https://www.youtube.com/embed/ngrOYWgJjb4?si=V1M2QpZVbXRwqcyl
```


---

Ao longo do curso, já usamos um método de validação. Dividimos um banco de treinamento em duas partes: treino e teste. O banco de teste também pode ser chamado de banco de validação. Com esse método, treinamos o modelo no banco de treinamento e avaliamos sua performance no banco de teste. A {numref}`Figura {number} <treinoteste>` ilustra a divisão neste método.


```{figure} ../aula8/images/islfig5.1.png
---
width: 100%
name: treinoteste
align: center
---
Divisão entre o banco de treino (azul) e teste/validação (bege). Fonte: James et al. ({cite}`james2023introduction`., p. 203)
```

Uma das principais desvantagens desse método sozinho é que, ao repetirmos o processo de separação entre treino e teste, obteremos resultados bem diferentes. A figura {numref}`Figura {number} <testevar>` mostra a variância das estimativas de erro obtida com diferentes divisões entre treino e teste. Fica claro que a aleatoriedade contida nessa divisão pode afetar bastante nossa estimativa: não temos como saber quando estaremos na linha amarela (menor MSE) ou na linha laranja (maior MSE). Portanto, precisaremos de outros métodos para termos mais confiança em nossos modelos.


```{figure} ../aula8/images/islfig.5.2.png
---
width: 100%
name: testevar
align: center
---
Uma única divisão entre treino e teste (Esquerda) e múltiplos resultados em várias divisões (Direita). Fonte: James et al. ({cite}`james2023introduction`., p. 204)
```

Em resumo, o método de validação só bom com o banco de teste tem duas desvantagens principais: 1) A estimativa de erro será única, e pode variar bastante dependendo de quais observações entraram no banco de teste. 2) O modelo treinará com menos observações (80 ou 90%), gerando resultados que podem supersestimar o erro de teste.


## Abordagem *Leave-one-out* (LOOCV)









## Notas

[^1]: A capacidade de generalização é a habilidade de um modelo de aprendizado de máquina manter alta precisão quando recebe dados que nunca fizeram parte do treinamento, porque aprendeu os padrões subjacentes em vez de memorizar casos específicos. Quando o modelo é complexo demais para a quantidade ou a qualidade dos dados, ele tende ao *overfitting*. Ou seja, acerta o treinamento, mas falha nos exemplos novos; E se for simples demais, ocorre *underfitting*, pois não captura nem mesmo os padrões básicos.