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

O método "Deixe um de fora" (***Leave-one-out***), ou "Exclua um", consiste em dividir o banco também em duas partes. No entanto, ao invés de deixar 10%, ou 20%, para teste, uma única observação ($x_1,y_1$) é usada como validação, e o resto é usado para o treinamento. O método é então repetido com $(x_2,y_2), (x_3,y_3) ..., (x_n,y_n)$ de fora. Ou seja, repetimos o processo para cada observação no banco de dados, cada vez deixando uma única observação de fora. Com isso, teremos *n* estimativas de erro. No caso da regressão, obteremos *n* MSEs.

$$CV_{(n)} = \dfrac{1}{n} \sum_{i=1}^{n} \text{MSE}_{i}$$


A {numref}`Figura {number} <LOOCV>` ilustra como funciona esse processo.

```{figure} ../aula8/images/islfig.5.3.png
---
width: 100%
name: LOOCV
align: center
---
Ilutração do método *Leave-one-out* (LOOCV). Fonte: James et al. ({cite}`james2023introduction`., p. 205)
```

As vantagens desse método com relação ao primeiro são as seguintes: 1) ele tem menos viés, gerando menor superestimação do erro no teste. 2) Não é aleatório, tendo resultados estáveis, já que sempre seguirá a mesma ordem dentro do banco de dados. No entanto, possui uma desvantagem clara: O seu gasto computacional e de tempo é muito alto. Para um banco de *N* observações, teremos que treinar o modelo *N* vezes, e isso é impraticável com bancos grande e com modelos que já exigem mais, como é o caso dos modelos de Aprendizado Profundo. Portanto, na prática, se usa um método que é um caso especial do *LOOCV*: o ***K-fold Cross-validation***.

## *K-fold Cross-Validation*



```{video} https://www.youtube.com/embed/rSGzUy13F_0?si=LYRw0bYA7rfJdio8
```

---

Uma alternativa menos computacionalmente exigente ao *LOOCV* é o ***K-fold***. Nessa estratégia, definimos em quantas repartições, ou "dobras" (daí o *fold*), dividíremos o banco de treino. Com base nessas $k$ dobras, iniciamos o treinamento com o primeiro *fold* sendo usado para validação e o resto para treino, repetindo isso $k$ vezes. Obtendo, por tanto, $k$ métricas de erro. Nossa fórmula para o MSE do *LOOCV* vira, então:

$$CV_{(k)} = \dfrac{1}{k} \sum_{i=1}^{k} \text{MSE}_{i}$$

A {numref}`Figura {number} <KFOLD>` ilustra como funciona esse processo.


```{figure} ../aula8/images/islfig.5.5.png
---
width: 100%
name: KFOLD
align: center
---
Ilutração do método *K-fold*. Fonte: James et al. ({cite}`james2023introduction`., p. 207)
```

Não só o método *k-fold* é menos custoso computacionalmente, ele também possui vantagens de viés com relação ao *LOOCV*: NA validação cruzada, “viés” é o erro introduzido porque cada modelo é treinado com um subconjunto dos dados, não com o conjunto inteiro.

No LOOCV (Leave-One-Out) cada split treina com n – 1 observações, praticamente todo o conjunto. Isso torna a estimativa de erro muito pouco enviesada, mas cada fold avalia o modelo num único ponto, o que gera alta variância.

No k-fold com k pequeno (por exemplo k = 5 ou 10) cada modelo treina com apenas (k – 1)/k do conjunto; como perde um pouco mais de dados para teste, o erro médio tende a ser levemente mais alto (maior viés) do que no LOOCV.

A “vantagem” é justamente esse pequeno aumento de viés: ele suaviza a avaliação e, combinado com o uso repetido de partes maiores do conjunto de teste, reduz drasticamente a variância da estimativa. Na prática, esse equilíbrio entre “um pouco mais de viés” e “muito menos variância” faz com que o k-fold produza uma previsão de desempenho mais estável e geralmente mais próxima do erro real em dados novos. 


Na {numref}`Figura {number} <KFOLDvsLOOCV>` estão as estimativas de validação cruzada e as taxas reais de erro de teste obtidas ao aplicar splines de suavização aos conjuntos de dados simulados mostrados nas Figuras 2.9–2.11 do Capítulo 2. O MSE de teste verdadeiro é exibido em azul. As linhas preta tracejada e laranja contínua representam, respectivamente, as estimativas do LOOCV e da validação cruzada com 10 dobras. Nos três gráficos, as duas estimativas de validação cruzada são muito semelhantes. No painel da direita, o MSE de teste verdadeiro e as curvas de validação cruzada são quase idênticos. No painel central, os dois conjuntos de curvas coincidem nos menores graus de flexibilidade, mas as curvas de validação cruzada superestimam o MSE do conjunto de teste para graus mais altos de flexibilidade. No painel da esquerda, as curvas de validação cruzada apresentam o formato geral correto, porém subestimam o MSE de teste verdadeiro.

```{figure} ../aula8/images/islfig5.6.png
---
width: 100%
name: KFOLDvsLOOCV
align: center
---
Erro quadrático médio (MSE) de teste verdadeiro e estimado para os conjuntos de dados simulados nas Figuras 2.9 (esquerda), 2.10 (centro) e 2.11 (direita). O MSE de teste verdadeiro é mostrado em azul, a estimativa do LOOCV aparece como uma linha tracejada preta e a estimativa da validação cruzada com 10 dobras em laranja. As cruzes indicam o mínimo de cada uma das curvas de MSE. Fonte: James et al. ({cite}`james2023introduction`., p. 209)

```



```{admonition} 💬 Com a palavra, os autores:
:class: quote
"Ao realizar validação cruzada k-fold — por exemplo, com k = 5 ou k = 10 — obtém-se um nível intermediário de viés, porque cada conjunto de treinamento contém aproximadamente (k − 1)n/k observações: menos do que no LOOCV, mas muito mais do que na estratégia de conjunto de validação único. Portanto, sob a ótica de redução de viés, o LOOCV é preferível ao k-fold. Contudo, viés não é a única preocupação; também importa a variância do procedimento. O LOOCV apresenta variância mais alta do que o k-fold com k < n. Por quê? No LOOCV, calculamos a média das saídas de n modelos ajustados sobre conjuntos quase idênticos, gerando resultados altamente correlacionados. Já no k-fold com k < n, fazemos a média das saídas de k modelos treinados em conjuntos que se sobrepõem menos, resultando em correlações menores. Como a média de variáveis muito correlacionadas tem variância maior que a média de variáveis menos correlacionadas, a estimativa do erro de teste via LOOCV tende a exibir variância mais alta que a obtida pelo k-fold. Em suma, há um trade-off viés-variância na escolha de k: valores como k = 5 ou k = 10 são usados com frequência, pois fornecem estimativas do erro de teste que não sofrem nem de viés excessivo nem de variância muito alta."
({cite}`james2023introduction`., p. 209, tradução nossa)
```

## Conclusão




## Notas

[^1]: A capacidade de generalização é a habilidade de um modelo de aprendizado de máquina manter alta precisão quando recebe dados que nunca fizeram parte do treinamento, porque aprendeu os padrões subjacentes em vez de memorizar casos específicos. Quando o modelo é complexo demais para a quantidade ou a qualidade dos dados, ele tende ao *overfitting*. Ou seja, acerta o treinamento, mas falha nos exemplos novos; E se for simples demais, ocorre *underfitting*, pois não captura nem mesmo os padrões básicos.