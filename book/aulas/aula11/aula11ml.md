# Métodos *Ensemble*

Na última aula vimos três componentes principais de técnicas de aprendizado de máquina: começamos explorando as Árvores de Decisão, um método fundamental baseado em estratificação do espaço de preditores que funciona através de divisão binária recursiva, onde o algoritmo guloso testa todos os pontos de corte possíveis para minimizar o RSS (em regressão) ou a impureza (medida pelo índice de Gini ou entropia) em classificação, criando estruturas hierárquicas fáceis de interpretar onde cada observação percorre nós decisórios até chegar em regiões/folhas que fornecem a predição final. Vimos também que o crescimento descontrolado das árvores leva a *overfitting*, por isso introduzimos a poda de custo-complexidade, um hiperparâmetro α que penaliza árvores muito grandes, conectando-se diretamente com os conceitos de ajuste de hiperparâmetros da aula anterior e demonstrando o trade-off entre viés e variância. 

Em seguida, exploramos as Máquinas de Vetores de Suporte (SVM), que surgem como generalização do classificador de margem máxima, começando com o conceito de hiperplano em espaços p-dimensionais para dividir observações em classes, depois relaxando essa restrição rígida através dos classificadores de vetores de suporte com margem "suave" controlada pelo hiperparâmetro C (orçamento de violação), e finalmente chegando ao SVM completo que usa o kernel trick para lidar elegantemente com relações não-lineares, transformando implicitamente os dados para espaços de dimensão superior onde se tornam separáveis sem criar manualmente termos polinomiais. 

Por fim, estudamos o TF-IDF (Term Frequency-Inverse Document Frequency), uma medida estatística que captura a importância de uma palavra em um documento específico em relação a uma coleção, multiplicando dois componentes: a frequência do termo (TF) com transformação logarítmica para desacelerar o crescimento, e a frequência inversa de documento (IDF) que penaliza palavras comuns no corpus mas recompensa palavras raras, permitindo que trabalhem bem com n-gramas e sendo especialmente compatível com SVMs em problemas de classificação textual em espaços de alta dimensionalidade.

Na aula de hoje vamos aprofundar nossa compreensão dos métodos de machine learning explorando duas técnicas fundamentais de aprendizado em conjunto: o Bagging (Bootstrap Aggregating) e o Boosting. Veremos também como esses métodos ensemble diferem radicalmente em suas estratégias: enquanto o Bagging treina múltiplos modelos de forma independente e paralela em subconjuntos aleatórios dos dados através de amostragem com reposição, reduzindo principalmente a variância e funcionando particularmente bem com modelos complexos como árvores profundas, o Boosting adota uma abordagem sequencial onde cada novo modelo é treinado com o objetivo específico de corrigir os erros cometidos pelos modelos anteriores, reduzindo o viés através de uma combinação ponderada de modelos fracos. O Bagging combina as previsões dos modelos via média ou votação, tornando o resultado mais estável e robusto, enquanto o Boosting atribui pesos maiores aos exemplos que foram classificados incorretamente, forçando os modelos posteriores a focar nesses casos difíceis. Entender essas duas abordagens distintas é essencial para escolher a estratégia correta para seus dados e objetivos específicos, pois cada uma oferece diferentes trade-offs entre redução de erro, complexidade computacional e risco de overfitting.

## Bagging


```{video} https://www.youtube.com/embed/Xz0x-8-cgaQ?si=kh22plivN21X6OCA
```

Um método *ensemble* é aquele que combina modelos como "tijolos" de forma a obter um único modelo com melhores resultados. Hoje iremos discutir dois grupos de métodos principais: *Bagging* e *Boosting*. O ***Bagging***, ou *Bootstrap Aggregation*, é um procedimento geral utilizado para reduzir a variância de um modelo de aprendizado estatístico. O *Bagging* se baseia no método de *Bootstrap*, e procura reduzir a variância do modelo pegando diversos conjuntos de treinamento da população, construindo um modelo em cada conjunto separadamente, e pegando a média das previsões de todos os modelos em todos os conjuntos. Em outras palavras, estimamos $\hat{f}^1(x)$, $\hat{f}^2(x)$, ... $\hat{f}^B(x)$ usando $B$ subconjuntos separados de treinamentos, pegando a média deles de forma a obter um únoc modelo com pouca variância, dado por:

$$
\hat{f}_{\text{avg}}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat{f}^b(x)
$$


No entanto, não temos a possibilidade de extrair amostras aleatórias de uma população, então fazemos essa amostragem com base no nosso banco de treinamento. O nosso conjunto de treinamento é então dividido em $B$ subconjuntos, e a equação se torna:


$$
\hat{f}_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat{f}^\text{*b}(x)
$$

É dessa maneira que funciona o bagging, e ele pode ser aplicado com diversos modelos de aprendizado estatístico. Para o caso específico de classificação, a previsão final é dada pelo "voto da maioria": A previsão final será aquela mais comum entre as $B$ previsões dos diferentes subconjuntos.


```{figure} ../aula11/images/datascientestbag.png
---
width: 100%
name: bagging
align: center
---
Processo de *Bagging* de modelos. Fonte: [DataScientest](https://datascientest.com/en/bagging-vs-boosting).)
```

A {numref}`Figura {number} <bagging>` ilustra como funciona o processo de bagging na prática: Os dados (CSV) são divididos em subconjuntos (*data*). Os classificadores são então treinados em cada um desses subconjuntos. O modelo final resulta da combinação dos diversos classificadores em uma única coisa.





## Conclusão


Nesta au



