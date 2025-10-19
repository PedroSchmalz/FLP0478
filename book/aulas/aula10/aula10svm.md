# Máquinas de Suporte de Vetores

Na seção anterior, trabalhamos o funcionamento do modelo de árvores de decisão. Agora, veremos o segundo modelo da aula, o modelo de **Máquinas de Suporte de Vetores**, ou *Support Vector Machines*

## *SVM*

Os métodos baseados em árvores consistem em estratificar e segmentar o espaço de preditores em um número de regiões. Na aula de hoje veremos o primeiro método, que serve de base para os outros, as **Árvores de Decisão** (*Decision Trees*)[^1]. Esses métodos fazem previsões para uma determinada observação usando o valor médio, ou resposta modal (de moda), das observações de treinamento para a região a que ela pertence. Métodos desse tipo possuem a principal vantagem de serem fáceis de interpretar, mas não são muito competitivos em termos de performance, especialmente em comparação com o *deep learning*.

### Árvores de Decisão



```{video} https://www.youtube.com/embed/_L39rN6gz7Y?si=wh_tmj_6hKx8GVtp
```

---


Vamos começar primeiro  com um exemplo de árvore de decisão no contexto da regressão. Ou seja, no contexto de um *outcome* numérico.


```{figure} ../aula10/images/islfig8.3.1.png
---
width: 100%
name: dtreg
align: center
---
Ilustração da Árvore de Decisão no contexto de Regressão. Fonte: James et al. ({cite}`james2023introduction`., p. 335)
```

A {numref}`Figura {number} <dtreg>` mostra o processo decisório em um modelo de árvore de decisão. Na figura, temos duas variáveis preditoras, $X_1$ e $X_2$, e a árvore vai se dividindo de acordo com os valores das duas. No começo da árvore, também conhecido como **Nó raiz** ou só **raiz**, a primeira decisão é com base no corte $t_1$: Valores menores que $t_1$ em $x_1$ jogam as observações para o lado esquerdo da árvore, e valores maiores vão para o lado direito. Do lado esquerdo da figura, a segunda decisão vem com base em $X_2$, com valores menores que o ponto de corte $t_2$ caindo para a primeira região $R_1$, e valores maiores que $t_2$ caindo na segunda região $R_2$. Cada observação vai passar por esses nós decisórios, chegando nos nós terminais que vão dar a previsão final ($R_1$, $R_2$, etc.). No caso das árvores de regressão, o valor previsto será a média das observações dentro dessa região.



```{figure} ../aula10/images/islfig8.3.2.png
---
width: 100%
name: dtregvar
align: center
---
Ilustração da Árvore de Decisão no contexto de Regressão. Fonte: James et al. ({cite}`james2023introduction`., p. 335)
```

A {numref}`Figura {number} <dtregvar>` mostra como fica a divisão das observações com base nas regiões, dentro do espaço de preditores. Com essa ilustração, fica mais fácil de ver como cada observação vai ser categorizada, e qual valor predito será utilizado. No entanto, quando temos mais variáveis, essa divisão não é tão clara assim. No contexto prático da aplicação do modelo em Python, você veria ele assim:


```{figure} ../aula10/images/geronfig6.4.png
---
width: 100%
name: dtgeronreg
align: center
---
Árvore de Regressão no Python. Fonte: Géron ({cite}`geron2022hands`.)
```

A {numref}`Figura {number} <dtgeronreg>` ilustra uma árvore de decisão de regressão que prediz valores numéricos contínuos, demonstrando na prática o processo de divisão binária recursiva baseado na minimização do RSS. A estrutura começa com um nó raiz no topo contendo todas as 200 amostras e se ramifica hierarquicamente através de nós de decisão internos que testam a variável $x1$ em diferentes pontos de corte, até chegar aos nós folha (terminais) coloridos que apresentam as predições finais — valores numéricos indicados por "value". Em cada nó, o MSE (Mean Squared Error) mede o erro quadrático médio naquela região, e quanto menor esse valor, mais homogêneos são os dados — observe como os nós folha apresentam MSE menores que os nós internos, indicando regiões mais puras. O algoritmo selecionou recursivamente em cada etapa o ponto de corte da variável $x1$ que mais reduziu o RSS/MSE, construindo uma estrutura hierárquica que particiona o espaço de preditores em regiões retangulares onde a predição é simplesmente a média dos valores observados naquela região.

### Como a árvore é construída?

As previsões da árvore de decisão são feitas com base na estratificação do espaço de preditores. Esse processo é dividido em dois passos:

1. O espaço de preditores (valores possíveis de $X_1, X_2, ..., X_n$) é divido em $J$ regiões distintas e sem sobreposição, chamadas de $R_j$. 

2. Para cada observação que cai na região $R_j$ é feita a mesma previsão, que é a média dos valores de treinamento que caem naquela determinada região. No contexto de classificação, é a moda dos valores de treinamento.

### Como as regiões $R_j$ são construídas?

As regiões $R_1,R_2,...,R_j$ não precisam ser retangulares como ilustrados na figura. O objetivo principal do modelo é encontrar as regiões $R_j$ que irão minimizar o erro das previsões, ou minimizar a função custo. Na quarta aula do curso, vimos que uma forma geral de ilustrar a função custo era por meio do seguinte mapeamento:


$$
L : (y, \hat{y}) \;\longrightarrow\; \mathbb{R}_{\ge 0}
$$

Vimos também que a função custo da regressão linear era o *Mean Squared Error*, ou Erro Quadrático Médio:


$$
MSE (X) = E[(y- f(x)²| x)]
$$

Para as árvores de decisão, usamos o **RSS** (*Residual Sum of Squares*, ou Soma dos Quadrados dos Resíduos). De maneira geral, ele é calculado da seguinte forma:


$$
\text{RSS} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

Onde:

- $y_i$ é o valor observado/real da i-ésima observação
- $\hat{y}_i$ é o valor previsto pelo modelo para a i-ésima observação
- $n$ é o número total de observações

Mas adaptado para as regiões da árvore de decisão:

$$
\text{RSS} = \sum_{j=1}^{J}\sum_{i \in R_j}(y_i - \hat{y}_{R_j})^2
$$

Onde:

- $J$ é o número de regiões/caixas na árvore
- $R_j$ é a j-ésima região
- $\hat{y}_{R_j}$ é a predição para a região $R_j$ (média dos valores naquela região)
- $i \in R_j$ indica todas as observações que pertencem à região $R_j$

Essa última equação soma todos os erros de previsão ao quadrado em todas as regiões criadas pela árvore. O somatório externo $\sum_{j=1}^{J}$ percorre cada uma das $J$ regiões em que o espaço de preditores está dividido, e o somatório interno $\sum_{i \in R_j}$ percorrre cada observação dentro das caixas $R_j$. Ou seja, para cada observação é calculada a diferença entre o valor real $y_i$ e o valor predito $\hat{y}$, ou $f(x)$, dentro daquela região. Essa diferença é então elevada ao quadrado, e depois soma o erro de todas as regiões, gerando um valor final do *RSS*. Quanto menor esse valor, melhor o resultado de treinamento do modelo.

### Divisão Binária Recursiva

No entanto, o cálculo do RSS dessa forma é computacionalmente inexequível, pois exigiria testar todas as partições possíveis em $J$ regiões. Portanto, as regiões são construídas usando a **Divisão Binária Recursiva**, ou *Recursive Binary Splitting*. Esse algoritmo é *top-down* e *greedy* (guloso). Ele é *top-down* pois começa pelo topo da árvore (raiz da árvore) e sucessivamente divide o espaço de preditores. Cada divisão separa a árvore em dois novos galhos. O processo é *greedy* por que a "melhor" divisão é decidida dentro daquele etapa (ou *split*), e não com base no passo anterior ou no próximo passo. Ou seja, em cada etapa, o algoritmo testa todos os preditores possíveis $X_i$, e todos os pontos de cortes $t$, criando duas regiões: uma onde $x_i < t$ e outra onde $x_i >= t$. O algoritmo então escolhe a combinação que gera o menor RSS naquela etapa (por isso *greedy*) e parte para a próxima divisão. 

### Podando a Árvore

Deixar a árvore crescer sem controle pode gerar boas previsões no banco de treinamento, mas pode gerar *overfitting* e baixa capacidade de generalização para outros dados. Para evitar esse problema, podemos aumentar o viés do modelo, reduzindo sua variância, ao limitar o crescimento da árvore, o que ira melhorar sua capacidade de generalização. Para fazer isso, uma estratégia é deixar a árvore crescer primeiro e depois ir podando ela, gerando sub-árvores. A pergunta passa a ser então "Como escolher a melhor sub-árvore"?

#### *Cost-Complexity Pruning*

*Cost-Complexity Pruning*, ou **Poda de Custo-Complexidade**, é uma das maneiras de escolher a melhor sub-árvore, reduzindo a variância do modelo. A fórmula do **Cost-Complexity Pruning (Poda de Custo-Complexidade)** é a seguinte:


$$
R_\alpha(T) = R(T) + \alpha|T|
$$

Onde:

- $R_\alpha(T)$ é a medida de custo-complexidade da árvore $T$
- $R(T)$ é o erro total da árvore (RSS para regressão ou impureza total para classificação)
- $\alpha \geq 0$ é o parâmetro de complexidade (hiperparâmetro de penalização)
- $|T|$ é o número de nós terminais (folhas) da árvore

Expandindo para incluir as regiões, temos:

$$
R_\alpha(T) = \sum_{m=1}^{|T|}\sum_{i \in R_m}(y_i - \hat{y}_{R_m})^2 + \alpha|T|
$$

A equalão é a função de custo-complexidade que balanceia o erro de previsão da árvore (o RSS ou impureza do nó) com sua complexidade. A primeira parte da equação é exatamente o RSS que vimos antes. A segunda parte $\alpha|T|$ é a penalidade de complexidade, onde o $|T|$ conta quantas folhas a árvore tem e $\alpha$ é um parâmetro de ajuste que controla o quanto você quer penalizar árvores grandes. Quando $\alpha = 0$, a equação é igual ao RSS, e não há nenhuma penalidade para árvores grandes demais.

**Interpretação:**

- Quando $\alpha = 0$, a fórmula se reduz apenas ao RSS e você mantém a árvore completa
- Quando $\alpha$ aumenta, a penalidade por ter muitos nós terminais cresce, forçando uma árvore mais simples (podada)
- O objetivo é encontrar o valor de $\alpha$ que minimiza $R_\alpha(T)$, balanceando erro de predição e complexidade do modelo

### Árvores de classificação

As árvores de classificação aplicam a mesma estratégia das árvores de regressão — particionar recursivamente o espaço de preditores através da **divisão binária recursiva** — mas com o objetivo de separar observações em classes discretas. Enquanto as árvores de regressão utilizam o RSS (Residual Sum of Squares) como critério de divisão, as árvores de classificação empregam **medidas de impureza** para avaliar a qualidade das separações.


```{figure} ../aula10/images/islfig8.6.png
---
width: 100%
name: dtclass
align: center
---
Ilustração da Árvore de Classificação. Fonte: James et al. ({cite}`james2023introduction`., p. 340)
```

A {numref}`Figura {number} <dtclass>` ilustra uma árvore de decisão de classificação binária que prediz a presença ou ausência de doença cardíaca (classes "Yes" e "No"), demonstrando na prática o processo de divisão binária recursiva que discutimos anteriormente. A estrutura começa com um nó raiz no topo que avalia uma característica clínica (Thal:a) e se ramifica hierarquicamente em nós de decisão internos que testam outras variáveis médicas, como frequência cardíaca máxima (MaxHR) e tipo de dor no peito (ChestPain), até chegar aos nós folha (terminais) coloridos que apresentam as classificações finais. Cada nó interno mostra estimativas de probabilidade das classes (indicadas por "Ca + 0.5"), refletindo a proporção de observações de cada categoria naquela região. O algoritmo guloso e top-down selecionou em cada etapa a variável e o ponto de corte que mais reduziram a impureza.

Em cada **nó terminal** (ou nó folha), a previsão pode ser feita de duas formas: (1) atribuindo a **classe modal** (mais frequente) entre as observações da folha; ou (2) fornecendo uma **estimativa de probabilidade** para cada classe, baseada nas proporções observadas na região.

**Estimativa de probabilidade na folha:**

$$
\hat{p}_k = \frac{n_k}{n}
$$

onde $n_k$ é o número de observações da classe $k$ na folha e $n$ é o total de observações na folha.

#### Critérios de Impureza

A **impureza** mede o grau de mistura de classes em um nó: um nó **puro** (impureza zero) contém apenas observações de uma única classe, enquanto um nó **impuro** contém uma mistura de classes. O algoritmo busca divisões que reduzem a impureza, tornando os nós filhos mais homogêneos. As três principais métricas são:

**Índice de Gini** (mais utilizado):

$$
G = 1 - \sum_{k} \hat{p}_k^2
$$

Mede a probabilidade de classificação incorreta aleatória. Varia de 0 (puro) a aproximadamente 0.5 (máxima impureza em problemas binários).

**Entropia** (baseada na teoria da informação):

$$
H = -\sum_{k} \hat{p}_k \log(\hat{p}_k)
$$

Quantifica a incerteza ou desordem no nó. Quanto maior a entropia, maior a mistura de classes.

**Erro de classificação** (misclassification error):

$$
E = 1 - \max_k \hat{p}_k
$$

Proporção de observações que não pertencem à classe majoritária. É menos sensível a mudanças na distribuição das classes.

Ao avaliar cada possível divisão durante a **divisão binária recursiva**, a árvore calcula a **impureza ponderada** das duas regiões geradas (esquerda e direita) e escolhe o split que **mais reduz** a impureza média :

$$
I_{\text{split}} = \frac{n_{L}}{n} I(L) + \frac{n_{R}}{n} I(R)
$$

onde $I(\cdot)$ é a medida de impureza escolhida (Gini, Entropia ou Erro), $n_L$ e $n_R$ são os tamanhos das partições esquerda e direita, e $n$ é o total de observações antes da divisão. O algoritmo testa todos os preditores e todos os pontos de corte possíveis, selecionando aquele que minimiza $I_{\text{split}}$ em cada etapa, de forma gulosa e top-down.


```{figure} ../aula10/images/geronfig6.1.png
---
width: 100%
name: dtgeron
align: center
---
Árvore de decisão no Python. Fonte: Géron ({cite}`geron2022hands`.)
```

A {numref}`Figura {number} <dtgeron>` ilustra uma árvore de decisão de classificação treinada no conjunto de dados Iris, que classifica flores em três espécies (setosa, versicolor e virginica) através de uma estrutura hierárquica de decisões. O nó raiz no topo avalia se o comprimento da pétala é menor ou igual a 2.45 cm e, quando verdadeiro, leva diretamente a um nó folha laranja completamente puro (Gini = 0.0) contendo todas as 50 amostras da classe setosa, enquanto o ramo falso conduz a um segundo nó de divisão que avalia a largura da pétala (≤ 1.75 cm) para separar as 100 amostras restantes. Esse segundo nó de divisão gera dois nós folha: um verde classificando 54 amostras como versicolor (com pequena impureza de Gini = 0.168 devido a 5 virginicas misturadas) e um roxo classificando 46 amostras como virginica (com Gini = 0.043, quase puro exceto por 1 versicolor). A estrutura demonstra como a árvore utiliza apenas duas características (comprimento e largura da pétala) e dois pontos de corte para separar eficientemente as três classes, com os valores de Gini em cada nó indicando a pureza da classificação e o número de amostras mostrando a distribuição dos dados em cada divisão.



```{admonition} 💬 Com a palavra, os autores:
:class: quote
"Então, você deveria usar impureza Gini ou entropia? A verdade é que, na maioria das vezes, isso não faz uma grande diferença: eles levam a árvores semelhantes. A impureza Gini é ligeiramente mais rápida de calcular, então é uma boa escolha padrão. No entanto, quando eles diferem, a impureza Gini tende a isolar a classe mais frequente em seu próprio ramo da árvore, enquanto a entropia tende a produzir árvores ligeiramente mais balanceadas."
({cite}`geron2022hands`., Capítulo 6, tradução nossa)
```

## Conclusão


Nesta aula exploramos as árvores de decisão, um dos métodos fundamentais de machine learning que serve de base para algoritmos mais sofisticados como Random Forests e Gradient Boosting. Aprendemos que tanto árvores de regressão quanto de classificação compartilham a mesma estratégia central: particionar recursivamente o espaço de preditores em regiões distintas através do algoritmo guloso e top-down de divisão binária recursiva, onde cada divisão busca localmente a melhor separação dos dados sem considerar o impacto global. Vimos que as árvores de regressão minimizam o RSS (Residual Sum of Squares) para encontrar as melhores divisões, enquanto as árvores de classificação utilizam medidas de impureza como o índice Gini ou entropia para avaliar a qualidade das separações, buscando criar nós filhos mais homogêneos e puros. Um conceito crucial que abordamos foi a poda de custo-complexidade (Cost-Complexity Pruning), que introduz o hiperparâmetro 
α
α para balancear erro de predição e complexidade do modelo, conectando-se diretamente com os conceitos de ajuste de hiperparâmetros da aula anterior e demonstrando como controlar o trade-off entre viés e variância para evitar overfitting. A principal vantagem das árvores de decisão é sua excepcional interpretabilidade: a estrutura hierárquica de regras "se-então" permite que profissionais de diversas áreas compreendam facilmente como o modelo toma decisões, tornando-as ideais para contextos onde explicabilidade é crucial, como diagnósticos médicos ou decisões de crédito. No entanto, como mencionado, árvores individuais geralmente não são tão competitivas em termos de performance pura quando comparadas a métodos mais modernos, tendendo a ter alta variância e sendo sensíveis a pequenas mudanças nos dados de treinamento. Na próxima seção, exploraremos o Support Vector Machine (SVM), um algoritmo com abordagem completamente diferente que busca maximizar margens de separação entre classes, e introduziremos o TF-IDF, uma técnica de ponderação de texto que será essencial para aplicar SVMs em problemas de classificação textual.


## Notas

[^1]: Outros métodos baseados em árvores incluem: Random Forests — ensembles de árvores construídas por amostragem bootstrap que reduzem a variância; Bagging (Bootstrap Aggregating) — agregação de várias árvores independentes; Extra-Trees (Extremely Randomized Trees) — similar a Random Forest com divisão mais aleatória; Boosting — métodos sequenciais que corrigem erros (ex.: AdaBoost); Gradient Boosting Machines (GBM) — otimização por gradiente de árvores fracas; implementações populares e otimizadas: XGBoost, LightGBM e CatBoost; Isolation Forest — uso de árvores para detecção de anomalias; e abordagens mais especializadas como Conditional Inference Trees e Bayesian Additive Regression Trees (BART). Cada família tem trade-offs distintos entre viés, variância, interpretabilidade e velocidade.

