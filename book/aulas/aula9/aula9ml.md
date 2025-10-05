# Ajuste de Hiperparâmetros

Na última aula vimos que o sobreajuste, ou overfitting, é um problema comum em aprendizado de máquina, onde o modelo aprende características específicas demais dos dados de treino, o que prejudica sua capacidade de generalizar para dados novos. Usamos exemplos práticos, como um modelo que se ajusta perfeitamente a pontos atípicos, apresentando desempenho muito bom no treino, mas ruim no teste. Discutimos também vários métodos para identificar e evitar esse problema, incluindo técnicas de reamostragem como validação simples, leave-one-out e, em especial, o k-fold cross-validation, que equilibra viés e variância na estimativa do erro do modelo. Aprendemos que o k-fold é uma boa prática porque oferece estimativas mais estáveis e realistas do desempenho do modelo, ajudando a garantir que ele não apenas “decore” o treino, mas realmente capture padrões que funcionam em novas situações. Assim, fechamos o ciclo com a ideia de que, para bons classificadores, é essencial controlar a flexibilidade do modelo, usar validação cruzada, regularização e seleção adequada de features para evitar que o modelo brilhe só no treino, mas fracasse na produção.

Na aula de hoje vamos aprofundar um tema fundamental para a construção de modelos de machine learning eficazes: o **ajuste de hiperparâmetros**, chamado também de *tuning*. Entenderemos o que são hiperparâmetros, a importância de configurá-los corretamente, e exploraremos os métodos mais usados para encontrar a melhor combinação possível. Vamos cobrir os métodos não informados, como grid search e random search, até os métodos informados, com destaque para a otimização bayesiana.

## Hiperparâmetros

Primeiramente, é importante distinguir **hiperparâmetros** dos parâmetros estimados pelo modelo de aprendizado de máquina durante o treinamento. Enquanto parâmetros são aprendidos diretamente a partir dos dados durante o treinamento (por exemplo, pesos de uma regressão logística), os hiperparâmetros são valores que definimos antes do treinamento começar e que controlam o comportamento do algoritmo. Exemplos comuns de hiperparâmetros incluem a taxa de aprendizado (learning rate); n-gramas em modelos de classificação com texto; número de árvores em uma floresta aleatória; ou o número de épocas em treinamento de redes neurais.

Esses hiperparâmetros podem ter valores discretos ou contínuos, e suas escolhas impactam diretamente a performance do modelo — seja pela capacidade de generalização, velocidade de convergência, ou até mesmo pela estabilidade do processo. Por isso, o tuning de hiperparâmetros é uma etapa crucial para maximizar a qualidade do modelo.

**Tabela 1: Exemplos de hiperparâmetros e parâmetros**
| Tipo | Exemplo | Descrição |
| :-- | :-- | :-- |
| Hiperparâmetro | Taxa de aprendizado (learning rate) | Controla o tamanho dos passos nas atualizações dos pesos durante o treinamento |
| Hiperparâmetro | Número de camadas em uma rede neural | Define quantas camadas ocultas o modelo terá, influenciando sua complexidade |
| Hiperparâmetro | Número de estimadores (n_estimators) | Quantidade de árvores em modelos ensemble como Random Forest ou XGBoost |
| Parâmetro | Peso das conexões (weights) | Valores ajustados internamente pelo algoritmo para representar a relação entre as features e a saída |
| Parâmetro | Bias (termo livre) | Valor ajustado que ajuda o modelo a se adaptar melhor aos dados |
| Parâmetro | Coeficientes de regressão | Parâmetros aprendidos em modelos lineares para ponderar cada variável explicativa |



## Ajustando Hiperparâmetros

O *tuning*, ou ajuste, de hiperparâmetros é o processo pelo qual nós procuramos pelo melhor conjunto de hiperparâmetros de um modelo de aprendizado de máquina dentro de um conjunto de parâmetros candidatos. Com isso, conseguimos otimizar algumas métricas de interesse (Precision da classe minoritária, f1-score, etc.). O objetivo principal é o de conseguir a melhor métrica de validação possível. Essa abordagem é uma abordagem um pouco diferente do que vimos no curso. Durante as aulas, nosso foco foi em como gerar o melhor conjunto de dados de treinamento possível, estabelecendo regras de anotação, calculando métricas de concordância entre anotadores, evitando erros amostrais, etc. No entanto, essa abordagem é focada no modelo e em sua otimização. Mas sempre é importante lembrar que se seus dados tiverem problemas graves, nenhuma otimização de modelo irá importar. Sempre se lembre do lema "*Garbage in, garbage out*": Não importa qual seu modelo ou hiperparâmetros se os dados são ruins. 

```{figure} ../aula9/images/kuhnfig4.4.png
---
width: 100%
name: hypertuning
align: center
---
Esquema do processo de ajuste dos Hiperparâmetros. Fonte Kuhn and Kjell (p.66, {cite}`kuhn2018applied`.)
```

A {numref}`Figura {number} <hypertuning>` esquematiza o processo padrão de ajuste de hiperparâmetros: 1) É necessário definir um espaço de valores de hiperparâmetros candidatos, que será usado para encontrar o melhor conjunto. 2) Para cada um dos hiperparâmetros (e para cada iteração), o modelo será treinado e resultados serão armazenados, alterando os valores dos hiperparâmetros. 3) Com isso, definiremos o melhor conjunto de valores dos hiperparâmetros. 4) Com esses valores, o modelo é treinado novamente, sendo validado e testado para conferir os resultados. A tabela 2 abaixo mostra os principais hiperparâmetros dos modelos clássicos de aprendizado de máquina (pré *deep learning*).

**Tabela 2 - Hiperparâmetros dos modelos Clássicos**
| Modelo | Hiperparâmetro | Descrição |
| :-- | :-- | :-- |
| Regressão Logística | C | Inverso da força da regularização |
|  | penalty | Tipo de penalização (l1, l2, elasticnet, none) |
|  | solver | Algoritmo para otimização (liblinear, lbfgs, saga, etc.) |
|  | max_iter | Max de iterações permitidas para convergência |
| K-Nearest Neighbors | n_neighbors | Número de vizinhos para classificação |
|  | weights | Peso dos vizinhos (uniform, distance) |
|  | algorithm | Algoritmo para busca da vizinhança (auto, ball_tree, etc.) |
|  | leaf_size | Tamanho da folha da árvore de busca |
|  | p | Parâmetro da métrica de distância (1=Manhattan, 2=Euclidiana) |
| Naive Bayes (Gaussian) | var_smoothing | Suavização para estabilidade numérica |
| Naive Bayes (Multinomial) | alpha | Parâmetro de suavização laplace |
| SVM | C | Parâmetro de regularização |
|  | kernel | Tipo de kernel (linear, poly, rbf, sigmoid) |
|  | degree | Grau do polinômio (para kernel polinomial) |
|  | gamma | Parâmetro do kernel (scale, auto, float) |
|  | coef0 | Termo independente no kernel (poly e sigmoid) |
| Decision Tree | criterion | Função para medir qualidade da divisão (gini, entropy) |
|  | splitter | Estratégia para escolher divisão (best, random) |
|  | max_depth | Profundidade máxima da árvore |
|  | min_samples_split | Mínimo de amostras para dividir um nó |
|  | min_samples_leaf | Mínimo de amostras em um nó folha |
|  | max_features | Número máximo de features para divisão |
| Random Forest | n_estimators | Número de árvores na floresta |
|  | criterion | Função para medir divisão (gini, entropy) |
|  | max_features | Máximo de features para divisão |
|  | max_depth | Profundidade máxima das árvores |
|  | min_samples_split | Mínimo de amostras para dividir nó |
|  | min_samples_leaf | Mínimo de amostras para folha |
|  | bootstrap | Uso ou não de amostragem bootstrap |
|  | oob_score | Uso do método out-of-bag para estimativa de erro |




### Espaço de Hiperparâmetros

Ao fazermos tuning, trabalhamos com o chamado espaço de hiperparâmetros, que consiste em todas as combinações possíveis dos valores de cada hiperparâmetro. Esse espaço pode ser bastante grande, multidimensional e até conter variáveis condicionais (por exemplo, valores de um hiperparâmetro dependem do valor de outro).

Além disso, para métodos mais avançados, é importante definir distribuições de probabilidade sobre os hiperparâmetros, que indicam a chance de cada valor ser testado. Essas distribuições podem ser uniformes, log-uniformes, normais, entre outras, e ajudam a guiar a busca por melhores combinações.

### Métodos Não Informados (Exaustivos)

#### Grid Search

Grid search é o método mais simples e direto: ele testa **exhaustivamente** todas as combinações possíveis definidas no espaço dos hiperparâmetros. Imagina que temos três hiperparâmetros, cada um com 5 possíveis valores; o grid search testaria todas as $5 \times 5 \times 5 = 125$ combinações.

Apesar de ser fácil de entender e implementar, grid search é computacionalmente caro, especialmente quando o espaço cresce — o número de avaliações aumenta exponencialmente com a quantidade e granularidade dos hiperparâmetros. Portanto, é prático apenas para espaços pequenos ou com poucos hiperparâmetros relevantes.

#### Random Search

Random search é uma alternativa mais eficiente ao grid search. Em vez de testar todas as combinações, ele sorteia aleatoriamente valores dentro do espaço definido para um número fixo de iterações. Isso permite explorar um espaço de busca maior com menos avaliações, e estudos mostram que random search pode ser mais eficaz do que grid search, pois nem todos os hiperparâmetros impactam igualmente a performance do modelo.

No random search, é importante definir quantas iterações serão feitas e as distribuições dos hiperparâmetros para garantir uma cobertura razoável do espaço. Esse método é simples e rápido, e ideal quando não se tem muita informação prévia sobre bons valores dos hiperparâmetros.

### Método Informado

#### Otimização Bayesiana

Diferente dos métodos exaustivos, a otimização bayesiana é um método **informado**, que aprende com as avaliações anteriores para dirigir a busca de forma mais eficiente. Ela utiliza um modelo probabilístico (chamado de modelo substituto ou surrogate) para estimar a relação entre conjuntos de hiperparâmetros e o desempenho do modelo, e uma função de aquisição para decidir quais hiperparâmetros testar a seguir.

Esse processo sequencial torna a otimização bayesiana mais **data eficiente**, exigindo menos avaliações para encontrar bons hiperparâmetros, especialmente em espaços grandes ou onde o custo de avaliar o modelo é alto (ex: modelos complexos ou conjuntos grandes de dados).

Existem variações da otimização bayesiana baseadas em diferentes modelos substitutos, como o processo gaussiano (GP), o modelo baseado em floresta aleatória (SMAC) e o estimador Parzen estruturado em árvore (TPE). Cada variante tem suas peculiaridades, vantagens e limitações, mas todas seguem a ideia principal de aprender com os resultados anteriores para guiar a busca.

### Conclusão

O tuning de hiperparâmetros é uma etapa essencial para garantir que modelos de machine learning atinjam seu potencial máximo. Enquanto métodos simples como grid search e random search são úteis e fáceis de aplicar, eles podem ser custosos e ineficientes para espaços grandes. A otimização bayesiana, por outro lado, oferece uma abordagem inteligente e eficiente, aprendendo com as iterações anteriores para focalizar a busca.

Agora que entendemos esses conceitos e métodos, estamos mais preparados para aplicar tuning eficazmente nos nossos projetos, maximizando a performance e a robustez dos modelos.




## Notas

[^1]: 