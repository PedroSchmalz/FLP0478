# Métodos *Ensemble*

Na última aula vimos três componentes principais de técnicas de aprendizado de máquina: começamos explorando as Árvores de Decisão, um método fundamental baseado em estratificação do espaço de preditores que funciona através de divisão binária recursiva, onde o algoritmo guloso testa todos os pontos de corte possíveis para minimizar o RSS (em regressão) ou a impureza (medida pelo índice de Gini ou entropia) em classificação, criando estruturas hierárquicas fáceis de interpretar onde cada observação percorre nós decisórios até chegar em regiões/folhas que fornecem a predição final. Vimos também que o crescimento descontrolado das árvores leva a *overfitting*, por isso introduzimos a poda de custo-complexidade, um hiperparâmetro α que penaliza árvores muito grandes, conectando-se diretamente com os conceitos de ajuste de hiperparâmetros da aula anterior e demonstrando o trade-off entre viés e variância. 

Em seguida, exploramos as Máquinas de Vetores de Suporte (SVM), que surgem como generalização do classificador de margem máxima, começando com o conceito de hiperplano em espaços p-dimensionais para dividir observações em classes, depois relaxando essa restrição rígida através dos classificadores de vetores de suporte com margem "suave" controlada pelo hiperparâmetro C (orçamento de violação), e finalmente chegando ao SVM completo que usa o kernel trick para lidar elegantemente com relações não-lineares, transformando implicitamente os dados para espaços de dimensão superior onde se tornam separáveis sem criar manualmente termos polinomiais. 

Por fim, estudamos o TF-IDF (Term Frequency-Inverse Document Frequency), uma medida estatística que captura a importância de uma palavra em um documento específico em relação a uma coleção, multiplicando dois componentes: a frequência do termo (TF) com transformação logarítmica para desacelerar o crescimento, e a frequência inversa de documento (IDF) que penaliza palavras comuns no corpus mas recompensa palavras raras, permitindo que trabalhem bem com n-gramas e sendo especialmente compatível com SVMs em problemas de classificação textual em espaços de alta dimensionalidade.

Na aula de hoje vamos aprofundar nossa compreensão dos métodos de machine learning explorando duas técnicas fundamentais de aprendizado em conjunto: o Bagging (Bootstrap Aggregating) e o Boosting. Veremos também como esses métodos ensemble diferem radicalmente em suas estratégias: enquanto o Bagging treina múltiplos modelos de forma independente e paralela em subconjuntos aleatórios dos dados através de amostragem com reposição, reduzindo principalmente a variância e funcionando particularmente bem com modelos complexos como árvores profundas, o Boosting adota uma abordagem sequencial onde cada novo modelo é treinado com o objetivo específico de corrigir os erros cometidos pelos modelos anteriores, reduzindo o viés através de uma combinação ponderada de modelos fracos. O Bagging combina as previsões dos modelos via média ou votação, tornando o resultado mais estável e robusto, enquanto o Boosting atribui pesos maiores aos exemplos que foram classificados incorretamente, forçando os modelos posteriores a focar nesses casos difíceis. Entender essas duas abordagens distintas é essencial para escolher a estratégia correta para seus dados e objetivos específicos, pois cada uma oferece diferentes trade-offs entre redução de erro, complexidade computacional e risco de overfitting.

## Bagging


```{video} https://www.youtube.com/embed/Xz0x-8-cgaQ?si=kh22plivN21X6OCA
```

Um método *ensemble* é aquele que combina modelos como "tijolos" de forma a obter um único modelo com melhores resultados. Hoje iremos discutir dois grupos de métodos principais: *Bagging* e *Boosting*. O ***Bagging***, ou *Bootstrap Aggregation*, é um procedimento geral utilizado para reduzir a variância de um modelo de aprendizado estatístico. O *Bagging* se baseia no método de *Bootstrap*, e procura reduzir a variância do modelo pegando diversos conjuntos de treinamento da população, construindo um modelo em cada conjunto separadamente, e pegando a média das previsões de todos os modelos em todos os conjuntos. Em outras palavras



```{figure} ../aula10/images/islfig8.3.1.png
---
width: 100%
name: dtreg
align: center
---
Ilustração da Árvore de Decisão no contexto de Regressão. Fonte: James et al. ({cite}`james2023introduction`., p. 335)
```

A {numref}`Figura {number} <dtreg>` mostra o processo decisório em um modelo de árvore de decisão. Na figura, temos duas variáveis preditoras, $X_1$ e $X_2$, e a árvore vai se dividindo de acordo com os valores das duas. No começo da árvore, também conhecido como **Nó raiz** ou só **raiz**, a primeira decisão é com base no corte $t_1$: Valores menores que $t_1$ em $x_1$ jogam as observações para o lado esquerdo da árvore, e valores maiores vão para o lado direito. Do lado esquerdo da figura, a segunda decisão vem com base em $X_2$, com valores menores que o ponto de corte $t_2$ caindo para a primeira região $R_1$, e valores maiores que $t_2$ caindo na segunda região $R_2$. Cada observação vai passar por esses nós decisórios, chegando nos nós terminais que vão dar a previsão final ($R_1$, $R_2$, etc.). No caso das árvores de regressão, o valor previsto será a média das observações dentro dessa região.





## Conclusão


Nesta aula exploramos as árvores de decisão, um dos métodos fundamentais de machine learning que serve de base para algoritmos mais sofisticados como Random Forests e Gradient Boosting. Aprendemos que tanto árvores de regressão quanto de classificação compartilham a mesma estratégia central: particionar recursivamente o espaço de preditores em regiões distintas através do algoritmo guloso e top-down de divisão binária recursiva, onde cada divisão busca localmente a melhor separação dos dados sem considerar o impacto global. Vimos que as árvores de regressão minimizam o RSS (Residual Sum of Squares) para encontrar as melhores divisões, enquanto as árvores de classificação utilizam medidas de impureza como o índice Gini ou entropia para avaliar a qualidade das separações, buscando criar nós filhos mais homogêneos e puros. Um conceito crucial que abordamos foi a poda de custo-complexidade (Cost-Complexity Pruning), que introduz o hiperparâmetro α para balancear erro de predição e complexidade do modelo, conectando-se diretamente com os conceitos de ajuste de hiperparâmetros da aula anterior e demonstrando como controlar o trade-off entre viés e variância para evitar overfitting. A principal vantagem das árvores de decisão é sua excepcional interpretabilidade: a estrutura hierárquica de regras "se-então" permite que profissionais de diversas áreas compreendam facilmente como o modelo toma decisões, tornando-as ideais para contextos onde explicabilidade é crucial, como diagnósticos médicos ou decisões de crédito. No entanto, como mencionado, árvores individuais geralmente não são tão competitivas em termos de performance pura quando comparadas a métodos mais modernos, tendendo a ter alta variância e sendo sensíveis a pequenas mudanças nos dados de treinamento. Na próxima seção, exploraremos o Support Vector Machine (SVM), um algoritmo com abordagem completamente diferente que busca maximizar margens de separação entre classes, e introduziremos o TF-IDF, uma técnica de ponderação de texto que será essencial para aplicar SVMs em problemas de classificação textual.


## Notas

[^1]: Outros métodos baseados em árvores incluem: Random Forests — ensembles de árvores construídas por amostragem bootstrap que reduzem a variância; Bagging (Bootstrap Aggregating) — agregação de várias árvores independentes; Extra-Trees (Extremely Randomized Trees) — similar a Random Forest com divisão mais aleatória; Boosting — métodos sequenciais que corrigem erros (ex.: AdaBoost); Gradient Boosting Machines (GBM) — otimização por gradiente de árvores fracas; implementações populares e otimizadas: XGBoost, LightGBM e CatBoost; Isolation Forest — uso de árvores para detecção de anomalias; e abordagens mais especializadas como Conditional Inference Trees e Bayesian Additive Regression Trees (BART). Cada família tem trade-offs distintos entre viés, variância, interpretabilidade e velocidade.

