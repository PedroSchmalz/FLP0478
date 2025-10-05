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




## Espaço de Hiperparâmetros

O espaço de hiperparâmetros é definido com o oconjunto universal de possíveis combinações de valores de hiperparâmetros. Em outras palavras, é o espaço contendo todos os valores possíveis de hiperparâmetros que serão usados como espaço de busca durante da fase do ajuste. Esse espaço pode ser bastante grande, multidimensional e até conter variáveis condicionais (por exemplo, valores de um hiperparâmetro dependem do valor de outro). Além disso, para métodos mais avançados, é importante definir distribuições de probabilidade sobre os hiperparâmetros, que indicam a chance de cada valor ser testado. Essas distribuições podem ser uniformes, log-uniformes, normais, entre outras, e ajudam a guiar a busca por melhores combinações.

Os métodos de ajuste de hiperparâmetros podem ser divididos em duas categorias principais: não informados e informados. Os métodos não informados (ou de busca exaustiva), como *grid search* e *random search*, realizam a busca pelos melhores hiperparâmetros de forma independente em cada iteração, sem aprender com os resultados das avaliações anteriores. Eles simplesmente testam combinações predefinidas ou aleatórias do espaço de hiperparâmetros sem usar essas informações para guiar iterações futuras. Em contraste, os métodos informados, como a otimização bayesiana, utilizam um modelo probabilístico que aprende com as avaliações anteriores para direcionar a busca de maneira mais eficiente. Esses métodos constroem um modelo substituto (surrogate model) que estima a relação entre hiperparâmetros e performance do modelo, usando essa informação para decidir quais combinações testar a seguir, focando em regiões promissoras do espaço de busca. Isso torna os métodos informados mais eficientes em termos de número de avaliações necessárias, especialmente quando o custo computacional de cada avaliação é alto ou o espaço de hiperparâmetros é muito grande.

## Métodos Não Informados (Exaustivos)

O primeiro grupo de métodos de ajuste de hiperparâmetros é conhecido como os métodos de **busca exaustiva**, ou não informados. São os mais usados e diretos. Métodos deste grupo fazem uma busca exaustiva dentro do espaço de hiperparâmetros, testando várias combinações e salvando a que tiver melhor resultado, sem aprender com as iterações. Três métodos são mais conhecidos: a busca manual, a busca em "grade" (*grid search*), e a busca aleatória (*random search*). 

### Busca Manual

A busca manual, como diz o nome, é feita manualmente com base no "instinto" do pesquisador. Um parâmetro é alterado manualmente (por exemplo n-gram), e olhamos os resultados de validação e teste. Se melhorar, mantemos. Se piorar, alteramos. Você ajusta os hiperparâmetros até ficar satisfeito com o resultado final. É um método cansativo, manual, e propenso ao erro. Para evitar problemas deste tipo, métodos de ajuste automatizados foram criados.

### Busca em "Grade" (*Grid Search*)

Grid search é o método mais simples e intuitivo de *tuning* de hiperparâmetros, funcionando essencialmente como uma busca exaustiva em um espaço discreto de valores predefinidos. O processo consiste em criar uma "grade" com todas as combinações possíveis dos hiperparâmetros especificados e avaliar o modelo para cada uma dessas combinações usando validação cruzada. Por exemplo, imagine que estamos ajustando uma regressão logística multinomial para classificar tweets sobre vacinas, e queremos otimizar três hiperparâmetros: o tipo de n-gram (unigrama, bigrama), o solver (liblinear, lbfgs, saga) e a força de regularização C (0.1, 1.0, 10.0). O grid search testaria sistematicamente todas as $2 × 3 × 3 = 18$ combinações possíveis, treinando e avaliando o modelo para cada uma delas com 10 folds de validação cruzada, e retornando aquela que obteve a melhor métrica de desempenho (por exemplo, F1-score médio). Na prática, isso significa que, se cada avaliação de validação cruzada leva 2 minutos, o processo completo levaria 36 minutos para encontrar a melhor configuração. Embora esse método garanta que encontremos a melhor combinação dentro do espaço especificado, ele se torna rapidamente impraticável quando temos muitos hiperparâmetros ou muitos valores possíveis para cada um — adicionar apenas mais um hiperparâmetro com 3 valores possíveis triplicaria o tempo total de execução, ilustrando o crescimento exponencial do custo computacional que caracteriza este método. Além disso, corremos o risco do melhor valor de  hiperparâmetro **não estar** dentro do nosso espaço de hiperparâmetros.


```{figure} ../aula9/images/owenfig3.2.png
---
width: 100%
name: gridsearch
align: center
---
Espaço de hiperparâmetros em uma busca em grade (*Grid Search*). Fonte Owen (p.23, {cite}`owen2022hyperparameter`.)
```

Por exemplo, a figura {numref}`Figura {number} <gridsearch>` mostra o número de combinações possíveis com dois hiperparâmetros. O hiperparâmetro 1 está definido com 4 valores possíveis, e o hiperparâmetro 2 está definido com 5 valores. Somente com esses dois hiperparâmetros, temos 20 combinações possíveis. Se estamos usando a validação cruzada com 10 folds, treinaremos o modelo $20*10 = 200$ vezes. Para um modelo simples, como a regressão multinomial, isso provavelmente não demandará muito tempo. No entanto, com modelos mais pesados, ou mais combinações de valores de parâmetros, isso rapidamente se torna proibitivo.

### Random Search

A busca aleatória é uma alternativa mais eficiente à busca em grade que, em vez de testar todas as combinações possíveis de hiperparâmetros de forma exaustiva, amostra aleatoriamente um número fixo de combinações dentro do espaço de busca definido. O funcionamento é simples: primeiro, definimos distribuições de probabilidade ou intervalos para cada hiperparâmetro — por exemplo, para uma regressão logística multinomial, poderíamos especificar que o parâmetro de regularização C seja sorteado de uma distribuição log-uniforme entre 0.001 e 100, o solver seja escolhido aleatoriamente entre ['liblinear', 'lbfgs', 'saga'], e o tipo de n-gram seja sorteado entre (1,1), (1,2), (1,3). Em seguida, o algoritmo sorteia aleatoriamente combinações desses valores e avalia cada uma usando validação cruzada, repetindo esse processo por um número predeterminado de iterações (por exemplo, 50 iterações). 

Voltando ao exemplo anterior dos tweets sobre vacinas, se a busca em grade testaria todas as 18 combinações em 36 minutos, a busca aleatória poderia testar 50 combinações aleatórias em aproximadamente 100 minutos, mas com uma vantagem crucial: geralmente a busca aleatória encontra configurações tão boas quanto — ou até melhores que — a busca em grade, especialmente quando alguns hiperparâmetros têm pouco impacto na performance do modelo. Isso acontece porque a busca aleatória explora melhor o espaço de hiperparâmetros importantes, já que não desperdiça avaliações testando sistematicamente todas as combinações de hiperparâmetros irrelevantes. Na prática, a busca aleatória é particularmente eficaz quando o espaço de busca é grande ou quando não temos certeza sobre quais hiperparâmetros são mais importantes, tornando-se a escolha preferida em muitos cenários.


```{figure} ../aula9/images/owenfig3.4.png
---
width: 100%
name: gridsearch
align: center
---
Ilustração do espaço de hiperparâmetros em uma busca aleatória (*Random Search*). Fonte Owen (p.25, {cite}`owen2022hyperparameter`.)
```


### Comparação métodos não informados

| Método | Prós | Contras |
|--------|------|---------|
| **Grid Search** | Simples de implementar e entender | Computacionalmente caro, especialmente com muitos hiperparâmetros |
| | Garante que todas as combinações sejam testadas | Ineficiente em espaços de busca grandes |
| | Ideal para espaços pequenos de hiperparâmetros | Tempo de execução cresce exponencialmente com o número de hiperparâmetros |
| | Permite análise detalhada do impacto de cada hiperparâmetro | Pode ser impraticável em cenários com grandes volumes de dados |
| **Random Search** | Menos custoso computacionalmente que Grid Search | Não garante que a melhor combinação seja encontrada |
| | Mais eficiente em espaços de busca grandes | Pode exigir mais iterações para encontrar uma solução satisfatória |
| | Pode explorar melhor o espaço de hiperparâmetros | Resultados podem variar entre execuções devido à aleatoriedade |
| | Frequentemente encontra soluções comparáveis ao Grid Search com menos avaliações | Menos determinístico que Grid Search |



## Métodos Informados


Os métodos informados de ajuste de hiperparâmetros representam uma abordagem fundamentalmente diferente dos métodos exaustivos (Não informados), pois utilizam informação acumulada das iterações anteriores para guiar a busca de forma inteligente e eficiente. O principal representante desta categoria é a otimização bayesiana, que se baseia no teorema de Bayes para construir um modelo probabilístico (chamado de modelo substituto ou surrogate model) que estima a relação entre configurações de hiperparâmetros e a performance do modelo. O funcionamento da otimização bayesiana segue um ciclo iterativo: primeiro, avalia-se algumas combinações iniciais de hiperparâmetros (geralmente escolhidas aleatoriamente); em seguida, constrói-se um modelo probabilístico (frequentemente usando Processos Gaussianos) que prevê quão bem o modelo funcionará para combinações não testadas; depois, usa-se uma função de aquisição para decidir qual combinação testar a seguir, equilibrando exploração (testar regiões desconhecidas) e explotação (focar em regiões promissoras); finalmente, atualiza-se o modelo substituto com os novos resultados. Por exemplo, retornando ao caso da classificação de tweets sobre vacinas, se a busca aleatória precisasse testar 50 combinações aleatórias em 100 minutos, a otimização bayesiana poderia encontrar uma configuração igualmente boa ou melhor testando apenas 20 a 30 combinações no mesmo período de tempo, pois aprende rapidamente quais regiões do espaço de hiperparâmetros são mais promissoras e concentra seus esforços ali. Isso torna a otimização bayesiana particularmente valiosa quando cada avaliação é computacionalmente cara — por exemplo, ao treinar modelos complexos em grandes conjuntos de dados, ou quando o espaço de hiperparâmetros é muito grande e a busca exaustiva se torna proibitiva. Na prática, bibliotecas como Optuna, Hyperopt e scikit-optimize implementam variantes da otimização bayesiana, tornando este método sofisticado acessível para pesquisadores e profissionais de machine learning que buscam maximizar a performance de seus modelos com eficiência computacional.

### Otimização Bayesiana

A otimização bayesiana é uma técnica sofisticada de otimização global que utiliza princípios da estatística bayesiana para encontrar o máximo ou mínimo de uma função desconhecida, sendo particularmente eficaz quando a avaliação da função é computacionalmente cara ou demorada. Diferentemente dos métodos exaustivos, a otimização bayesiana constrói um modelo probabilístico (tipicamente usando Processos Gaussianos) que representa nossa crença sobre a função objetivo desconhecida, atualizando essa crença a cada nova observação através do teorema de Bayes. O processo funciona iterativamente: primeiro avalia algumas combinações iniciais de hiperparâmetros, então constrói o modelo probabilístico que prevê tanto o valor esperado quanto a incerteza para combinações não testadas, utiliza uma função de aquisição para decidir qual combinação testar a seguir, avalia essa nova combinação, e finalmente atualiza o modelo com as novas informações. Essa abordagem sequencial permite que o algoritmo "aprenda" quais regiões do espaço de hiperparâmetros são mais promissoras, concentrando os esforços computacionais onde há maior potencial de melhoria.

### Função Objetivo

A função objetivo é a métrica que queremos otimizar durante o processo de tuning de hiperparâmetros, representando a performance do modelo que estamos tentando maximizar ou minimizar. No contexto de machine learning, a função objetivo geralmente é uma métrica de avaliação calculada através de validação cruzada, como F1-score, acurácia, precisão, recall, ou AUC-ROC. Por exemplo, ao ajustar hiperparâmetros de uma regressão logística multinomial para classificar tweets sobre vacinas, nossa função objetivo poderia ser o F1-score médio obtido através de validação cruzada k-fold. A função objetivo é tipicamente uma "caixa-preta" — não conhecemos sua forma matemática exata e só podemos avaliá-la através de experimentos computacionais caros, que envolvem treinar e validar o modelo com diferentes combinações de hiperparâmetros. É justamente essa característica custosa da função objetivo que torna a otimização bayesiana tão valiosa, pois ela minimiza o número de avaliações necessárias.

### Função de Aquisição

A função de aquisição é o componente que diferencia a otimização bayesiana de outros métodos, sendo responsável por decidir qual combinação de hiperparâmetros testar a seguir, equilibrando cuidadosamente exploração, *exploration*, (testar regiões desconhecidas) e *exploitation* (focar em regiões promissoras). Existem várias estratégias de aquisição, cada uma com características específicas: *Expected Improvement* (EI) calcula o ganho esperado sobre o melhor resultado atual, sendo conservadora e focando em melhorias incrementais; *Upper Confidence Bound *(UCB) seleciona pontos com alta média predita ou alta incerteza, sendo mais agressiva na exploração; *Probability of Improvement* (PI) escolhe pontos com maior probabilidade de superar o melhor resultado atual, sendo a mais simples mas potencialmente muito conservadora. Por exemplo, em um problema onde o melhor F1-score atual é 0.85, a função EI priorizaria combinações que têm boa chance de superar esse valor, enquanto UCB também consideraria regiões com alta incerteza mesmo que a média predita seja menor, garantindo uma exploração mais ampla do espaço de hiperparâmetros.



```{figure} ../aula9/images/owenfig4.1.png
---
width: 100%
name: BOsearch
align: center
---
Ilustração do Método de Otimização Bayesiana. Fonte Owen (p.30, {cite}`owen2022hyperparameter`.)
```


A {numref}`Figura {number} <gridsearch>` ilustra de forma didática o princípio fundamental da otimização bayesiana através do conceito de modelo de regressão probabilístico, também conhecido como modelo substituto ou *surrogate model*. No gráfico, o eixo horizontal representa o espaço de um hiperparâmetro (x), enquanto o eixo vertical representa a função objetivo f(x) — por exemplo, o F1-score do modelo. Os pontos laranjas representam os pares de valores conhecidos, ou seja, combinações de hiperparâmetros que já foram avaliadas e cujas performances são conhecidas. A curva tracejada representa a "curva desconhecida", que é justamente a função objetivo real que queremos otimizar, mas que não conhecemos completamente. O modelo substituto funciona como uma aproximação probabilística dessa curva desconhecida, construída a partir dos pontos já observados, permitindo ao algoritmo fazer previsões sobre quão bem o modelo performará para valores de hiperparâmetros ainda não testados. A cada nova avaliação, um novo ponto laranja é adicionado ao conjunto de observações, e o modelo substituto é atualizado, refinando sua estimativa da curva desconhecida e melhorando progressivamente sua capacidade de identificar regiões promissoras do espaço de hiperparâmetros onde o próximo teste deve ser realizado. Essa abordagem iterativa e informada é o que torna a otimização bayesiana mais eficiente que métodos de busca exaustiva, pois ela "aprende" com cada iteração para focar os esforços computacionais nas regiões mais promissoras do espaço de busca.


Dentro da família de métodos de otimização bayesiana, existem diferentes implementações que se distinguem principalmente pelo tipo de modelo substituto utilizado para aproximar a função objetivo. A abordagem clássica, conhecida como Bayesian Optimization with Gaussian Processes (BOGP), utiliza Processos Gaussianos como modelo substituto, oferecendo estimativas probabilísticas suaves e contínuas da função objetivo, sendo particularmente eficaz em espaços de hiperparâmetros contínuos de baixa dimensionalidade. O Sequential Model-based Algorithm Configuration (SMAC) substitui Processos Gaussianos por Random Forests como modelo substituto, o que permite trabalhar melhor com hiperparâmetros categóricos e condicionais, além de escalar melhor para espaços de maior dimensionalidade. Já o Tree-structured Parzen Estimator (TPE), implementado em bibliotecas populares como Hyperopt e Optuna, adota uma abordagem diferente ao modelar diretamente as distribuições de hiperparâmetros que levam a bons e maus resultados, sendo computacionalmente eficiente e robusto em diversos cenários práticos. Cada uma dessas variações tem suas vantagens específicas, mas todas compartilham o princípio fundamental de aprender com avaliações anteriores para guiar a busca de forma inteligente, tornando-se ferramentas valiosas quando o custo computacional de cada avaliação é alto e queremos maximizar a eficiência do processo de tuning.


Embora a otimização bayesiana seja o método informado mais amplamente utilizado e estudado, existem outras abordagens sofisticadas que também aprendem com iterações anteriores para guiar a busca por hiperparâmetros. Os algoritmos evolutivos, inspirados na evolução biológica, mantêm uma "população" de configurações de hiperparâmetros que evoluem através de operações como mutação e cruzamento, selecionando as melhores configurações para gerar novas gerações. O Population-Based Training (PBT) combina busca aleatória com exploração evolutiva, treinando múltiplos modelos em paralelo e periodicamente copiando pesos de modelos bem-sucedidos para substituir os menos performantes. O Simulated Annealing simula o processo de resfriamento de metais, começando com uma "temperatura" alta que permite explorações mais amplas e gradualmente diminuindo para focar em refinamentos locais. Métodos de multi-fidelidade, como Hyperband e BOHB (Bayesian Optimization HyperBand), aceleram a busca treinando modelos com recursos reduzidos (menos épocas, menos dados) para eliminar rapidamente configurações ruins antes de investir recursos completos nas promissoras. Para estudantes interessados em aprofundar seus conhecimentos, bibliotecas como Optuna, Ray Tune, Hyperopt e NNI (Neural Network Intelligence) implementam vários desses métodos, oferecendo uma excelente oportunidade para experimentação prática com diferentes abordagens de tuning informado em projetos futuros.

### Conclusão

Esta aula consolida um dos pilares fundamentais do machine learning: a interconexão entre validação cruzada, controle de sobreajuste e otimização de hiperparâmetros. A validação cruzada k-fold, que exploramos na aula anterior, não é apenas uma técnica de avaliação isolada, mas sim a base que torna possível todo o processo de tuning de hiperparâmetros de forma confiável e robusta. Sem validação cruzada, estaríamos navegando às cegas no espaço de hiperparâmetros, correndo o risco de otimizar para um conjunto específico de dados de validação e criar modelos que falham miseravelmente em produção. A validação cruzada fornece estimativas estáveis e menos enviesadas da performance do modelo, permitindo que métodos como grid search, random search e otimização bayesiana tomem decisões informadas sobre quais configurações realmente generalizam bem para dados não vistos.

O processo de tuning de hiperparâmetros que discutimos hoje — desde os métodos exaustivos mais simples até a sofisticada otimização bayesiana — depende crucialmente da validação cruzada para avaliar cada combinação de hiperparâmetros de forma consistente e confiável. Quando executamos grid search com 18 combinações ou random search com 50 iterações, cada uma dessas avaliações usa validação cruzada k-fold internamente, garantindo que não estamos simplesmente encontrando hiperparâmetros que funcionam bem por acaso em uma divisão específica dos dados. Da mesma forma, a otimização bayesiana constrói seu modelo substituto com base em avaliações de validação cruzada, permitindo que ela faça previsões precisas sobre regiões não exploradas do espaço de hiperparâmetros.

Mais profundamente, esta aula revela como diferentes aspectos do machine learning se complementam para resolver o problema central do sobreajuste. A validação cruzada detecta quando um modelo está decorando ao invés de aprender; o tuning de hiperparâmetros ajusta a complexidade do modelo para encontrar o ponto ideal entre underfitting e overfitting; e os diferentes métodos de busca nos permitem explorar eficientemente o espaço de configurações possíveis. Juntos, esses elementos formam um sistema integrado que maximiza as chances de desenvolvermos modelos que não apenas performam bem durante o desenvolvimento, mas também mantêm essa performance quando confrontados com dados reais em outros bancos. 



