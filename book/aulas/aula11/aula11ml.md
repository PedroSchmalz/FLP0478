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

### Métricas de Importância de Preditores/Variável

*Bagging* pode ser útil para aumentar o poder de previsão do modelo, mas perdemos a interpretabilidade, especialmente em um *bagging* que usa *Decision Trees*. No entanto, podemos recuperar a importância de cada preditor/variável independente usando o indíce de Gini (ou outro parâmetro de impureza na classificação), ou métricas de erro nas regressões. 


```{figure} ../aula11/images/islfig8.9.png
---
width: 100%
name: variableimp
align: center
---
Importância de preditores no contexto de Regressão. Fonte: James et al. ({cite}`james2023introduction`., p. 346))
```

A figura  {numref}`Figura {number} <variableimp>` ilustra algumas variáveis que teriam maior peso no contexto do banco de dados *Heart*. Vemos que a variável de nome Thal é a mais importante, seguida de Ca, *ChestPain* e assim por diante. Com isso, podemos entender quais preditores foram mais importantes para o modelo na hora de fazer as previsões. Para problemas de regressão, calculamos a redução total do RSS (Residual Sum of Squares) atribuível a cada preditor, somando as reduções de erro obtidas sempre que aquele preditor é usado para divisão em cada uma das B árvores do ensemble, e depois calculando a média dessa contribuição total. De forma análoga, em problemas de classificação, utilizamos o índice de Gini como medida, quantificando a redução total da impureza provocada por cada preditor em todas as árvores agregadas. Quanto maior esse valor acumulado e normalizado, mais importante é o preditor para as previsões finais do modelo — uma abordagem que, apesar de não fornecer as regras interpretáveis de uma única árvore, oferece um diagnóstico agregado robusto sobre quais variáveis realmente importam. Com isso, recuperamos um pouco da interpretabilidade dos modelos, mas aumentando também sua acurácia.


### Random Forests



```{video} https://www.youtube.com/embed/J4Wdy0Wc_xQ?si=kIoeuYOBYPDPBCE2
```

Um problema presente no *Bagging* com Árvores de Decisão é que ele gera árvores muito correlacionadas entre si. O *Random Forest* é um método de *Bagging* que procura aprimorar isso, reduzindo a correlação entre as árvores. Ele funciona da seguinte forma: Na hora de fazer cada divisão (*split*) dentro das árvores, uma amostra aleatória de $m$ preditores é escolhida dentro do espaço de *p* preditores. Esses $m$ preditores serão então reamostrados a cada *split*. Dessa forma, o algoritmo não tem acesso a todos os preditores em todos os momentos, e isso gera maior variância em suas decisões, gerando árvores mais diferentes e melhores resultados. Com isso, o *Random Forest* garante uma média de preditores $\hat{f}^\text{*b}(x)$ menos correlacionados entre si.

## *Boosting*

Assim como o *Bagging*, o ***Boosting*** é um método geral que pode ser utilizado com diferentes modelos de aprendizado de máquina. No caso do *Decision Trees*, o *Boosting* funciona de maneira parecida com o *Bagging*, mas as árvores são criadas sequencialmente. Isto é, cada nova árvore usa informações das árvores anteriores para aprimorar as previsões. O método consiste em combinar uma grande quantidade de árvores de decisão $\hat{f}^1(x)$, $\hat{f}^2(x)$, ... $\hat{f}^B(x)$, mas ao invés de ajustar várias árvores de uma vez e pegar a média das previsões, o *Boosting* aprende gradualmente. Ele irá usar o modelo para classificar os dados, e um modelo subsequente irá usar o erro do modelo anterior como guia para ajustar os parâmetros e previsões. Ou seja, o modelo subsequente irá ser ajustado com base nos resíduos do modelo anterior. Ao encaixar pequenas árvores nos resíduos, o *Boosting* melhora gradualmente a função $f$ em áreas em que ela vai mal. Um hiperparâmetro importante aqui é o $\lambda$, ou parâmetro de encolhimento, que vai determinar o quão rápido esse processo vai ser.


```{figure} ../aula11/images/datascientesboos.png
---
width: 100%
name: boosting
align: center
---
Processo de *Boosting* de modelos. Fonte: [DataScientest](https://datascientest.com/en/bagging-vs-boosting).)
```

A força do Boosting reside em sua capacidade de corrigir sistematicamente os erros cometidos pelos modelos anteriores através de um processo iterativo e sequencial, transformando modelos individualmente fracos em um preditor ensemble poderoso. Diferentemente do Bagging, que reduz principalmente a variância pela agregação paralela de modelos complexos, o Boosting ataca o problema do viés, focando em observações que foram mal classificadas ou com resíduos elevados em iterações anteriores — essas instâncias recebem pesos maiores no treinamento subsequente, forçando o novo modelo a concentrar seus esforços em corrigir exatamente onde o modelo anterior falhou. O hiperparâmetro λ (lambda), conhecido como parâmetro de encolhimento ou shrinkage parameter, controla a taxa de aprendizado desse processo iterativo: valores menores de λ significam que cada árvore contribui com uma fração menor das previsões, resultando em um aprendizado mais gradual e controlado que geralmente produz melhores generalizações, enquanto valores maiores aceleram o processo mas aumentam o risco de overfitting. Essa abordagem sequencial torna o Boosting computacionalmente mais intensivo que o Bagging, pois os modelos não podem ser treinados em paralelo, mas frequentemente compensa com acurácia superior, especialmente em datasets com características complexas ou quando o objetivo é maximizar a performance preditiva ao custo de maior complexidade computacional.

### AdaBoost



```{video} https://www.youtube.com/embed/LsK-xG1cLYA?si=HmHvS3bwniQynCGJ
```


O ***AdaBoost*** (*Adaptive Boosting*) é um algoritmo de boosting pioneiro, desenvolvido por Yoav Freund e Robert Schapire, que implementa a filosofia central do boosting de forma elegante e adaptativa: ele começa atribuindo pesos iguais a todas as observações de treinamento e treina sequencialmente um modelo fraco (tipicamente uma árvore de decisão rasa chamada decision stump, por que só tem uma decisão), depois identifica as instâncias que foram classificadas incorretamente e aumenta seus pesos, forçando o próximo modelo a concentrar seus esforços exatamente nessas observações problemáticas. Esse processo iterativo continua até que um número máximo de modelos seja atingido ou o desempenho se estabilize, com cada modelo subsequente se beneficiando da retroalimentação do anterior através dessa redistribuição adaptativa de pesos — daí o nome "adaptativo". A previsão final é obtida através de uma combinação ponderada dos modelos, onde modelos mais precisos recebem pesos maiores na votação final. O AdaBoost é particularmente eficaz em problemas de classificação binária e pode ser aplicado com qualquer algoritmo de aprendizado como base, não sendo especialmente sensível a multicolinearidade, embora seja sensível a ruído e observações discrepantes nos dados. Sua simplicidade conceitual, eficácia prática e capacidade de transformar modelos fracos em um ensemble robusto o tornaram uma escolha fundamental em machine learning, inspirando variantes mais sofisticadas como Gradient Boosting e XGBoost.


### Gradient Boosting



```{video} https://www.youtube.com/embed/3CC4N4z3GJc?si=DBa1Yk2cSGoPsHSA
```

O *Gradient Boosting* é uma generalização sofisticada do conceito de boosting que substitui a redistribuição adaptativa de pesos (como no AdaBoost) por uma otimização baseada em gradientes, tornando-o significativamente mais flexível e poderoso. Ao invés de simplesmente atribuir pesos maiores aos erros, o *Gradient Boosting* treina sequencialmente novos modelos para prever diretamente os resíduos (erros) do modelo anterior, utilizando o gradiente descendente para minimizar uma função de perda arbitrária — isso significa que cada nova árvore é ajustada não através de uma heurística de rebalanceamento, mas através de uma otimização matemática explícita na direção que mais reduz o erro. O processo funciona em três etapas iterativas: primeiro um modelo base simples faz previsões iniciais, depois calcula-se os erros residuais (diferença entre valores observados e preditos), e finalmente treina-se um novo modelo para prever esses resíduos; ao somar todas as previsões, os erros sucessivamente diminuem. O *Gradient Boosting* é aplicável tanto a regressão quanto a classificação, funciona muito bem com funções de perda diferenciáveis (como erro quadrático médio ou entropia cruzada), e é reconhecido como um dos algoritmos mais eficazes em competições de ciência de dados, tendo originado implementações otimizadas e altamente escaláveis como XGBoost, LightGBM e CatBoost que agregam paralelização, regularização avançada e gerenciamento de memória eficiente.

O *XGBoost* (*Extreme Gradient Boosting*), desenvolvido por Tianqi Chen, é uma implementação altamente otimizada e sofisticada do algoritmo de gradient boosting que se tornou praticamente sinônimo de excelência em machine learning, dominando competições de ciência de dados como o Kaggle. Enquanto o gradient boosting clássico já apresenta melhorias significativas, o XGBoost potencializa ainda mais esse conceito através de várias otimizações engenhosas: regularização L1 e L2 embutida para prevenir overfitting, paralelização de alto desempenho que permite usar múltiplos núcleos de CPU simultaneamente (acelerando o treinamento consideravelmente em comparação com implementações sequenciais tradicionais), e um manejo inteligente de dados ausentes que identifica automaticamente o melhor caminho para cada observação sem precisar de pré-processamento extenso. O XGBoost também oferece validação cruzada interna, métricas de importância de características para interpretabilidade, e suporte a funções de perda customizadas, tornando-o extraordinariamente versátil para regressão, classificação binária, multiclasse e ranking.


## Combinando Modelos Diferentes

Até então, a maior parte dos métodos aqui consistem em combinar diferentes iterações dos mesmos métodos de aprendizado de máquina. Ou seja, combinamos diversos tipos de árvores de decisão para atingir um melhor modelo único, seja no *Bagging* ou no *Boosting*. Terminaremos a aula discutindo dois métodos que usam diferentes classificadores (*SVM*, *KNN*, *Decision Trees*, etc) para agregar suas previsões em um único modelo.


### *Voting Classifiers*

Os Voting Classifiers são um método fundamental e  simples de *ensemble learning* que combina as previsões de múltiplos modelos independentes para produzir uma previsão final robusta, operando sob o princípio de que uma decisão em grupo frequentemente supera uma decisão individual. O método oferece duas estratégias distintas: o *hard voting*, onde cada modelo "vota" por uma classe e aquela com a maioria absoluta de votos é escolhida como previsão final (semelhante a uma eleição democrática simples), ideal para modelos que produzem apenas rótulos de classe; e o *soft voting*, onde cada modelo fornece probabilidades para cada classe possível, e a classe com a probabilidade média mais alta é selecionada como previsão final, exigindo que os modelos suportem o método predict_proba() mas oferecendo maior nuance ao considerar o nível de confiança de cada classificador. A força dos Voting Classifiers reside em sua diversidade: ao agregar modelos de tipos diferentes (como Logistic Regression, SVM, Decision Trees, KNN), capitaliza-se sobre os pontos fortes distintos de cada abordagem, mitigando suas fraquezas individuais e resultando em melhor generalização e robustez em dados não vistos, tornando-os especialmente valiosos quando não há clareza sobre qual modelo individual escolher ou quando recursos computacionais permitem o treinamento paralelo de múltiplos modelos sem custo proibitivo.

```{figure} ../aula11/images/geronfig7.2.png
---
width: 100%
name: votingclassifiers
align: center
---
*Voting Classifier* em ação. Fonte: Géron ({cite}`geron2022hands`.)
```

### *Stacking*

O *Stacking* é uma técnica sofisticada de ensemble learning que implementa uma abordagem em dois níveis hierárquicos para maximizar a performance preditiva através de uma estratégia fundamentalmente diferente tanto de *Voting* quanto de *Bagging/Boosting*. Diferentemente do *Voting Classifier* que combina previsões finais de forma relativamente simples, o *Stacking* treina um meta-modelo (ou *meta-learner*) que aprende como agregar inteligentemente as previsões de múltiplos modelos base — primeiro, todos os modelos base (Decision Trees, SVM, Random Forest, KNN, etc.) são treinados independentemente no conjunto de treinamento (geralmente através de validação cruzada k-fold para evitar vazamento de dados), depois suas previsões são coletadas e utilizadas como novas features para treinar o meta-modelo, que finalmente faz a previsão agregada. Essa abordagem oferece máxima flexibilidade: o meta-modelo pode aprender padrões complexos sobre quando confiar em cada modelo base, detectando padrões de erro ou especialidades de cada preditor e até descartando contribuições de modelos menos confiáveis em contextos específicos. No entanto, o Stacking é computacionalmente intensivo e exige cuidados cuidadosos com validação cruzada para evitar overfitting, tornando-o a escolha preferida em competições de machine learning e aplicações onde máxima acurácia justifica o investimento computacional adicional.

```{figure} ../aula11/images/geronfig7.12.png
---
width: 100%
name: votingclassifiers
align: center
---
*Stacking* em ação. Fonte: Géron ({cite}`geron2022hands`.)
```

## Conclusão


Nesta aula exploramos os métodos ensemble que transformam a ideia simples de "a sabedoria das multidões" em ferramentas computacionais sofisticadas capazes de superar significativamente modelos individuais. Começamos com o *Bagging*, que demonstra como treinar múltiplos modelos independentes em subconjuntos aleatórios dos dados reduz a variância através de agregação paralela, e vimos como o *Random Forest* aprimorou essa ideia ao decorrelacionar as árvores através de amostragem aleatória de preditores. Transitamos então para o *Boosting*, uma abordagem fundamentalmente diferente que reduz o viés através do aprendizado sequencial, começando com AdaBoost, evoluindo para a sofisticação do Gradient Boosting, e finalmente chegando em implementações otimizadas como XGBoost, que domina competições reais de ciência de dados através de paralelização, regularização e manejo inteligente de dados ausentes.

Com isso, finalizamos os principais modelos e métodos antes de *deep learning*. Agora, veremos por que esses modelos são chamados de *deep* e por que acabaram substituindo os modelos clássicos para as aplicações de aprendizado de máquina. No entanto, com isso vem o custo de menor interpretabilidade, e também maior custo computacional.




