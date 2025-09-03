# Aprendizado Supervisionado II

Na última aula, vimos que o Aprendizado de máquina é um campo dentro da Inteligência artificial que busca, com base em experiências prévias (i.e. dados de treinamento), fazer classificações ou previsões para nossa variável de interesse (*target*). Para isso, os métodos de aprendizado estatístico precisam de uma função erro, que busca reduzir a distância entre os valores previstos e os valores reais dos dados de treinamento. Vimos também os passos principais para a criação de um banco de dados de treinamento (Codebook, anotação, concordância entre anotadores), e o *pipeline* básico de uma aplicação de classificação. Destacamos métricas essenciais como acurácia, precisão, recall e F1-score para avaliar o desempenho dos classificadores. Por fim, reforçamos a importância de testar o modelo em dados novos para garantir sua capacidade de generalização e utilidade prática. Na aula de hoje, retomaremos algumas discussões da aula anterior, e discutiremos a diferença entre inferência e predição, métodos paramétricos e não-paramétricos, e os trade-offs clássicos de aprendizado de máquina entre Flexibilidade x Interpretabilidade, e viés x variância.


Como dito, o objetivo de uma tarefa de aprendizado de máquina é usar um conjunto de dados para fazer previsões e classificações para outros dados não observados durante o treinamento. Esse conjunto de dados é conhecido como o **banco de treinamento** (ou Córpus anotado, em PLN), e é composto de dois tipos de variáveis principais: A variável ***Target*** (alvo), representada muitas vezes por $Y$, e também conhecida por variável resposta, variável dependente, ou variável explicada. O segundo tipo de variável é o que é chamado, no *ML*, de ***Features***, representadas por $\mathbf{X}$, que são vetores de variáveis preditoras, ou independentes/explicativas, ou variáveis *input*. No contexto de Processamento da Língua Natural, nosso principal $X$ é o texto em cada documento, e o principal $Y$ (em classificação) são as categorias de interesse, seja sentimento, posicionamento, relevância, tópicos, etc. Na literatura de Aprendizado Estatístico e de Aprendizado de máquina, $X$ e $Y$ podem ser referidos de todas as formas mencionadas acima, mas significam a mesma coisa. 

Com esse banco de dados de treinamento, contendo $\mathbf{X}$ e $Y$, o principal objetivo é de modelar a relação entre eles, tal que tenhamos

$$
 Y = f(x) + \epsilon
$$

Em que $Y$ é nosso *target* (e.g. Sentimento, posicionamento) e $\mathbf{x}$ é o vetor de nossas variáveis explicativas (o Texto em representação numérica). $f(x)$ seria, portanto, uma função geral que representa a relação entre nossas *features* e o *target*.

## Por que estimar $f(x)$?


Existem dois contextos em que o pesquisador está interessado em estimar $f(x)$: Inferência e Predição. Grosso modo, as pesquisas em **inferência** procuram entender o impacto de cada variável explicativa ($X_1, X_2, ..., X_3$) em $Y$, e como essa relação se altera com a inclusão de novas variáveis, interações, polinômios, etc. Por exemplo, uma pesquisa pode estar preocupada em entender como a religião de um indivíduo pode impactar em seu apoio ao bolsonarismo. Um possível resultado dessa pesquisa poderia ser de que o indivíduo ser evangélico tem um efeito positivo constante no apoio ao bolsonarismo, em comparação com outras religiões/denominações. 

No contexto da **predição**, o foco é em utilizar os dados de treinamento rotulados (e, com isso, as variáveis $X_1, X_2, ..., X_3 $) para prever os valores de $Y$, sejam estes valores contínuos ou categóricos. Um exemplo clássico de classificação neste contexto é o de classificação de e-mails em *Spam* ou não *Spam*. Nessa tarefa, utiliza-se o texto do email em alguma representação numérica (*Bag-of-words*, *embeddings*, etc.) para a classificação binária de *Spam* ou não. 

### Predição

No contexto de predição, estimamos os seguintes valores:

$$
\hat{y} = \hat{f}(\mathbf{x})
$$


Onde $\hat{f}$ é a estimativa de $f$ e $\hat{y}$ é a estimativa de $y$. Aqui, $\hat{f}$ é tratado como uma "caixa preta", no sentido de que a preocupação não é com sua forma, nem com sua especificação, mas se ele fornece boas previsões de $y$. No exemplo do e-mail, não importa quais palavras são melhores preditoras de se um e-mail é ou não *spam*, mas sim que o modelo consiga classificar corretamente essa categoria, na maior parte dos casos. A precisão da função $\hat{f}$ é determinada por dois tipos de erro: um redutível e outro irredutível. Mesmo que tenhamos um ótimo modelo e especificação, ainda existirá uma parcela de erro devido à fatores estocásticos (i.e. aleatórios).

$$
 E(y-\hat{y}) = E[f(x)+ \epsilon - \hat{f}(x)]² \\
 = [f(x)-\hat{f}(x)]² + Var(\epsilon)
$$

Onde

$$
\underbrace{[f(x)-\hat{f}(x)]^2}_{\text{Erro redutível: diferença entre a função verdadeira e a estimada}} \\
$$

$f(x)$ seria a verdadeira relação de variáveis que melhor explicam e prevem $y$ (ou o verdadeiro *Data Generating Process*) e $\hat{f}(x)$ é a função que o pesquisador estabeleceu com as variáveis existentes no banco de dados. Sempre é possível, com base na rotulação de treinamento, reduzir a diferença entre o que encontramos nos dados e o que melhor aproxima $y$. No entanto, o outro componente da equação é

````{margin}
```{note}
Um problema comum que pode existir em aplicações de aprendizado de máquina é o *data leakage*. *Data leakage* é um problema que ocorre quando informações do conjunto de teste ou de validação acabam sendo utilizadas, direta ou indiretamente, durante o treinamento do modelo. Isso faz com que o modelo tenha acesso a dados que não deveria conhecer, levando a resultados artificialmente altos nas métricas de avaliação e prejudicando sua capacidade de generalização para dados realmente novos. Portanto, sabendo que existe um erro irredutível nas aplicações de predição, resultados **bons demais** na validação e teste (i.e. resultados muito próximos da perfeição) podem indicar que o pesquisador está com vazamento de dados.
```
````



$$
+ \underbrace{Var(\epsilon)}_{\text{Erro irredutível: variabilidade aleatória dos dados}}
$$

Esse erro é irredutível e estocástico, e sempre estará presente em qualquer aplicação, seja ela inferencial ou de previsão. Esse erro faz com que, independente da nossa especificação de $\hat{f}(x)$, $E(y-\hat{y})$ nunca será igual a zero.

### Inferência













## Conclusão

O aprendizado supervisionado é uma abordagem fundamental para análise de textos e classificação de documentos em Processamento de Linguagem Natural. Ao longo do processo, é essencial construir um banco de treinamento confiável, com regras claras de anotação e validação, garantindo objetividade, replicabilidade e generalizabilidade dos resultados. A escolha do modelo de aprendizado de máquina deve considerar o tipo de problema, a qualidade dos dados e o objetivo da análise, equilibrando simplicidade, interpretabilidade e desempenho. A avaliação rigorosa do classificador, por meio de métricas como acurácia, precisão, recall e F1-score, assegura que o modelo seja capaz de generalizar para novos dados e produzir resultados úteis em aplicações reais. Por fim, aplicar o modelo em um banco de teste é indispensável para validar sua capacidade de classificação em situações inéditas, consolidando o papel do aprendizado supervisionado como ferramenta poderosa para extrair conhecimento e apoiar decisões baseadas em grandes




















