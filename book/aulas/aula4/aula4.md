# Aprendizado Supervisionado


Na última aula vimos a representação textual *Bag-of-words*, que divide o texto com base nos seus componentes básicos, as palavras. Vimos também como um modelo de aprendizado supervisionado básico, a Regressão Multinomial, pode utilizar essa representação como *features* no treinamento, utilizando a frequência das palavras em cada documento para tentar prever a classe à qual aquele documento pertence. Hoje, discutiremos alguns conceitos que foram discutidos de maneira breve na seção de "Visão Geral do Curso". Definiremos o que é o Aprendizado de Máquina, o Aprendizado Supervisionado, e os principais conceitos relacionados à essas tarefas.

## Aprendizado de Máquina

O Aprendizado de Máquina é uma tecnologia dentro do campo de Inteligência Artificial que permite que computadores aprendam e façam predições sem programação explícita. Inteligência Artificial é a "Inteligência apresentada por artefatos (e.g. Máquinas), em oposição à inteligência natural (IN) apresentada por animais, como os humanos. [...] De maneira geral, definimos inteligência como a habilidade de perceber um ambiente, analisá-lo e tomar acões/decisões que maximizam a chance de atingir determinado objetivo" ({cite}`cerulli2023fundamentals`., p. 5). Como mostra a {numref}`Figura {number} <AIML>`, Aprendizado de Máquina é uma subárea da Inteligência artificial e, como definimos na primeira linha, permitem que a máquina tome acões/decisões com base em um conjunto de dados/experiências prévias, e não em programação explícita. Aprendizado de Máquina e Aprendizado Estatístico são utilizados de maneira intercambiável na literatura. Dentro da área de aprendizado supervisionado, temos a subárea de Aprendizado Profundo (ou *Deep Learning*), em que a característica definidora dos modelos são de que possuirão diversas camadas neurais (veremos o que é isso futuramente).


```{figure} ../aula4/images/AIML.png
---
width: 100%
name: AIML
align: center
---
 Relação Entre Inteligência Artificial, Aprendizado de Máquina e *Deep Learning*. Fonte: [Somos Tera](https://blog.somostera.com/data-science/deep-learning-vs-machine-learning)
```


A {numref}`Figura {number} <classicdiv>` mostra como a literatura faz a divisão clássica do Aprendizado de Máquina. Temos aplicações supervisionadas, em que um conjunto de valores $Y$ (*targets*) são preditos com base em um conjunto de variáveis explicativas (ou *features*). Existem dois tipos de aplicações supervisionadas: As com *targets* de valores contínuos (Regressão) e as de valores categóricos (Classificação). No decorrer do curso, focaremos em aplicações de Classificação. No entanto, existem também aplicações não supervisionadas, como as de *Clustering, que buscam encontrar padrões nos dados (e.g. Classificação de Tópicos, Divisão em grupos) sem que o humano/pesquisador forneça rótulos ou valores alvo. Há ainda uma terceira categoria, a dos métodos semi-supervisionados, que combinam um pequeno conjunto de dados rotulados com muitos dados não rotulados para melhorar o desempenho dos modelos. Por fim, existe o Aprendizado por Reforço (*Reinforcement Learning*), em que um agente interage com um ambiente e aprende, por tentativa e erro, a escolher ações que maximizem a recompensa acumulada ao longo do tempo. Aqui estão alguns exemplos típicos de cada família de aplicações:

- Classificação (supervisionado) – filtragem de e-mails spam × não spam.

- Regressão (supervisionado) – previsão de salário a partir de experiência profissional e localização.

- Clustering (não supervisionado) – segmentação de clientes em grupos com padrões de compra semelhantes.

- Semi-supervisionado – treinamento de um classificador de imagens médicas usando poucas tomografias rotuladas e milhares sem rótulo.

- Reinforcement Learning – agentes que aprendem a jogar Go ou a controlar braços robóticos por meio de recompensas de desempenho.



```{figure} ../aula4/images/classicdivision.jpg
---
width: 100%
name: classicdiv
align: center
---
 Divisão Clássica do Aprendizado de Máquina. Fonte: [Ribeiro e Gomes](https://www.researchgate.net/publication/374010223_On_the_Use_of_Machine_Learning_for_Damage_Assessment_in_Composite_Structures_A_Review) (2023) {cite}`ribeiro2023machinelearning`.
```


O Paradigma central do aprendizado supervisionado se articula na ideia de traduzir uma tarefa cognitiva em um problema estatístico (Cerulli, p. 7). O aprendizado estatístico começa com a coleta de informações do passado armazenados em um objeto $D$, o **banco de dados**. Um banco de dados é uma coleção de informações sobre $N$ casos, nos quais observamos um resultado (ou *outcome*) $y$, e um conjunto $p$ de preditores (ou variáveis explicativas) $ X = (X_i, ..., Xp) $:


$$
D_i := {(y_i, \mathbf{x_i}), i = 1,\dots, N}
$$

Grosso modo, $D_i$ é igual ao conjunto dos pares ordenados $(y_i,x_i)$ com i indo de 1 a $N$, o tamanho do banco de dados. Na aplicação de PLN que estamos trabalhando ao longo do curso, $D_i$ é o banco de dados contendo todas as publicações dos políticos no *X*, os pares ordenados $(y_i,x_i)$ representam a nossa classificação para cada publicação(Sentimento ou Posicionamento), em $y_i$, e a representação do nosso texto em $x_i$. A tarefa principal do aprendizado de máquina é mapear, usando o banco de dados, os preditores $x_i$ para cada resultado $y_i$. Com isso, temos o seguinte algoritmo (ou função) geral:

$$
(x_1,...,x_p) \xrightarrow{\,f\,} y 
$$


### Função Erro


Para mapear isso da melhor maneira, precisamos de outra função: A **função erro**. A **função erro**, de forma muito geral e superficial, é um mapeamento  

$$
L : (y, \hat{y}) \;\longrightarrow\; \mathbb{R}_{\ge 0}
$$

que devolve um escalar não-negativo indicando o quanto a predição $\hat{y}=f(\mathbf{x})$ diverge do valor verdadeiro $y$. Traduzindo, o modelo de aprendizado de máquina utilizará a função erro para entender o quão distante ele está do melhor resultado. Nas sucessivas iterações, ele vai tentar minimizar o erro, consequentemente minimizando a função de custo. Uma função de erro comum na regressão linear é o MSE (*Mean Squared Error*):

$$
MSE (X) = E[(y- f(x)²| x)]
$$

Cada tarefa de aprendizado de máquina terá sua função de erro específica (até a não supervisionada). Não é necessário memorizar todas as funções erro/custo, e alguns modelos (como os de aprendizado profundo) usam funções erro próprias. O importante é entender o que são e que o objetivo do modelo, numa aplicação deste tipo, é o de reduzir o erro. O melhor modelo, na concepção clássica do Aprendizado de máquina, é aquele que consegue o melhor resultado na aproximação de $E(y|x)$. Ou seja, o que consegue o melhor resultado (Acurácia, Precisão, etc.) utilizando as *features* para tentar prever o *target*.



```{admonition} 💬 Com a palavra, os autores:
:class: quote
"Em Aprendizado de Máquina, prever a variável-alvo é tão central que podemos definir a área como um conjunto de estratégias de modelagem (paramétricas ou não paramétricas) cujo objetivo é obter uma aproximação confiável de $E(y∣x)$, tomando a acurácia de predição como princípio orientador. Assim, alguns métodos podem ser considerados superiores a outros desde que a predição seja o único propósito da análise. A estimativa estatística de $E(y∣x)$ está sujeita a dois tipos de erro possíveis: (1) erro amostral e (2) erro de especificação."
({cite}`cerulli2023fundamentals`, p. 15, tradução nossa)
```



















