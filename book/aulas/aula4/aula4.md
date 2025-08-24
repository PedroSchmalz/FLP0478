# Aprendizado Supervisionado


Na última aula vimos a representação textual *Bag-of-words*, que divide o texto com base nos seus componentes básicos, as palavras. Vimos também como um modelo de aprendizado supervisionado básico, a Regressão Multinomial, pode utilizar essa representação como *features* no treinamento, utilizando a frequência das palavras em cada documento para tentar prever a classe à qual aquele documento pertence. Hoje, discutiremos alguns conceitos que foram discutidos de maneira breve na seção de "Visão Geral do Curso". Definiremos o que é o Aprendizado de Máquina, o Aprendizado Supervisionado, e os principais conceitos relacionados à essas tarefas.

## Aprendizado de Máquina

O Aprendizado de Máquina é uma tecnologia dentro do campo de Inteligência Artificial que permite que computadores aprendam e façam predições sem programação explícita. Inteligência Artificial é a "Inteligência apresentada por artefatos (e.g. Máquinas), em oposição à inteligência natural (IN) apresentada por animais, como os humanos. [...] De maneira geral, definimos inteligência como a habilidade de perceber um ambiente, analisá-lo e tomar acões/decisões que maximizam a chance de atingir determinado objetivo" ({cite}`cerulli2023fundamentals`., p. 5). Como mostra a figura {numref}`Figura {number} <AIML>`, Aprendizado de Máquina é uma subárea da Inteligência artificial e, como definimos na primeira linha, permitem que a máquina tome acões/decisões com base em um conjunto de dados/experiências prévias, e não em programação explícita. Aprendizado de Máquina e Aprendizado Estatístico são utilizados de maneira intercambiável na literatura. Dentro da área de aprendizado supervisionado, temos a subárea de Aprendizado Profundo (ou *Deep Learning*), em que a característica definidora dos modelos são de que possuirão diversas camadas neurais (veremos o que é isso futuramente).


```{figure} ../aula4/images/AIML.png
---
width: 100%
name: AIML
align: center
---
 Relação Entre Inteligência Artificial, Aprendizado de Máquina e *Deep Learning*. Fonte: [Somos Tera](https://blog.somostera.com/data-science/deep-learning-vs-machine-learning)
```


a figura {numref}`Figura {number} <classicdiv>` mostra como a literatura faz a divisão clássica do Aprendizado de Máquina. Temos aplicações supervisionadas, em que um conjunto de valores $Y$ (*targets*) são preditos com base em um conjunto de variáveis explicativas (ou *features*). Existem dois tipos de aplicações superivisionadas: As com *targets* de valores contínuos (Regressão) e as de valores categóricos (Classificação). No decorrer do curso, focaremos em aplicações de Classificação. No entanto, existem também aplicações não supervisionadas, como as de *Clustering, que buscam encontrar padrões nos dados (e.g. Classificação de Tópicos, Divisão em grupos) sem que o humano/pesquisador forneça rótulos ou valores alvo. Há ainda uma terceira categoria, a dos métodos semi-supervisionados, que combinam um pequeno conjunto de dados rotulados com muitos dados não rotulados para melhorar o desempenho dos modelos. Por fim, existe o Aprendizado por Reforço (*Reinforcement Learning*), em que um agente interage com um ambiente e aprende, por tentativa e erro, a escolher ações que maximizem a recompensa acumulada ao longo do tempo. Aqui estão alguns exemplos típicos de cada família de aplicações:

- Classificação (supervisionado) – filtragem de e-mails spam × não spam.

- Regressão (supervisionado) – previsão de salário a partir de experiência profissional e localização.

- Clustering (não supervisionado) – segmentação de clientes em grupos com padrões de compra semelhantes.

- Semi-supervisionado – treinamento de um classificador de imagens médicas usando poucas tomografias rotuladas e milhares sem rótulo.

- Reinforcement Learning – agentes que aprendem a jogar Go ou a controlar braços robóticos por meio de recompensas de desempenho.



```{figure} ../aula4/images/classicdivision.png
---
width: 100%
name: classicdiv
align: center
---
 Divisão Clássica do Aprendizado de Máquina. Fonte: [Ribeiro e Gomes](https://www.researchgate.net/publication/374010223_On_the_Use_of_Machine_Learning_for_Damage_Assessment_in_Composite_Structures_A_Review) (2023) {cite}`ribeiro2023machinelearning`.
```


















