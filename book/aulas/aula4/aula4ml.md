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


A {numref}`Figura {number} <classicdiv>` mostra como a literatura faz a divisão clássica do Aprendizado de Máquina. Temos aplicações supervisionadas, em que um conjunto de valores $Y$ (*targets*) são preditos com base em um conjunto de variáveis explicativas (ou *features*). Existem dois tipos de aplicações supervisionadas: As com *targets* de valores contínuos (Regressão) e as de valores categóricos (Classificação). No decorrer do curso, focaremos em aplicações de Classificação. No entanto, existem também aplicações não supervisionadas, como as de *Clustering*, que buscam encontrar padrões nos dados (e.g. Classificação de Tópicos, Divisão em grupos) sem que o humano/pesquisador forneça rótulos ou valores alvo. Há ainda uma terceira categoria, a dos métodos semi-supervisionados, que combinam um pequeno conjunto de dados rotulados com muitos dados não rotulados para melhorar o desempenho dos modelos. Por fim, existe o Aprendizado por Reforço (*Reinforcement Learning*), em que um agente interage com um ambiente e aprende, por tentativa e erro, a escolher ações que maximizem a recompensa acumulada ao longo do tempo. Aqui estão alguns exemplos típicos de cada família de aplicações:

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


## Função Erro


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

## Classificação Supervisionada em PLN

A tarefa de classificação é uma das aplicações mais comuns do aprendizado supervisionado, especialmente em Processamento de Linguagem Natural (PLN). O objetivo central da classificação é atribuir um rótulo ou categoria a cada exemplo do conjunto de dados, com base em suas características (features). No contexto do curso, isso significa, por exemplo, identificar o sentimento ou posicionamento de uma publicação de político a partir do texto (representado numericamente) de uma publicação no *X*. A classificação supervisionada usa um conjunto de documentos rotulados em categorias para criar um modelo estatístico que relaciona as palavras nos documentos aos rótulos, e aplica este modelo para conseguir rótulos comparáveis para outros documentos não rotulados. Nesse tipo de aplicação, o pesquisador é responsável por providenciar três tipos de informação: as categorias/rótulos para os documentos de treinamento; a representação do texto (*BOW* ou outra), e a classe de modelos que vai ser utilizada no treinamento. Na tarefa de classificação, o modelo de aprendizado de máquina também é referido como *Classificador*.

No projeto *Mapping Political Elites COVID-19 Vaccine Tweets in Brazil*, o objetivo é o de anotar publicações no X (antigo *Twitter*) de políticos brasileiros sobre as vacinas de COVID-19 no período da pandemia. As categorias de anotação são: Se as publicações são relevantes com relação às vacinas de COVID-19 (**Relevância**), se possuem sentimentos negativos, positivos ou indeterminados (**Análise de Sentimento**); e se posicionam de forma contrária, favorável ou indeterminada com relação às vacinas (**Detecção de Posicionamento**). Mais detalhes sobre o processo de anotação, coleta, e os dados disponíveis estão no [Github](https://github.com/NUPRAM/CoViD-Pol) do NUPRAM (Núcleo de Políticas, Redes Sociais e Aprendizado de Máquina). Esse é um projeto no formato clássico de uma tarefa de classificação em PLN: Temos três categorias para cada publicação (Relevância, Sentimento e Posicionamento); temos a representação do texto (*Embeddings* do BERTimbau, que veremos no final do curso); e temos a classe de modelos (O BERTimbau, um modelo de aprendizado profundo para o português brasileiro). Segundo Grimmer et al., Um processo de anotação tem quatro passos:


### 1. Criar o banco de Treinamento

Criar um banco de treinamento (ou usar um já existente) é o primeiro passo em toda aplicação de Aprendizado de Máquina. Para isso, é necessário codificação/anotação por feita por humanos. A codificação humana de textos é existe há muito tempo para a organização e quantificação de textos (e.g. gêneros textuais, gêneros musicais, categorias de livros). Assim que um pesquisador define categorias, a codificação humana é o processo de colocar manualmente esses documentos em categorias. Esse processo é uma combinação de um *codebook*, o treinamento de anotadores, e os processos internos específicos de cada anotador. Com base em Neuendorf (2016), Grimmer et al. estabelecem as seguintes características de um bom banco de treinamento:

* Objetividade-Intersubjetividade: A categoria mensurada é objetiva, e seu entendimento não é restrito à uma única pessoa (ou grupo de pessoas);
* Desenho *a priori*: O banco de dados deve ser classificado com base em um *codebook*;
* Confiabilidade: Diferentes conjuntos de anotadores humanos deveriam ser capazes de atingir mais ou menos a mesma classificação no mesmo conjunto de dados;
* Validade: A métrica deve estar alinhada com o conceito de interesse;
* Generalizabilidade: O banco de treinamento será baseado em uma amostra de um conjunto maior de documentos. A anotação feita nessa amostra deve ser generalizável para o resto dos documentos;
* Replicabilidade: Outros pesquisadores e anotadores deve ser capazes de replicar a anotação no mesmo conjunto de dados (ou aplicar em outros conjuntos de documentos).


#### a) Criando um codebook

Um *codebook* é um documento que define de forma clara e detalhada as categorias, critérios e regras que devem ser seguidos durante o processo de anotação dos dados. Ele serve como guia para os anotadores humanos, garantindo que todos compreendam e apliquem os conceitos de maneira consistente e objetiva. O codebook descreve exemplos, contraexemplos e situações ambíguas, ajudando a reduzir interpretações subjetivas e aumentando a confiabilidade e a replicabilidade da classificação. Em projetos de aprendizado supervisionado, um codebook bem elaborado é fundamental para assegurar que o banco de treinamento reflita fielmente o conceito de interesse e possa ser utilizado por outros pesquisadores/anotadores. Vamos pegar o exemplo da nossa classificação de Relevância das publicações de políticos. Essas foram as regras: Posts cujo conteúdo se referia a vacinas e à vacinação contra a COVID-19 receberam valor 1, enquanto posts que apenas continham palavras-chave, mas não tratavam de vacinas/vacinação contra a COVID-19, receberam valor 0. Posts classificados como **relevantes** incluíam:

* Citação direta de vacinas/vacinação contra a COVID-19;

* Referência indireta a vacinas e vacinação contra a COVID-19 (por exemplo, discussão sobre outras vacinas e/ou campanhas de vacinação);

* Termos específicos — ou que possam ser inferidos como tais — relativos às vacinas contra a COVID-19, como “segunda e terceira doses/aplicações”, “doses de reforço”;

* Considerando o período deste estudo, tweets que mencionem vacinas e/ou vacinação no Brasil ou em outros países, mesmo que não especifiquem COVID-19;

* Menção ao trabalho relacionado a vacinas de instituições (por exemplo, Fiocruz, Butantan), cientistas (por exemplo, Peter Hotez) ou políticos, ou ainda opiniões e comportamentos pró-vacina ou anti-vacina (por exemplo, Osmar Terra, CPICOVID19), mesmo em hashtags (#);

* Menção ao trabalho relacionado a vacinas de laboratórios, indústrias ou organizações responsáveis pela produção ou desenvolvimento de vacinas, como Fiocruz, Butantan, Covaxin, AstraZeneca, Oxford etc.;

* Campanhas de vacinação e comunicados de utilidade pública mencionando faixas etárias específicas e limitadas (por exemplo, “Vacinação para 37–39 anos começa amanhã”), pois tais anúncios eram quase exclusivamente para campanhas de vacinação contra a COVID-19;

* Mensagens que discutam terapias e tratamentos para infecção por COVID-19 no contexto da própria COVID-19;

* Menções à imunização obtida por vacinas ou por contaminação com o vírus da COVID-19 (por exemplo, imunização natural, imunização de rebanho etc.); ou

* Tweets que incluam termos como “negacionista”, “negacionismo” e equivalentes, que se refiram direta ou indiretamente às posições da elite política sobre vacinas.

Os dois exemplos abaixo são de *tweets* reais considerados relevantes na nossa anotação:

```{admonition} 🐦 Tweet
:class: tweet
**@capitaoassumção**: Vacina Para todos!
13:00 · 14 mar. 2021
```

```{admonition} 🐦 Tweet
:class: tweet
**@celsorussomano**: Cuidado com a ditadura que querem nos impor com a vacina da COVID-19. Estamos falando de uma vacina experimental, e todos nos corremos o risco de sermos cobaias. Isso é um desrespeito com a vida dos paulistanos, e não podemos aceitar que siga adiante!
12:34 · 12 fev. 2021
```



Posts considerados **não relevantes** incluíam mensagens que se referiam a:

* Vacinação em animais;

* “Vacina” usada como metáfora para outro tema (por exemplo, transparência como vacina contra a corrupção);

* Mensagem sobre outro assunto, mas contendo uma hashtag de vacina da COVID-19;

* Ausência de menção a vacinação, vacinas, laboratórios ou qualquer palavra apresentada acima como relevante;

* Mensagens em idioma estrangeiro que seriam relevantes se estivessem em português; ou

* Mensagens que discutam terapias e tratamentos para infecção por COVID-19, mas cujo contexto não esteja relacionado à COVID-19.


Abaixo estão dois exemplos também reais de tweets *Irrelevantes*:

```{admonition} 🐦 Tweet
:class: tweet
**@anisiomaiapb**: Alguns partidos ainda funcionam como monarquias onde as decisões são tomadas entre pai e filho ou entre marido e esposa. Não passam de projetos familiares. Ainda bem que o PT está vacinado contra isto e continuamos militando num projeto plural, democrático e participativo.
```



```{admonition} 🐦 Tweet
:class: tweet
**@JoaoCampos**: O @governope decretou, a partir de hoje, a obrigatoriedade do uso de máscara para quem trabalha em estabelecimentos comercias durante a pandemia. Mas, se puder, saia de casa sempre de máscara, ela é a única vacina que temos contra o coronavírus.
```

A segunda publicação, apesar de ser durante a pandemia e referenciar as máscaras, não se refere às vacinas de COVID-19, mas usa as máscaras como analogia à vacina. Portanto, é considerada Irrelevante. 

Esses são exemplos de uma única tarefa de codificação dentro do nosso projeto, e a mais "direta". Mesmo nessa tarefa simples, precisamos definir muitas regras para desambiguar situações atípicas e garantir a replicabilidade e confiabilidade do nosso conjunto de treinamento.





### Métricas de Classificação

A qualidade de um classificador é avaliada por métricas como acurácia, precisão, recall e F1-score, que medem o quão bem o modelo consegue distinguir entre as diferentes classes. Além disso, é fundamental validar o desempenho do modelo em dados não vistos durante o treinamento, garantindo que ele não apenas memorize os exemplos, mas realmente aprenda padrões úteis.

Em PLN, tarefas de classificação incluem desde a filtragem de spam, análise de sentimentos, identificação de tópicos, até a detecção de fake news. O sucesso dessas aplicações depende de uma boa representação dos dados, escolha adequada do modelo e validação rigorosa dos resultados. Assim, a classificação se torna uma ferramenta poderosa para extrair conhecimento e tomar decisões baseadas em dados.


## *Trade-offs* do Aprendizado de Máquina





## Como criar um Banco de Treinamento















