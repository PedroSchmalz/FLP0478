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

Criar um banco de treinamento (ou usar um já existente) é o primeiro passo em toda aplicação de Aprendizado de Máquina. Para isso, é necessário codificação/anotação por feita por humanos. A codificação humana de textos existe há muito tempo para a organização e quantificação de textos (e.g. gêneros textuais, gêneros musicais, categorias de livros). Assim que um pesquisador define categorias, a codificação humana é o processo de colocar manualmente esses documentos em categorias. Esse processo é uma combinação de um *codebook*, o treinamento de anotadores, e os processos internos específicos de cada anotador. Com base em Neuendorf (2016), Grimmer et al. estabelecem as seguintes características de um bom banco de treinamento:

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

Esses são exemplos de uma única tarefa de codificação dentro do nosso projeto, e a mais "direta". Mesmo nessa tarefa simples, precisamos definir muitas regras para desambiguar situações atípicas e garantir a replicabilidade e confiabilidade do nosso conjunto de treinamento. Veremos, futuramente, que a classificação de Posicionamento e Sentimento são mais complexas e geraram menor concordância entre anotadores, além de problemas de desbalanceamento dos dados.


#### b) Escolher e Gerenciar anotadores

Seguindo o desenvolvimento de regras detalhadas para cada categoria de relevância, posicionamento e sentimento, foi criado um manual de codificação para treinar a equipe de anotadores; além disso, literatura sobre hesitação vacinal foi compartilhada a fim de aprimorar a compreensão da complexidade das atitudes e emoções em relação à vacinação. A equipe de pesquisa também recebeu treinamento sobre as diferenças de contexto dos três anos analisados (por exemplo, 2020 como ano sem vacinas; 2021, início da vacinação de adultos e adolescentes; e 2022, início da imunização de crianças e bebês contra a SARS-CoV-2).

A anotação do corpus foi realizada em 61 rodadas. Em cada rodada, uma amostra aleatória de 200 publicações era classificada usando um banco de dados criado especificamente para o projeto, no qual os posts eram anonimizados, removendo informações sobre autor, data ou imagens associadas à mensagem. O processo de classificação teve duas etapas: 

Relevância — Três anotadores classificavam cada post como relevante ou não relevante. Em seguida, conflitos eram revisados por três pesquisadores seniores.

Posicionamento e sentimento — Depois de resolvidos os conflitos de relevância, os posts considerados relevantes eram novamente anotados, agora quanto ao posicionamento e ao tipo de sentimento, de forma independente por três anotadores. Conflitos nessas duas categorias também eram revistos pelos supervisores de pesquisa.


#### c) Selecionar Documentos

O processo de amostragem teve duas fases: Primeiro, definiu-se que o grupo político a ser estudado era os candidatos que estavam concorrendo às eleições municipais nas capitais em 2020, para o cargo de prefeito. Depois, coletou-se todas as publicações feitas por esses indivíduos nos anos 2020, 2021 e 2022. Para a anotação do córpus final, foram amostradas aleatoriamente 9.045 publicações, que ficaram divididas em 61 rodadas de aproximadamente 200 publicações. 


#### d) Checar Confiabilidade

A confiabilidade da anotação de um córpus pode ser checada de algumas maneiras. A primeira é se consultar regularmente com os anotadores para verificar se não há contextos que não foram levados em conta, se eles conseguiram entender o processo de anotação, e se eles concordam entre si. Existem métricas que permitem mensurar o quanto os anotadores concordam em cada fase da anotação, comparando os rótulos dados por cada um em determinado documento. A {numref}`Figura {number} <krippendorf>` mostra o quanto a concordância entre anotadores variou ao longo das três tarefas (Relevância, Posicionamento e Sentimento). [Veja mais](https://pt.wikipedia.org/wiki/Alfa_de_Krippendorff) sobre o Alfa de Krippendorf.


```{figure} ../aula4/images/Krippendorff_Alpha.png
---
width: 100%
name: krippendorf
align: center
---
Alfa de Krippendorf para as três tarefas de anotação. Fonte: Barberia et al., 2025 ({cite}`barberia2025its`)
```

A relevância apresentou o maior acordo geral entre os anotadores, com alfa médio de 0,94 e variação mínima entre as rodadas. Essa tarefa se beneficiou de um conjunto de dados mais balanceado, menos conflitos entre os anotadores e um número total maior de observações. Os critérios de sentimento e posicionamento obtiveram acordo moderado. O alfa médio foi 0,67 para Sentimento e 0,70 para Posicionamento. Contudo, ambas as tarefas mostraram grande variabilidade ao longo das rodadas. Essa oscilação pode ser atribuída ao desequilíbrio entre classes nessas duas classificações, o que gerou maior discordância nos conteúdos analisados em cada rodada, sobretudo nas categorias minoritárias (desfavorável e indefinido).


### 2. Escolher o Modelo de Aprendizado de Máquina

Após a criação de um banco de treinamento confiável e bem anotado, o próximo passo é selecionar o modelo de aprendizado de máquina mais adequado para a tarefa. A escolha do modelo depende de diversos fatores, como o tipo de problema (classificação, regressão, etc.), o tamanho e a qualidade dos dados, o número de categorias, e o objetivo da análise.

No contexto de Processamento de Linguagem Natural (PLN), modelos clássicos como Regressão Logística, Naive Bayes e Árvores de Decisão são frequentemente utilizados para tarefas de classificação de texto, especialmente quando se trabalha com representações simples como *Bag-of-Words*. Para problemas mais complexos ou conjuntos de dados maiores, modelos de aprendizado profundo, como redes neurais e transformadores (por exemplo, BERTimbau para português), podem oferecer ganhos significativos de desempenho.

É importante considerar também a interpretabilidade do modelo, o tempo de treinamento, e a facilidade de ajuste dos hiperparâmetros (discutiremos isso novamente na próxima aula). Muitas vezes, recomenda-se começar com modelos mais simples e, conforme necessário, avançar para modelos mais sofisticados. Independentemente da escolha, o modelo deve ser treinado utilizando o banco de dados anotado, buscando aprender padrões que permitam prever corretamente os rótulos dos novos exemplos.

Por fim, a seleção do modelo deve ser acompanhada de uma avaliação rigorosa de sua performance, utilizando métricas apropriadas e validação em dados não vistos, para garantir que o classificador seja confiável.

### 3. Checar a Performance

A qualidade de um classificador é avaliada por métricas como **acurácia**, **precisão**, **recall** e **F1-score**, que medem o quão bem o modelo consegue distinguir entre as diferentes classes ([Veja Mais](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall?hl=pt-br)). Além disso, é fundamental validar o desempenho do modelo em dados não vistos durante o treinamento, garantindo que ele não apenas memorize os exemplos, mas realmente aprenda padrões úteis e generalize para outros conjuntos de documentos relacionados.

A **acurácia** é uma das métricas mais utilizadas para avaliar o desempenho de um classificador. Ela indica a proporção de previsões corretas feitas pelo modelo em relação ao total de exemplos avaliados. Em outras palavras, a acurácia mostra o quanto o modelo acertou ao classificar os documentos ou exemplos em suas respectivas categorias.

A fórmula da acurácia é:

$$
\text{Acurácia} = \frac{\text{Número de previsões corretas}}{\text{Número total de exemplos}}
$$

Por exemplo, se um modelo classificou corretamente 80 de 100 exemplos, sua acurácia será 0,8 (ou 80%). Embora seja uma métrica intuitiva e fácil de interpretar, é importante lembrar que ela pode ser enganosa em situações de classes desbalanceadas, onde outras métricas como precisão, recall e F1-score também devem ser consideradas.

A **precisão** (ou *precision*) é outra métrica importante para avaliar o desempenho de um classificador, especialmente em situações onde o custo de falsos positivos é alto. A precisão indica, dentre todas as previsões positivas feitas pelo modelo, qual proporção realmente corresponde a exemplos positivos. Em outras palavras, ela mostra o quanto o modelo foi assertivo ao identificar exemplos de uma determinada classe.

A fórmula da precisão é:

$$
\text{Precisão} = \frac{\text{Número de verdadeiros positivos}}{\text{Número total de exemplos classificados como positivos}}
$$

Por exemplo, se o modelo classificou 50 exemplos como positivos, mas apenas 40 deles eram realmente positivos, a precisão será $40/50 = 0,8$ (ou 80%). A precisão é especialmente relevante quando queremos evitar que o modelo faça muitas previsões positivas incorretas, como em tarefas de detecção de fraudes ou diagnósticos médicos.

O **recall** (ou sensibilidade) é uma métrica que indica a capacidade do classificador de identificar corretamente todos os exemplos positivos presentes no conjunto de dados. Em outras palavras, o recall mostra a proporção de exemplos positivos que foram corretamente classificados como positivos pelo modelo.

A fórmula do recall é:

$$
\text{Recall} = \frac{\text{Número de verdadeiros positivos}}{\text{Número de verdadeiros positivos} + \text{Número de falsos negativos}}
$$

Ou seja, é a proporção de exemplos positivos corretamente identificados pelo modelo, considerando todos os exemplos que realmente pertencem à classe positiva (verdadeiros positivos + falsos negativos).

O **F1-score** é uma métrica que combina precisão e recall em um único valor, representando a média harmônica entre as duas. O F1-score é útil para avaliar o desempenho do modelo quando há um equilíbrio desejado entre precisão e recall, especialmente em conjuntos de dados desbalanceados.

A fórmula do F1-score é:

$$
\text{F1-score} = 2 \times \frac{\text{Precisão} \times \text{Recall}}{\text{Precisão} + \text{Recall}}
$$

O F1-score varia de 0 a 1, sendo 1 o valor ideal, indicando que o modelo tem alta precisão e alto recall ao mesmo tempo.

### 4. Aplicar no Banco de Teste

Após treinar e validar o modelo de aprendizado supervisionado, o passo final é aplicar o classificador no banco de teste. O banco de teste consiste em exemplos que não foram utilizados durante o treinamento do modelo, permitindo avaliar sua capacidade de generalização para dados novos e reais. Ao aplicar o modelo no banco de teste, obtemos previsões para cada exemplo e podemos calcular as métricas de desempenho (acurácia, precisão, recall, F1-score) de forma mais confiável. Esse processo garante que o modelo não apenas aprendeu padrões específicos do conjunto de treinamento, mas também é capaz de realizar classificações corretas em situações inéditas, tornando-se útil para aplicações práticas e futuras

## Conclusão

O aprendizado supervisionado é uma abordagem fundamental para análise de textos e classificação de documentos em Processamento de Linguagem Natural. Ao longo do processo, é essencial construir um banco de treinamento confiável, com regras claras de anotação e validação, garantindo objetividade, replicabilidade e generalizabilidade dos resultados. A escolha do modelo de aprendizado de máquina deve considerar o tipo de problema, a qualidade dos dados e o objetivo da análise, equilibrando simplicidade, interpretabilidade e desempenho. A avaliação rigorosa do classificador, por meio de métricas como acurácia, precisão, recall e F1-score, assegura que o modelo seja capaz de generalizar para novos dados e produzir resultados úteis em aplicações reais. Por fim, aplicar o modelo em um banco de teste é indispensável para validar sua capacidade de classificação em situações inéditas, consolidando o papel do aprendizado supervisionado como ferramenta poderosa para extrair conhecimento e apoiar decisões baseadas em grandes




















