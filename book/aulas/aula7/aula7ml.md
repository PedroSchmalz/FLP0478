# Classificação com texto

Na última aula, vimos que o problema de classificação exige métodos específicos quando a variável resposta é categórica, como sentimentos, diagnósticos ou posicionamentos. Discutimos por que a regressão linear não é adequada para esse tipo de tarefa e exploramos alternativas como a regressão logística, que modela probabilidades de forma apropriada. Aprendemos sobre extensões da regressão logística para múltiplos preditores e múltiplas classes, além de conhecer os modelos generativos, como LDA, QDA e Naive Bayes, que modelam o processo de geração dos dados e utilizam o Teorema de Bayes para estimar probabilidades de pertencimento às classes. Por fim, destacamos as vantagens, limitações e pressupostos de cada abordagem, reforçando a importância de escolher o método mais adequado ao contexto do problema.

Na aula de hoje, iremos discutir como utilizar esses modelos para a classificação supervisionada **com texto**. Para isso, precisamos entender como representar a língua de maneira computacional, e qual será nossa unidade mínima de análise neste caso. Veremos o que são modelos de linguagem e como nos ajudam para esta tarefa, discutindo o uso de *n-grams*, como prever a probabilidade da próxima palavra em uma frase, etc.


## Do Número para a Língua, da Língua para o Número

Na aula anterior, discutimos como aplicar classificadores (modelos de aprendizado supervisionado para classificação) no contexto de variáveis quantitativas numéricas: tinhamos características quantitativas de indivíduos (renda, saldo do cartão, etc.) e queríamos classificar esses indivíduos como potenciais devedores ou não; Ou tínhamos informações de saúde (também quantitativas) e queríamos classificar os indivíduos como pessoas com diabetes ou não. No entanto, o objetivo do curso é ensiná-los a classificar observações em categorias (favorável/desfavorável, positivo/negativo) de acordo com o conteúdo textual (conteúdo de uma publicação no *Twitter*). Para isso, precisamos entender como representar a língua de maneira computacional. Daí, surge a seguinte pergunta: Qual a unidade mínima quando tratamos, computacionalmente, a língua? Para Caseli e Nunes (2024, {cite}`caseli_nunes_pln_2024`.), depende do seu critério e finalidade: Na fonologia, será o fonema; na morfologia, o morfema; podemos separar a língua em palavras, caractéres, e assim por diante. Para os propósitos dessa disciplina, focaremos em uma unidade: O *Token*.



```{admonition} 💬 Com a palavra, os autores:
:class: quote
"Ao fazer o processamento computacional de textos escritos, a definição de que tipo de unidade de processamento se quer buscar/estudar parece estar atrelada às necessidades da tarefa ou trabalho pretendidos. Geralmente, considera-se que uma palavra é, simplesmente, uma unidade grafológica delimitada, nas línguas europeias, entre espaços em branco na representação gráfica, ou entre um espaço em branco e um sinal de pontuação. Essa é uma definição bastante concreta, e bastante prática. No entanto, ao pensarmos em nossos modelos computacionais e suas aplicações no mundo, é importante nos aprofundarmos um pouco mais na conceituação do que é uma palavra e nas possibilidades de processamento e implicações das decisões tomadas no pré-processamento dos corpora."
({cite}`caseli_nunes_pln_2024`., p. 68, tradução nossa)
```


### Token e Type

*Token* é um termo que significa qualquer sequência de caractéres à qual se atribui um valor. Nas línguas européias, a sequência consiste em caractéres delimitados por espaços gráficos, e a tokenização é ajustada para lidar com sinais de pontuação. No entanto, isso não é verdade para todas as línguas. Mas, com essa definição, podemos associar o *token* à palavra escrita. E o *type* seriam os tokens/palavras únicos encontrados em uma frase ou texto. Vamos retomar o que foi discutido na seção "*Bag-of-words* e o modelo multinomial" da terceira aula do curso. Vimos que o modelo *bag-of-words* tem como ideia principal a de representar cada documento pelo número de vezes que cada palavra aparece nele. No exemplo, tínhamos:

1. O cachorro ama o osso;
2. O Osso ama o cachorro;

Ignorando o artigo "O", teríamos a seguinte matriz *Document-feature*, ou o seguinte BOW (*Bag-of-words*):


<div align="center">

| Documento | Cachorro | ama | Osso |
|-----------|------|-------|----------|
| Doc1      | 1    | 1     | 1        |
| Doc2      | 1    | 1     | 1        |

</div>

Essa matriz é chamada assim por que cada linha (ou observação) contém um documento/sentença/frase e cada coluna contém a palavra em consideração. Por exemplo, a coluna "Cachorro" mostra quantas vezes o termo Cachorro aparece em cada documento. Neste caso, os dois documentos possuem as mesmas palavras, e seriam tratados como "iguais". Isso se deve ao fato de que, nesse tipo de representação, a ordem das palavras e o contexto não são considerados; por isso, embora as duas frases expressem relações diferentes, a Bag-of-Words as representa de forma idêntica, pois contêm as mesmas palavras com as mesmas contagens. 

"Tokenizar", ou transformar em *tokens* faz com que cada documento (ou observação na matriz *Document-Feature*) será quebrado em suas partes individuais, as palavras. Esse é o primeiro passo para a criação do saco de palavras (BOW). O exemplo mais comum é o do tokenizar em n-grams de 1. A frase "Diga não à vacinação obrigatória!" pode ser quebrada da seguinte forma:


<div align="center">

["Diga","não","à","vacinação","obrigatória", "!"]

</div>


Seguindo a mesma frase e a mesma etapa de tokenização em palavras, podemos formar bi-grams, que são pares de palavras consecutivas no texto. Bi-grams ajudam a capturar um pouco de contexto local que o unigram (n=1) não representa, como expressões fixas, negações e relações imediatas entre termos.

<div align="center">

["Diga_não", "não_à", "à_vacinação", "vacinação_obrigatória", "obrigatória_!"]

</div>

Se o objetivo é capturar resistência à vacinação, ou hesitação vacinal, talvez quebrar o texto em bigramas seja mais informativo do que quebrá-lo apenas em suas palavras individuais. Com isso, se dá mais contexto ao modelo de aprendizado de máquina, ao custo de adicionar mais combinações raras que podem não aparecer tanto em seu banco de dados. 💬 "Usar ordens maiores de n-gramas pode aumentar substancialmente o número de tipos únicos, mas pode ajudar nossa análise textual ao reter mais informações" ({cite}`grimmer2022text`, p. 99, tradução nossa). Novamente, isso não é uma escolha trivial: Assim como todos os passos e princípios discutidos nas últimas aulas, a forma de processar e representar numericamente o texto altera substancialmente os resultados. O pesquisador deve tentar sempre estar consciente dessas escolhas e relatá-las aos leitores quando necessário. 

### Processamento Morfológico em PLN

Para desenvolver qualquer aplicação de PLN, é necessário realizar fases/etapas que convencionamos chamar de pré-processamento do texto. No pré-processamento, algumas tarefas usuais são: Segmentação do texto em sentenças (Sentenciação); Separação de Palavras (tokenização); tokenização em subpalavras; normalização de palavras (lematização, radicalização), entre outras. Como as tarefas mais usuais foram discutidas na seção "*Bag-of-words* e o modelo multinomial", não iremos repetir o conteúdo, partindo para os modelos de linguagem.



