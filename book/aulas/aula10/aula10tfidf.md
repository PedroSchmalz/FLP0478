# TF‑IDF

Quando falamos de representação de texto em machine learning, uma das técnicas mais clássicas e ainda muito utilizadas é o **TF-IDF** (*Term Frequency-Inverse Document Frequency*). A ideia por trás dele é bem intuitiva: nem todas as palavras em um documento têm a mesma importância. Algumas palavras aparecem muitas vezes porque são realmente importantes para aquele documento específico, enquanto outras aparecem muito simplesmente porque são palavras comuns da língua, como "o", "a", "de", "que". O TF-IDF tenta capturar essa diferença, dando mais peso para palavras que são frequentes no documento mas raras na coleção toda.

Imagine que você tem uma coleção de notícias sobre esportes. Se em uma notícia específica a palavra "Pelé" aparece 10 vezes, isso provavelmente significa que essa notícia é sobre ele. Mas se a palavra "futebol" também aparece 10 vezes, ela não necessariamente é tão informativa, porque "futebol" provavelmente aparece em quase todas as notícias dessa coleção. O TF-IDF faz exatamente esse tipo de distinção: aumenta o peso de "Pelé" (rara no corpus, mas frequente naquele documento) e diminui o peso de "futebol" (frequente em todo o corpus).

## Como funciona o TF-IDF?

O TF-IDF é o produto de dois componentes: a **frequência do termo** (TF) e a **frequência inversa de documento** (IDF). Vamos entender cada um deles.

### Frequência do termo (TF)

A frequência do termo simplesmente conta quantas vezes uma palavra aparece em um documento. A forma mais básica é usar a contagem bruta:

$$
\text{tf}_{t,d} = \text{número de vezes que o termo } t \text{ aparece no documento } d
$$

Mas aqui tem um problema: se uma palavra aparece 20 vezes em um documento, ela realmente é 20 vezes mais importante do que uma palavra que aparece apenas uma vez? Provavelmente não. Por isso, é comum usar uma **versão logarítmica** da frequência, que cresce mais devagar:

$$
\text{tf}'_{t,d} = 
\begin{cases}
1 + \log(\text{tf}_{t,d}) & \text{se } \text{tf}_{t,d} > 0,\\
0 & \text{caso contrário.}
\end{cases}
$$

Com essa transformação, a diferença entre 1 e 2 ocorrências é grande, mas a diferença entre 20 e 21 ocorrências é bem pequena, o que faz mais sentido intuitivamente.

### Frequência inversa de documento (IDF)

Agora vem a parte que torna o TF-IDF especial. O IDF mede o quão raro é um termo em toda a coleção de documentos. Se temos $N$ documentos no total e um termo $t$ aparece em $\text{df}_t$ deles, calculamos:

$$
\text{idf}_t = \log\left(\frac{N}{\text{df}_t}\right)
$$

Repare que se um termo aparece em **muitos** documentos, o $\text{df}_t$ fica grande, e o IDF fica pequeno (pode até ser próximo de zero). Se um termo aparece em **poucos** documentos, o IDF fica alto. Na prática, muitas implementações usam uma versão suavizada para evitar divisão por zero:

$$
\text{idf}_t = \log\left(1 + \frac{N}{1 + \text{df}_t}\right)
$$

### Juntando tudo: TF-IDF

O peso TF-IDF de um termo $t$ no documento $d$ é simplesmente o produto dos dois:

$$
\text{tfidf}_{t,d} = \text{tf}'_{t,d} \times \text{idf}_t
$$

Opcionalmente, é comum normalizar o vetor resultante (por exemplo, usando a norma L2) para que documentos de tamanhos diferentes fiquem comparáveis. Sem normalização, documentos mais longos teriam vetores com valores maiores simplesmente por serem mais longos, não necessariamente por serem mais relevantes.

## E os n-gramas?

Até agora falamos de palavras individuais (unigrams), mas o TF-IDF funciona igualmente bem com **n-gramas**, que são sequências de $n$ palavras consecutivas. Por exemplo, se usarmos bigrams, além de contar "machine" e "learning" separadamente, também contaremos "machine learning" como um termo único. Isso ajuda a capturar expressões compostas, negações ("não gostei"), locuções ("New York"), e outros fenômenos linguísticos que dependem da ordem das palavras.

Usar n-gramas tem um custo: o vocabulário cresce exponencialmente. Com unigrams, se temos 10.000 palavras únicas, temos 10.000 dimensões. Com bigrams, podemos facilmente ter centenas de milhares de dimensões, deixando os vetores muito mais esparsos. É um *trade-off* entre capturar mais contexto e lidar com maior complexidade computacional.

## Implementando na prática

Vamos ver como usar TF-IDF em Python com `scikit-learn`. O código é bem direto:

```python

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Seus documentos e rótulos

documents = [
"O gato comeu o peixe",
"O cão late à noite",
"Peixe é alimento saudável",
\# ...
]
labels =[^10]

# Dividir em treino e teste

X_train, X_test, y_train, y_test = train_test_split(
documents, labels, test_size=0.2, random_state=42
)

# Criar o vetorizador TF-IDF

vectorizer = TfidfVectorizer(
max_df=0.8,         \# ignora termos que aparecem em >80% dos docs
min_df=2,           \# ignora termos que aparecem em <2 docs
ngram_range=(1, 2), \# usa unigrams e bigrams
sublinear_tf=True,  \# usa 1 + log(tf)
norm='l2'           \# normaliza os vetores
)

# Transformar os textos em vetores TF-IDF

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Treinar um classificador (SVM, por exemplo)

svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train_tfidf, y_train)

print("Acurácia:", svm.score(X_test_tfidf, y_test))

```

Os parâmetros `max_df` e `min_df` são importantes na prática. O `max_df=0.8` remove automaticamente palavras que aparecem em mais de 80% dos documentos — são as "stop words" naturais da sua coleção, palavras tão comuns que não ajudam a discriminar documentos. O `min_df=2` remove palavras muito raras que podem ser erros de digitação ou ruído. Ajustar esses parâmetros faz diferença nos resultados.

## TF-IDF e SVMs: uma boa combinação

O TF-IDF é especialmente popular quando combinado com **Support Vector Machines (SVMs)** para classificação de texto. SVMs funcionam bem em espaços de alta dimensionalidade, que é exatamente o que temos com TF-IDF — vetores com milhares de dimensões, a maioria delas zero (esparsos). A representação TF-IDF também é compatível com a forma como SVMs calculam distâncias e margens entre classes, usando a similaridade de cosseno (que é equivalente ao produto escalar de vetores normalizados).

Na prática, você vai gerar os vetores TF-IDF dos seus textos, treinar o SVM com esses vetores como features, e depois usar o modelo treinado para classificar novos textos. É uma pipeline clássica e funciona surpreendentemente bem para muitos problemas de classificação de texto.

## Vantagens e limitações

O TF-IDF tem várias vantagens práticas. Primeiro, é **simples de entender e implementar** — não precisa de redes neurais nem GPUs. Segundo, é **rápido** de treinar e usar, mesmo com conjuntos de dados grandes. Terceiro, os resultados são **interpretáveis**: você pode olhar para os pesos TF-IDF e entender exatamente quais palavras o modelo considera importantes para cada documento. Isso é valioso em aplicações onde precisamos explicar as decisões do modelo.

Mas também tem limitações. A maior delas é que TF-IDF segue o modelo **bag-of-words**: ignora completamente a ordem das palavras (exceto quando usamos n-gramas, que capturam ordem local). Os documentos "Maria é mais rápida que João" e "João é mais rápido que Maria" são idênticos na representação TF-IDF de unigrams, mesmo tendo significados opostos. Além disso, o TF-IDF não captura similaridade semântica — "carro" e "automóvel" são tratados como dimensões totalmente diferentes, mesmo sendo sinônimos.

Outra questão é a **alta dimensionalidade**. Quando usamos n-gramas, especialmente trigrams ou quadrigrams, o vocabulário explode e os vetores ficam extremamente esparsos. Isso aumenta o custo de memória e pode levar a overfitting se o conjunto de treinamento for pequeno.

## Alternativas mais modernas

Hoje em dia temos alternativas mais sofisticadas que capturam semântica de forma mais profunda. **Word embeddings** como Word2Vec, GloVe e FastText representam palavras como vetores densos de 100-300 dimensões, onde palavras semanticamente similares ficam próximas no espaço vetorial. Modelos de linguagem baseados em **Transformers** como BERT vão ainda mais longe, gerando embeddings contextualizados que variam dependendo do contexto da palavra na frase.

Esses modelos modernos geralmente superam TF-IDF em tarefas complexas de NLP como análise de sentimento, perguntas e respostas, e compreensão de texto. Mas eles também exigem muito mais recursos computacionais e dados de treinamento. Para muitas aplicações práticas, especialmente com conjuntos de dados pequenos ou quando interpretabilidade é importante, TF-IDF continua sendo uma escolha sólida.

## Quando usar TF-IDF?

TF-IDF ainda é uma excelente escolha quando você tem um conjunto de dados pequeno ou moderado, quando precisa de uma solução rápida e interpretável, ou quando recursos computacionais são limitados. É também uma ótima **baseline** — antes de investir tempo em modelos complexos como BERT, vale a pena treinar um classificador simples com TF-IDF para ter um ponto de comparação. Muitas vezes você vai se surpreender com o quão bem ele funciona.

Em aplicações de recuperação de informação e busca textual, TF-IDF continua sendo amplamente usado. Buscadores como o Elasticsearch ainda usam variantes de TF-IDF (como BM25) como parte de seus algoritmos de ranking. E em cenários onde você precisa explicar por que um documento foi considerado relevante, poder apontar para palavras específicas com altos pesos TF-IDF é muito mais fácil do que tentar explicar os pesos de uma rede neural de 12 camadas.

## Conclusão

O TF-IDF é uma daquelas técnicas que, apesar de existir há décadas, continua relevante porque resolve um problema real de forma eficiente e interpretável. Ele balanceia frequência local (quão importante é um termo dentro do documento) com frequência global (quão discriminativo é o termo na coleção toda), e essa ideia simples funciona muito bem na prática. Quando combinado com n-gramas, consegue capturar um pouco de contexto local, tornando-se ainda mais útil. E quando você precisa de uma representação de texto rápida, interpretável e eficaz, especialmente para usar com SVMs ou outros classificadores lineares, TF-IDF é uma escolha difícil de bater. Veremos no laboratório como usar o TF-IDF na classificação de texto.



## Leituras

MANNING, Christopher D.; RAGHAVAN, Prabhakar; SCHÜTZE, Hinrich. {cite}`manning2008introduction`. Introduction to Information Retrieval. Capítulo 6. Cambridge: Cambridge University Press, 2008. 482 p. ISBN 978-0-521-86571-5.

ANTIĆ, Zhenya; CHAKRAVARTY, Saurabh. {cite}`antic2024python`. Python Natural Language Processing Cookbook: over 60 recipes for building powerful NLP solutions using Python and LLM libraries. 2. ed. Birmingham: Packt Publishing, 2024. ISBN 978-1-80324-574-4.

