# Mensuração e Aprendizado Supervisionado

## Princípios da Mensuração

O principal objetivo ao se mensurar algo é fundamentar um conceito dentro de nossa hipótese ou teoria de maneira a facilitar a quantificação. Nenhuma mensuração pode estar divorciada de uma boa compreensão teórica da coisa a ser observada. Os seguintes passos são ressaltados por Grimmer et al. para uma boa mensuração:

1. Definir a conceituação que queremos medir;
2. Localizar uma fonte de dados que contém implicações do conceito identificado;
3. Gerar uma forma de traduzir os dados em representação latente;
4. Rotular a representação e conectá-la com o conceito identificado;
5. Validar a mensuração obtida.

### Do conceito à mensuração

Como vimos antes, descoberta é o processo de propor um entendimento do mundo, desarranjar esse entendimento com dados, revisando os resultados até que vejamos o mundo de uma maneira diferente. Quando esse processo se assenta, aí é que começa a mensuração. Como ressaltado algumas vezes, não há um único modo correto e verdadeiro de se mensurar. Portanto, é necessário apresentar os principais caminhos e formas de atingir uma boa mensuração.

### Do que é feita uma boa mensuração?

O objetivo de uma boa mensuração é a de atingir uma simplificação de um fenômeno do mundo real que permita uma descrição precisa do processo ou que permita testar as implicações observáveis de uma teoria. Grimmer et al. estabelecem, novamente, alguns princípios que devem guiar o pesquisador:

#### Princípio 1 - Mensurações devem ter um objetivo claro.

A medida/mensuração deve ter **escopo** e **objetivos** claros, garantindo que foi usada de forma correta na análise inicial do pesquisador, e que será utilizada de forma correta por outros pesquisadores, na medida do possível.

#### Princípio 2 - O material fonte deve ser sempre identificado e, se possível, tornado público.

Se o pesquisador obteve os documentos de um jornal, deve explicar ao longo do texto qual jornal, como obteve, e como outros pesquisadores podem replicar o que ele fez. Se os dados obtidos não tiverem questões de LGPD, éticas, ou de anonimização, o pesquisador deve tornar esses dados públicos, seja por meio de *Github*, *Harvard Dataverses* ou quaisquer outros meios de disponibilização dos dados para pesquisadores interessados. Isso garantirá uma clareza do que foi feito mesmo que os leitores não tenham interesse em replicar a pesquisa.

#### Princípio 3 - O processo de codificação deve ser possível de explicar e de reproduzir.

Assim como os dados devem ser públicos, também deve estar claro como foram rotulados e quais regras foram seguidas. No nosso projeto, disponibilizamos um codebook extenso com coleta e descrição dos dados, como os anotadores foram treinados e quais regras seguiram, e os resultados de concordância entre os anotadores (Alpha de Krippendorf, porcentagem de concordância,etc.)

#### Princípio 4 - A métrica deve ser validada.

A validação é o processo de estabelecer para os leitores que a mensuração se conecta bem com o conceito teórico que procura mensurar. Isso se garante na clareza das regras de anotação (junto com as métricas de concordância). Mas também existe a validação dentro do aprendizado de máquina, que consiste em testar o quanto o que o modelo aprendeu com o banco de treinamento pode ser generalizado para outras amostras. Ambas as validações são necessárias e importantes.

#### Princípio 5 - Limitações devem ser exploradas, documentadas e comunicadas.

O quinto princípio é bem autoexplicativo: Se o pesquisador sabe de problemas, ou não tem tanta confiança em certas métricas ou regras, isso *deve* estar claro no texto. A pesquisa não é o local de esconder suas falhas, muito pelo contrário. Se o pesquisador está ciente de problemas na anotação e codificação, deve deixar claro para os leitores. Se não está ciente, o problema é muito mais grave.

## Contagem de Palavras

Antes da criação de categorias ou tópicos de documentos, uma forma mais simples de trabalhar com texto era a de contar a frequência de certas palavras ao longo de todo o documento. Ainda que contagem de palavras seja uma forma mais direta e clara de trabalhar com o texto, ela pode não ser tão útil para conceitos mais complexos. No entanto, pode ser extretamente efetiva em alguns casos. Grimmer et al. citam o exemplo de Duneier (2016, {cite}`duneier2016ghetto`) com a exploração de como a palavra *Ghetto* muda de significado e utilização antes e depois do nazismo.


```{figure} ../aula3/images/ghetto.png
---
width: 100%
name: duneierghetto
align: center
---
 Gráfico 1 de Duneier (2016) mostrando a proporção do uso da palavra "Ghetto" ao longo dos anos. Dados coletados do Google *Ngrams*

```

Como mostra a {numref}`Figura {number} <duneierghetto>`, o significado associado à palavra "Ghetto" mudou profundamente após a Segunda Guerra Mundial. Inicialmente, o termo era usado principalmente para descrever áreas de segregação de judeus na Europa. No pós-guerra, influenciado pela experiência nazista de discriminação e confinamento, o uso da palavra passou a abranger também os guetos de negros americanos, refletindo processos de segregação racial nos Estados Unidos. Essa mudança revela como eventos históricos podem transformar o sentido e o contexto de termos utilizados pela sociedade.


```{admonition} 💬 Com a palavra, os autores:
:class: quote
"Em outras palavras, Duneier sustenta não apenas que a associação da palavra “gueto” mudou — algo claramente visível no gráfico —, mas também que a introdução de “gueto” como termo para descrever a segregação residencial negra esteve ligada ao uso nazista da palavra. O livro de Duneier é um exemplo poderoso de como uma análise quantitativa simples pode complementar, de forma útil, uma sociologia histórica detalhada e sutil."
({cite}`grimmer2022text`, p. 286, tradução nossa)
```

Grimmer et al. reforçam, e nós assinamos embaixo, de que bons trabalhos de análise de texto não precisam sempre utilizar os métodos mais novos e refinados. Bons trabalhos podem surgir de uma análise simples, desde que bem embasada com a teoria, literatura, e processos históricos.

### Métodos de Dicionário

Dicionários são a generalização do método de palavras chaves. Eles usam a frequência média das palavras chaves que aparecem em um texto para classificar os documentos em categorias ou mensurar quanto esses documentos pertencem à categorias particulares. Como um pesquisador pode aplicar esses métodos?

1. Devem identificar um conjunto de palavras que separam categorias e mensurar a frequência delas em um conjunto de documentos;
2. Criar um dicionário ou utilizar um dicionário existente.

#### Limitações

Os métodos de dicionário já não são utilizados de maneira tão ampla, e alguns estudos mostram que costumam ter performance pior do que modelos de aprendizado supervisionado simples ({cite}`barbera2020automated`). Aqui estão alguns problemas:

1. Os valores atribuídos às palavras devem se alinhar à como as palavras são usadas naquele contexto;
2. Devem ser usados de forma cautelosa e com bastante validação explícita.
3. Não produzem métricas com propriedades particulares;
4. Performam pior do que modelos simples de aprendizado de máquina supervisionado (e.g. regressão logística, multinomial, *Decision Trees*, etc.)

## Conclusão

A mensuração em análise de texto exige rigor conceitual, transparência e validação constante. Como vimos, bons resultados dependem de uma clara definição dos objetivos, da identificação e disponibilização dos dados, de processos de codificação reprodutíveis e da comunicação aberta sobre limitações. Métodos simples, como a contagem de palavras, podem ser extremamente úteis para revelar mudanças históricas no significado de termos, desde que bem fundamentados teoricamente. Por outro lado, métodos de dicionário oferecem uma abordagem prática para classificação de textos, mas apresentam limitações e exigem validação cuidadosa. Independentemente do método escolhido, o compromisso com a clareza, a replicabilidade e a honestidade sobre as restrições do processo é fundamental para garantir a qualidade e a credibilidade da pesquisa em PLN (e no geral).

