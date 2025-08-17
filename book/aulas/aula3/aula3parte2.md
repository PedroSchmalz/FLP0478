
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