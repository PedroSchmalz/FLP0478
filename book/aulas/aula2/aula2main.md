
# Seleção e Representação


````{margin}
```{note}
É possível que o pesquisador tenha interesse em utilizar um córpus criado por terceiros. Nesses casos, é importante analisar se esse material é adequado para responder à sua pergunta de pesquisa. Assim como em levantamentos (surveys), o córpus foi coletado e anotado com objetivos específicos em mente. Por isso, é necessário compreender suas limitações e até onde ele pode ser utilizado em seu próprio trabalho.
```
````

Como foi dito anteriormente, novas oportunidades e ferramentas de análise de texto estão disponíveis para os pesquisadores das humanidades e ciências sociais. O volume e a velocidade em que novos dados textuais e documentos são disponibilizados aumenta diariamente. Agências governamentais podem utilizar documentos e relatórios para a melhoria dos serviços públicas; Cientistas políticos podem mensurar mudanças de tópicos de debate entre os políticos, como certos tópicos flutuam ao longo do tempo nas redes sociais, etc. As possibilidades de pesquisa são inúmeras, assim como as de erros. Muitas são as dificuldades e desafios para os pesquisadores no momento de pensar em possíveis perguntas e desenhos de pesquisa, além de como operacionalizar esses dados da forma mais adequada para o seu projeto. Portanto, o pesquisador deve pensar com muito cuidado em qual sua pergunta de pesquisa, população e quantidades de interesse, o universo de documentos, e se esses documentos podem responder a sua pergunta e/ou se refletem seus interesses de pesquisa.


Na nossa disciplina, focaremos em como fazer a pesquisa utilizando métodos de aprendizado de máquina supervisionado. Para que uma aplicação desse tipo seja possível é necessário um córpus anotado. Córpus (Corpora, no plural) é um conjunto de textos coletados de forma sistemática, representativo de uma língua ou variedade linguística específica, utilizado para alimentar, treinar, testar ou validar modelos e técnicas de análise automática de linguagem humana. Portanto, antes mesmo de pensar em qual técnica de aprendizado utilizar, o pesquisador precisa refletir profundamente sobre como irá construir o seu córpus e de qual universo serão retirados os documentos textuais. A qualidade, a representatividade e a precisão das anotações têm impacto direto na performance e na generalização dos modelos treinados, assim como das inferências e resultados da pesquisa.


```{admonition} 💬 Com a palavra, os autores:
:class: quote
"Os dados textuais refletem interações sociais, transações econômicas e processos políticos. Para utilizar essa riqueza de informações para formular e responder perguntas interessantes, o pesquisador deve primeiro selecionar cuidadosamente o corpus de interesse e, em seguida, representar esses documentos de forma numérica. Embora seja mais complicado devido à enorme quantidade de informações armazenadas nos textos, as decisões sobre como coletar e representar numericamente textos são semelhantes às decisões que pesquisadores tomam para representar numericamente outras variáveis de interesse nas ciências sociais. Coletar um corpus é análogo a identificar uma amostra de uma população de interesse."
({cite}`grimmer2022text`, p. 72, tradução nossa)
```

## Princípios de Seleção e Representação


````{margin}
```{note}
"Representação numérica do texto" refere-se à conversão do conteúdo textual (palavras, frases, parágrafos) em formatos quantitativos que possam ser manipulados e analisados por ferramentas estatísticas ou computacionais. Isso inclui, por exemplo, transformar textos em vetores numéricos por meio de métodos como contagem de palavras (Bag-of-Words), frequências de termos (TF-IDF), embeddings (Word2Vec, GloVe, BERT), ou outras formas que capturam características do texto em números. Veremos as diversas formas de representação do texto a partir da aula 06 do curso. Esses números permitem que algoritmos identifiquem padrões, similaridades, tópicos ou outras relações estruturais nos dados textuais, tornando-os operacionalizáveis para análises sociais, mesmo em grande escala. Assim, a "representação numérica do texto" é o processo de traduzir a informação qualitativa do texto para variáveis quantitativas que podem ser estudadas e interpretadas sistematicamente.
```
````

O capítulo 3 do livro de Grimmer et al. aprofunda a importância de selecionar e representar textos de modo criterioso para transformá-los em dados úteis para pesquisa social. Nem toda informação textual é relevante para toda pergunta de pesquisa. O pesquisador deve ir para além do acúmulo de documentos e textos, focalizando na qualidade do que é coletado. Como o pesquisador decide quais textos são relevantes? Devido à relativa novidade desse tipo de métodos nas ciências sociais, há um descompasso entre a literatura e teoria clássica de muitas das ciências sociais e os métodos utilizados no Processamento da Língua Natural. Nem sempre será possível se guiar pela literatura para a operacionalização de variáveis e representação numérica do texto. Portanto, o pesquisador precisará de muita validação, dentro e fora da construção do córpus. Para guiar os pesquisadores nessa empreitada, Grimmer et al. propõem quatro princípios que devem guiá-los na pesquisa em PLN.


### Princípio 1. Construção do Corpus Guiada pela Pergunta

O primeiro princípio, já mencionado de forma indireta ao longo do texto, é que a pergunta de pesquisa deve orientar a construção do córpus — e não o contrário. Mesmo no exemplo de King, Pan e Roberts (2013), em que a pergunta emergiu de modo aparentemente ‘acidental’, a elaboração do primeiro córpus foi guiada por uma questão de pesquisa delineada à luz de uma literatura científica específica. Isso não implica um retorno a um modelo estritamente dedutivo; significa, antes, que a construção do córpus deve ser conduzida por uma boa pergunta de pesquisa, sustentada, sempre que possível, por referências bibliográficas sólidas. Os pontos centrais desse princípio são:

- **Definir a pergunta de pesquisa**: formular uma questão clara, específica e operacionalizável, explicitando o fenômeno, a unidade de análise, o recorte temporal e/ou espacial, e o resultado esperado (ou a hipótese a testar).
- **Delimitar o universo e a amostra**: especificar com precisão o universo relevante (por exemplo, mídia nacional vs. regional; órgãos oficiais vs. redes sociais) e adotar estratégias de amostragem coerentes com a pergunta (probabilística, teórica, intencional, por cotas), justificando as escolhas.
- **Identificar as quantidades de interesse**: deixar explícito quais medidas serão estimadas e como serão operacionalizadas (por exemplo, proporção/volume de tópicos discutidos por políticos, supervisionado ou não; posicionamento de jornais sobre tópicos/figuras em escalas definidas; polaridade/valência do sentimento em discussões online; intensidade, saliência, centralidade, diversidade). Indicar a unidade de medida, o método de estimação e potenciais vieses.
- **Avaliar o uso de um córpus existente**: ao reutilizar um córpus de terceiros, verificar a adequação à pergunta (cobertura temporal, fontes, idiomas, gêneros), a validade das variáveis/rotulagens, a qualidade e documentação, permissões e limitações; checar se as medidas desejadas são mensuráveis com aquele material ou se será necessário complementar/ajustar o córpus.


### 2. Não Existe Corpus Neutro/Sem valores

A construção de um córpus nunca é neutra: envolve escolhas sobre fontes, períodos, gêneros, critérios de inclusão/exclusão e formas de representação que refletem pressupostos teóricos, limitações práticas e valores do pesquisador. Essas decisões têm implicações metodológicas e éticas — especialmente para quem é incluído/excluído, como variáveis são medidas/rotuladas e que inferências se tornam possíveis. À luz dos princípios e alertas de seleção e representação, convém atentar para:

- **Preocupações éticas e de LGPD**: garantir base legal, finalidade específica e minimização de dados; adotar anonimização/pseudonimização quando cabível; considerar riscos de reidentificação, sobretudo quando textos são vinculados a outros dados sensíveis.

- **Consentimento, publicidade e “integridade contextual”**: mesmo conteúdos “públicos” podem ter normas contextuais de uso e expectativas de privacidade distintas; avalie a adequação do uso de textos de redes sociais, fóruns fechados ou listas restritas em função de contexto e audiência previstos originalmente.

- **Diferenças de recursos e incentivos entre grupos**: textos refletem mais os grupos com capacidade de produzir, registrar e preservar documentos; ausência de registros não é aleatória em termos socioeconômicos, temporais ou institucionais. Outros viéses podem surgir de ocultação intencional de documentos, censura, uso inadequado de palavras chaves, métodos de coleta, etc.

- **Linguagem prejudicial e danos potenciais**: reconhecer e tratar a presença de conteúdo nocivo (ódio, estereótipos), ponderando efeitos de sua inclusão em modelos e de sua divulgação pública, sobretudo sobre grupos vulneráveis.


### 3. Não Há Uma Única Representação Correta

A representação deve ser guiada pela pergunta e pela quantidade de interesse. Prefira o mais simples que capture o fenômeno e valide. Não existirá uma única forma de representar o texto, nem mesmo dentro da mesma pergunta. Aqui é necessário validação, acompanhamento por outros pesquisadores, codificadores trabalhando em pares (ou trios). Tudo isso assegurará que o córpus possua validade. 

- **Definir o objetivo de mensuração**: explicitar o que se quer medir no texto (p. ex., tema, posição, tom, estilo, saliência) e como isso pode se manifestar linguisticamente.

- **Escolher a representação mínima suficiente**: optar por features tão simples quanto possível (palavras, n-gramas, dicionários, bag-of-words) antes de adotar modelos mais complexos (embeddings, contextuais, estruturas sintáticas). Em certos casos, se deseja somente contar a frequência de certas palavras e não o contexto em que elas estão. Nesses casos, o pesquisador pode optar por modelos mais leves com representações simples.

- **Alinhar unidade e contexto**: decidir o nível de análise (token, sentença, documento) e se ordem, sintaxe ou metadados são necessários ao objetivo.

- **Tornar operacional e testável**: especificar métricas, procedimentos de pré-processamento e critérios de sucesso; comparar alternativas e manter a que melhor atende à pergunta com validação externa. Registrar a concordância entre anotadores (Alpha de Krippendorf, etc.), e registrar também as regras de anotação.


### 4. Validação é Essencial

A garantia de que a representação e as medidas funcionam vem de validação sistemática, alinhada ao objetivo (medir, prever, inferir causalmente). Essa validação deve ser feita na construção do córpus e no treinamento dos modelos de aprendizado de máquina supervisionado. Alguns pontos importantes são:

- **Definir o critério de sucesso**: especificar o que significa “funcionar” para a tarefa (correlação com gold standard, acurácia preditiva, replicação de efeitos causais).

- **Usar validações externas**: comparar medidas com codificação humana, dados administrativos ou fatos conhecidos; quando possível, fazer auditorias cegas.

- **Testes de sensibilidade e robustez**: verificar robustez a escolhas de pré-processamento, features, amostragem e janelas temporais; remover componentes e checar impacto.

- **Particionar e replicar**: separar treino/validação/teste, usar validação cruzada e replicar resultados em períodos, fontes ou populações distintas.

- **Revisitar desenho e representação**: se a validação falhar, ajustar pergunta, corpus, features e modelo; documentar limitações e escolhas que afetam a inferência.

---

## Capítulo 4 — Seleção de Documentos


### 1. População e quantidades de interesse: ancoragem do desenho

Definir com precisão a população de interesse (quem, onde, quando) e as quantidades de interesse (o que será medido) é o passo que orienta todas as decisões subsequentes de coleta e filtragem. Um mesmo acervo pode ser excelente para uma pergunta (p.ex., estratégia de candidatos) e inadequado para outra (p.ex., opinião pública), pois o corpus reflete os produtores dos textos, não “a sociedade” em geral. Sem essa ancoragem, a amostra tende a deslizar para conveniência ou disponibilidade, comprometendo validade externa e interpretação.

### 2. “Found data”: potencial e limites

Grande parte dos corpora são “dados achados”, produzidos por agentes com recursos e incentivos próprios, e divulgados sob regras institucionais e tecnológicas que não visam à pesquisa. Isso cria lacunas estruturais (ausência de grupos sem capacidade de registro/preservação), mudanças de regime de coleta/acesso ao longo do tempo e assimetrias por plataforma. Usar “found data” é legítimo, mas requer: explicitar por que esse acervo responde à pergunta, o que ele não captura e como isso afeta o que pode ser inferido.

### 3. Quatro vieses recorrentes de seleção

- **Viés de recursos**: textos representam desproporcionalmente quem consegue produzir, armazenar e tornar acessíveis documentos; arquivos históricos, bases editoriais e acervos como Ngrams tendem a super-representar certos gêneros/áreas por custos e infraestrutura. Eventos/vozes sem presença de mídia ou capacidade de registro desaparecem do corpus, afetando séries temporais e comparações.
- **Viés de incentivos**: atores estrategicamente produzem, ocultam, removem ou deslocam conversas para canais menos visíveis (censura, autopromoção, telefonemas em vez de e‑mails quando há transparência). A disponibilidade de transcrições pode alterar o próprio comportamento (mais formalização, menos franqueza), o que precisa ser entendido e, quando possível, verificado qualitativamente.
- **Viés de meio**: o suporte molda conteúdo e forma (limites de caracteres, multimodalidade, efemeridade, feeds personalizados), e diferentes plataformas/línguas sofrem constrangimentos distintos. Mudanças de produto (p.ex., novas features, políticas) reconfiguram o que é dito e como é medido, quebrando a comparabilidade longitudinal se não forem controladas.
- **Viés de recuperação**: critérios e ferramentas de busca (palavras‑chave, APIs, filtros) introduzem inclusões/omissões sistemáticas; termos lembrados tendem a enviesar a cobertura temática e o tipo de documento retornado. Interfaces opacas (amostragem, relevância, case sensitivity) podem alterar a população efetivamente observada sem aviso.


Abaixo, dois exemplos de viés de recuperação que podem surgir dependendo das palavras-chaves que o pesquisador estabelecer para pesquisar opiniões sobre Bolsonaro ou Lula durante a pandemia.

```{admonition} 🐦 Tweet
:class: tweet
**@usuario_exemplo**: Se fosse o *Nine* não ia ter nenhuma vacina por que ele ia roubar tudo. 
12:34 · 10 ago. 2021
```

Se o pesquisador só tivesse como palavra-chave para o presidente Lula o seu nome (ou nome inteiro), poderia perder um tweet nesse formato, que o menciona através de um apelido pejorativo. O seguinte exemplo era algo mais comum em nosso banco de dados.



```{admonition} 🐦 Tweet
:class: tweet
**@usuario_exemplo**: O genocida não comprou vacinas, e ainda queria ganhar dinheiro em cima delas. Impeachment agora! 
12:34 · 12 ago. 2021
```

Novamente, por estar se referenciando ao ex-presidente Bolsonaro por meio de um outro nome, o pesquisador poderia perder certos *tweets*, gerando um viés na amostra. Isso é especialmente sensível quando queremos mensurar a opinião pública sobre determinados candidatos. O pesquisador não deve ter somente um contato amplo com a literatura, mas também deve explorar bem os seus dados. Isso garante que ele esteja no controle da pesquisa, não o método.


### 4. Estratégias de mitigação e boas práticas

- Planejar com o alvo em mente: colecionar após definir população/quantidades de interesse e testar se o corpus permite estimá-las com qualidade suficiente.
- Ampliar e calibrar fontes: combinar acervos com perfis distintos (institucionais, jornalísticos, redes sociais, arquivos) para reduzir lacunas de recursos e incentivos; monitorar mudanças de plataforma e política de dados.
- Revisar consultas e pipelines: construir e iterar listas de termos/estratégias de busca, testar consultas alternativas, amostrar manualmente falsos positivos/negativos e documentar diferenças entre APIs.
- Amostrar com desenho explícito: quando possível, aplicar amostragem estratificada/temporal ou reponderar para corrigir desequilíbrios conhecidos (volume tardio, fontes dominantes).
- Validar externamente: cruzar medidas com codificação humana, dados administrativos, fatos conhecidos ou séries paralelas; analisar sensibilidade a janelas, filtros e critérios de inclusão.
- Documentar o escopo de inferência: declarar claramente a que população os resultados se aplicam, o que não pode ser generalizado e por quê; registrar versões, datas, consultas, filtros e limitações conhecidas.


Para permitir avaliação e replicação, registrar e disponibilizar: critérios de inclusão/exclusão, fontes e versões, janelas temporais, consultas e parâmetros, mudanças de política/plataforma, taxas de erro estimadas na recuperação e impacto esperado dos vieses identificados sobre as conclusões.


## Conclusão

Construir e representar um corpus de textos é um processo fundamental, cheio de decisões estratégicas e éticas. A chave está em alinhar cada etapa à pergunta de pesquisa, ser transparente sobre limitações e validar constantemente suas escolhas. Com isso, é possível transformar textos em dados realmente úteis para entender fenômenos sociais complexos.


