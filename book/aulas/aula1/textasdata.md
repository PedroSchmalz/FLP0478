# O Texto como um Dado

````{margin}
```{admonition} 💬 Com a palavra, os autores
:class: quote
"Os pesquisadores podem aprender muito sobre o comportamento humano a partir de textos, mas, para isso, é necessário um engajamento com o contexto em que esses textos são produzidos. Uma compreensão profunda do contexto das ciências sociais permitirá que os pesquisadores façam perguntas mais importantes e impactantes, garantam que as medidas extraídas sejam válidas e estejam mais atentos às implicações práticas e éticas de seu trabalho." ({cite}`grimmer2022text`, p. 25, Tradução nossa).
```
````

Nos últimos anos, a explosão de dados digitais e o avanço das capacidades computacionais transformaram a análise de texto em uma ferramenta poderosa para diferentes áreas — da ciência política às humanidades digitais e à indústria. No curso, focaremos em aplicações na ciência política e, mais especificamente, na nossa aplicação para publicações de políticos brasileiros na rede social X/Twitter. No entanto, cresce cada vez mais as aplicações nas humanidades e ciências sociais ([Exemplo](https://aclanthology.org/volumes/2024.nlp4dh-1/)). 


Textos são registros riquíssimos da atividade humana: é por meio da linguagem que se expressam políticas públicas, manifestações culturais, debates eleitorais e sentimentos. Ainda assim, até pouco tempo atrás, a análise sistemática de grandes volumes de texto era restrita ou inviável. Com o avanço dos métodos de **Aprendizado de Máquina** e técnicas de **Processamento de Linguagem Natural (PLN)**, passou a ser possível transformar textos em dados estruturados e analisá-los para fazer descrever e fazer inferências sobre o comportamento humano. No entanto, ainda há uma escassez de bancos de dados de textos com classificação e supervisão humana para a utilização no aprendizado de máquina, especialmente na Língua Portuguesa. Por isso, é necessário que mais pesquisadores tenham domínio das técnicas de aprendizado supervisionado (e não supervisionado). E também entendam que essas técnicas não substituem a análise atenciosa e dedicada do pesquisador, mas as amplificam, gerando oportunidades de pesquisa e descrição do comportamento humana de forma mais ampla.



## O Paradigma proposto por Grimmer et al. 

O livro *Text as Data: A New Framework for Machine Learning and the Social Sciences*  de Grimmer, Roberts e Stewart (2022) propõe um paradigma (ou *framework*) específico para integrar métodos de PLN e machine learning ao trabalho de pesquisa social. Esse paradigma envolve uma nova forma de ver o processo de produção científico, agora indutivo ao invés de dedutivo. A {numref}`Figura {number} <modeloindut>` mostra como os autores constroem as diferenças entre esses dois modelos.

```{figure} ../aula1/images/image1.png
---
width: 100%
name: modeloindut
align: center
---
 Modelos Dedutivos e Interativos. Fonte: Grimmer et al. (p.41)
```


### Modelo Dedutivo

Os autores dividem o processo de pesquisa nas ciências sociais em dois modelos principais: o dedutivo e o indutivo. O modelo dedutivo, mais tradicional e amplamente disseminado, é caracterizado por um fluxo linear e sequencial de etapas. Nele, a pesquisa se inicia com a formulação de uma pergunta ancorada na literatura e em teorias pré-existentes. A partir dessa base, são elaboradas hipóteses que orientam a coleta e a análise de dados, culminando na apresentação dos resultados em artigos ou livros. Esse modelo pressupõe uma sequência rígida e acíclica, em que cada etapa é concebida para não interferir nas anteriores. Contudo, os autores argumentam que essa visão é ilusória e constitui, na prática, uma “ficção” (Id., p. 40), pois não corresponde à forma como a pesquisa efetivamente se desenvolve nas ciências sociais. Manter essa ficção pode dificultar a compreensão sobre como perguntas e hipóteses foram originalmente construídas, de que maneira a análise de dados impactou suas formulações e como conceitos e problemas foram refinados ao longo de um processo investigativo marcado pela interação constante entre teoria, dados e interpretação.

```{admonition} 💬 Com a palavra, os autores:
:class: quote
"Se o procedimento dedutivo padrão for seguido de forma muito rígida e os dados forem coletados apenas no último momento, os pesquisadores podem perder a oportunidade de refinar seus conceitos, desenvolver novas teorias e avaliar novas hipóteses. Grande parte do aprendizado ocorre durante a análise dos dados. Mesmo quando um projeto de pesquisa começa com uma pergunta clara de interesse, frequentemente termina com um foco substancialmente diferente. Foi isso que aconteceu em um de nossos próprios projetos, uma análise das mídias sociais chinesas conduzida por Gary King, Jennifer Pan e Margaret Roberts (King, Pan e Roberts, 2013)."  
({cite}`grimmer2022text`, p. 39, tradução nossa).
```


### Modelo Indutivo

O modelo indutivo destaca-se por reconhecer que os processos de formulação e desenho de pesquisa, coleta de dados e análise são complementares e interdependentes, ocorrendo de forma cíclica. Nesse modelo, é possível começar com uma pergunta de pesquisa e hipóteses iniciais e, ao longo da análise de dados, reformular essas perguntas ou até mesmo identificar questões completamente novas e mais relevantes. Isso não significa que se deva ignorar a literatura e as teorias existentes, mas sim que é necessário estabelecer uma relação simbiótica entre teoria e prática. Essa abordagem permite articular o que é observado na literatura com os novos conceitos e descobertas emergentes dos dados, promovendo um processo de pesquisa mais dinâmico e adaptativo. Grimmer et al. ({cite}`grimmer2022text`) dividem o modelo em três partes:

- **Descoberta**: Este estágio está intimamente ligado a construção da pergunta de pesquisa e avaliação da literatura. Envolve, sobretudo, teoria, organização dos principais conceitos, e formulação da pergunta e hipóteses de pesquisa. Um passo extremamente importante do desenho de pesquisa e geralmente ignorada no modelo de pesquisa quantitativa tradicional. Entender o papel da descoberta e articulação da teoria é essencial, e pode levar ao desenvolvimento de novas teorias, especialmente se os pesquisadores tiverem o espaço para serem mais sinceros em como operaram nesse passo essencial da pesquisa.
  * >  🗨️ "Decidir o que você quer medir a partir dos dados envolve desenvolver uma conceitualização — uma forma de organizar o mundo — que nos ajude a compreender o complexo mundo em que vivemos. A conceitualização ajudará a simplificar o mundo altamente complexo em que vivemos para estudar um ou dois aspectos específicos dele." ({cite}`grimmer2022text`, p. 43).



- **Mensuração**: O estágio de descoberta está bem conectado com o estágio de mensuração: A análise de textos e a coleta e categorização dos dados pode levar a novas conceitualizações e operacionalizações, ou gerar novas perguntas, como é o caso do KPR ({cite}`king2013how`,2013). Nesse estágio, o pesquisador está interessado em mensurar os conceitos, e coletar os dados que permitam entender indivíduos ou textos, classificando e diminuindo a complexidade destes. Os autores destacam que esse processo, intimamente ligado com a descrição e mensuração, é também deixado de lado. No entanto, se a operacionalização não reflete adequadamente os conceitos, ou se a descrição foi deixada de lado, isso pode ter sérias consequências para a pesquisa e os resultados da inferência.
  * >  🗨️ "A mensuração é o ingrediente essencial para a descrição: um objetivo importante por si só que é frequentemente negligenciado na pesquisa em ciências sociais. Se realizada adequadamente, a descrição fornece resumos valiosos dos dados, que, por sua vez, podem informar teorias, fornecer as medidas necessárias para inferências causais ou caracterizar o estado do mundo." ({cite}`grimmer2022text`, p. 45).

- **Predição/Inferência**: Por fim, os autores destacam um último estágio da pesquisa: a Predição (em aplicações mais usuais do Aprendizado Estatístico) ou a Inferência, entendida aqui como a causal. Nesse estágio, o objetivo é prever valores (contínuos ou classes) ou fazer inferências sobre efeitos e causalidade no mundo a partir dos dados 
  * >  🗨️ "Os pesquisadores podem usar textos para prever valores de preços de ações ou os locais onde é provável que ocorram conflitos políticos. Essas são questões preditivas porque perguntam como as informações disponíveis hoje ajudam a entender o que acontecerá amanhã. Os pesquisadores também podem avaliar o efeito causal de adotar uma abordagem negativa em uma campanha — uma intervenção — na cobertura da mídia sobre a campanha. Ou podem estar interessados em como certos tipos de conteúdo político afetam o engajamento dos usuários em fóruns online." ({cite}`grimmer2022text`, p. 46).


---

## Análise Agnóstica do Texto


Seis principios



```{admonition} 📝 Exercício: Explorando o Ciclo de Descoberta → Mensuração → Inferência
:class: exercise

Com base no exemplo de Catalinac ({cite}`catalinac2016from`,2016) e King, Pand and Roberts ({cite}`king2013how`,2013), reflita sobre como o ciclo de **descoberta → mensuração → inferência** pode ser aplicado em diferentes contextos de pesquisa. Responda às perguntas abaixo:

1. **Definição do Problema**  
   - Identifique um problema de pesquisa relevante na sua área de interesse. Por exemplo, "Como as redes sociais influenciam o debate público sobre mudanças climáticas?".
   
2. **Coleta de Dados**  
   - Que tipo de dados textuais você utilizaria para abordar esse problema? Considere fontes como redes sociais, discursos políticos, artigos de jornal, etc.

3. **Método**  
   - Qual método de análise você aplicaria para identificar padrões ou temas nos textos? Exemplos incluem Latent Dirichlet Allocation (LDA), análise de sentimentos ou classificação supervisionada.

4. **Validação**  
   - Como você validaria os resultados da sua análise? Pense em estratégias como leitura manual de amostras, comparação com eventos conhecidos ou validação cruzada.

5. **Inferência**  
   - Que tipo de inferência você poderia fazer com base nos resultados? Por exemplo, estimar o impacto de uma política pública ou identificar mudanças no discurso político ao longo do tempo.

6. **Reflexão Final**  
   - Como o ciclo iterativo de descoberta e mensuração pode ajudar a refinar suas perguntas de pesquisa e hipóteses iniciais? Considere como os dados podem influenciar o foco do seu estudo.

Após responder às perguntas, discuta suas respostas com um colega ou no grupo de estudos. Reflita sobre como o paradigma proposto por Grimmer et al. pode ser aplicado para enriquecer sua pesquisa.
```


## 🚀 Conclusão

Trabalhar com **text as data** exige:
- Combinar métodos computacionais e teoria social.
- Respeitar a lógica iterativa de descoberta e refinamento conceitual.
- Validação constante, adequada aos objetivos substantivos e não apenas métricas padrão de machine learning.
- Entender que não existe organização “correta” dos textos — e sim representações mais ou menos úteis para determinadas perguntas.



## 📖 Referência

Grimmer, J., Roberts, M. E., & Stewart, B. M. (2022). {cite}`grimmer2022text` *Text as Data: A New Framework for Machine Learning and the Social Sciences*. Princeton University Press.

CATALINAC, Amy. {cite}`catalinac2016from` From Pork to Policy: The Rise of Programmatic Campaigning in Japanese Elections. The Journal of Politics. [S. l.]: University of Chicago Press, jan. 2016. DOI 10.1086/683073. Disponível em: http://dx.doi.org/10.1086/683073.

KING, GARY; PAN, JENNIFER; ROBERTS, MARGARET E (2013). {cite}`king2013how` How Censorship in China Allows Government Criticism but Silences Collective Expression. American Political Science Review. [S. l.]: Cambridge University Press (CUP), maio 2013. DOI 10.1017/s0003055413000014. Disponível em: http://dx.doi.org/10.1017/S0003055413000014.
