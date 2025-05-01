# O Texto como um Dado

````{margin}
```{admonition} Citação
:class: note
"Os pesquisadores podem aprender muito sobre o comportamento humano a partir de textos, mas, para isso, é necessário um engajamento com o contexto em que esses textos são produzidos. Uma compreensão profunda do contexto das ciências sociais permitirá que os pesquisadores façam perguntas mais importantes e impactantes, garantam que as medidas extraídas sejam válidas e estejam mais atentos às implicações práticas e éticas de seu trabalho." ({cite}`grimmer2022text`, p. 25, Tradução nossa).
```
````

Nos últimos anos, a explosão de dados digitais e o avanço das capacidades computacionais transformaram a análise de texto em uma ferramenta poderosa para diferentes áreas — da ciência política às humanidades digitais e à indústria. No curso, focaremos em aplicações na ciência política e, mais especificamente, na nossa aplicação para publicações de políticos brasileiros na rede social X/Twitter. No entanto, cresce cada vez mais as aplicações nas humanidades e ciências sociais ([Exemplo](https://aclanthology.org/volumes/2024.nlp4dh-1/)). 


Textos são registros riquíssimos da atividade humana: é por meio da linguagem que se expressam políticas públicas, manifestações culturais, debates eleitorais e sentimentos. Ainda assim, até pouco tempo atrás, a análise sistemática de grandes volumes de texto era restrita ou inviável. Com o avanço dos métodos de **Aprendizado de Máquina** e técnicas de **Processamento de Linguagem Natural (PLN)**, passou a ser possível transformar textos em dados estruturados e analisá-los para fazer descrever e fazer inferências sobre o comportamento humano. No entanto, ainda há uma escassez de bancos de dados de textos com classificação e supervisão humana para a utilização no aprendizado de máquina, especialmente na Língua Portuguesa. Por isso, é necessário que mais pesquisadores tenham domínio das técnicas de aprendizado supervisionado (e não supervisionado). E também entendam que essas técnicas não substituem a análise atenciosa e dedicada do pesquisador, mas as amplificam, gerando oportunidades de pesquisa e descrição do comportamento humana de forma mais ampla.



## O Paradigma proposto por Grimmer et al. 

O livro *Text as Data: A New Framework for Machine Learning and the Social Sciences*  de Grimmer, Roberts e Stewart (2022) propõe um paradigma (ou *framework*) específico para integrar métodos de PLN e machine learning ao trabalho de pesquisa social. Esse paradigma envolve uma nova forma de ver o processo de produção científico, agora indutivo ao invés de dedutivo.

<figure>
  <img src="../aula1/images/image1.png" alt="Modelos Dedutivos e Interativos" style="width: 100%; max-width: 2400px;">
  <figcaption>Figura 1: Modelos Dedutivos e Interativos. 
  Fonte: Grimmer et al. (p.41)</figcaption>
</figure>


### Modelo Dedutivo

Os autores dividem o processo de pesquisa nas ciências sociais nos modelos indutivo e dedutivo. O dedutivo é o método em que estamos mais acostumados a ver: Com base em uma literatura e teoria, construímos a pergunta de pesquisa. Linearmente, seguimos para a construção das hipóteses. Só depois dessas duas fases, partimos para a coleta dos dados e análise, apresentando nossos resultados, seja na forma de artigos ou livros. Há o pressuposto forte de que esse processo é acíclico. Ou seja, um passo do processo não pode influenciar o anterior, há uma direção única e sequencial. 

No entanto, os autores argumentam que isso é ilusório e uma "ficção" (Id., p. 40), e não é a maneira que a pesquisa avança nas ciências sociais de fato. Sustentar essa ficção atrapalharia a discussão de como certas hipóteses e perguntas foram formuladas, como a análise dos dados as influenciou, e como os conceitos e perguntas foram refinados pela análise detalhada dos dados e textos coletados.

```{admonition} Com a palavra, os autores
:class: quote
"Se o procedimento dedutivo padrão for seguido de forma muito rígida e os dados forem coletados apenas no último momento, os pesquisadores podem perder a oportunidade de refinar seus conceitos, desenvolver novas teorias e avaliar novas hipóteses. Grande parte do aprendizado ocorre durante a análise dos dados. Mesmo quando um projeto de pesquisa começa com uma pergunta clara de interesse, frequentemente termina com um foco substancialmente diferente. Foi isso que aconteceu em um de nossos próprios projetos, uma análise das mídias sociais chinesas conduzida por Gary King, Jennifer Pan e Margaret Roberts (King, Pan e Roberts, 2013)."  
({cite}`grimmer2022text`, p. 39, tradução nossa).
```


### Modelo Indutivo

- **Descoberta**: identificar padrões, categorias ou temas em grandes volumes de texto.
- **Mensuração**: quantificar a presença de conceitos em textos.
- **Predição**: prever características ou resultados com base em dados textuais.
- **Inferência causal**: estimar efeitos de intervenções ou mudanças usando textos como variáveis.



---

## Análise Agnóstica do Texto


Seis principios



## 💡 Exemplo didático: Catalinac (2016)

Estudo clássico que ilustra o ciclo de descoberta → mensuração → inferência:
- **Problema**: por que políticos japoneses começaram a discutir mais segurança nacional após 1994?
- **Dados**: manifestos de campanha de todos os candidatos ao parlamento.
- **Método**: Latent Dirichlet Allocation (LDA) para identificar temas.
- **Validação**: leitura dos tópicos e comparação com fatos conhecidos.
- **Inferência**: estimativa do efeito da reforma eleitoral na mudança de agenda.



## 🚀 Conclusão

Trabalhar com **text as data** exige:
- Combinar métodos computacionais e teoria social.
- Respeitar a lógica iterativa de descoberta e refinamento conceitual.
- Validação constante, adequada aos objetivos substantivos e não apenas métricas padrão de machine learning.
- Entender que não existe organização “correta” dos textos — e sim representações mais ou menos úteis para determinadas perguntas.



## 📖 Referência

Grimmer, J., Roberts, M. E., & Stewart, B. M. (2022). *Text as Data: A New Framework for Machine Learning and the Social Sciences*. Princeton University Press.
