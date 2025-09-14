# O Problema de Classificação

Na última aula, aprofundamos os conceitos fundamentais do aprendizado supervisionado, diferenciando os objetivos de inferência e predição, e discutindo como construir bancos de dados de treinamento confiáveis para aplicações em PLN. Exploramos o papel dos métodos paramétricos e não-paramétricos, os principais trade-offs entre flexibilidade e interpretabilidade, e a importância de equilibrar viés e variância para obter modelos robustos e generalizáveis. Também revisamos métricas essenciais para avaliação de classificadores e destacamos a necessidade de testar os modelos em dados novos para garantir sua utilidade prática. Por fim, apresentamos um protocolo padrão para conduzir pesquisas rigorosas e transparentes em aprendizado de máquina supervisionado.

Na aula de hoje, iremos discutir o problema específico de classificação, e alguns dos modelos mais básicos utilizados para esta tarefa. O problema de classificação surge quando temos uma variável categórica como nossa variável resposta $y$. Ou seja, não queremos prever um valor numérico contínuo (e.g. valor de uma casa), mas uma classe (favorável, desfavorável, incerto). Alguns dos classificadores[^1] mais comuns são: Regressão Logística, *Linear Discriminant Analysis* (*LDA*), *Quadratic Discriminant Analysis*, *Naive Bayes* e *K Nearest Neighbors* (KNN). No capítulo 2, James et al. ({cite}`james2023introduction`.) discutem o KNN, e no capítulo 4 focam nos outros citados acima.


## Por que não Regressão Linear?


Uma questão que pode surgir é a de por que não usar a regressão linear para classificação se podemos colocar as categorias como números? Vamos supor o seguinte caso de classificação em três diagnósticos:

$$
Y =
\begin{cases}
  1 & \text{se AVC;} \\\\
  2 & \text{se Overdose;} \\\\
  3 & \text{se Crise Epiléptica.}
\end{cases}
$$


[^1]: **Classificadores** são modelos de aprendizado de máquina supervisionado projetados para atribuir exemplos a categorias ou classes distintas com base em suas características. Eles são utilizados quando a variável resposta é categórica, como na identificação de sentimentos em textos, classificação de imagens ou detecção de spam em e-mails.