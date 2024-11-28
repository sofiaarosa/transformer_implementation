import streamlit as st

st.title("O Modelo Transformer")
st.caption("A base para os grandes modelos de linguagem")

col1, col2 = st.columns(2)

col2.image("https://www.tensorflow.org/images/tutorials/transformer/transformer.png?hl=pt-br")
col1.subheader("Introdução")
col1.markdown("""Em 2017, Vaswani et. al publicam o artigo  **"Attention is all you need"**, introduzindo uma arquitetura
           chamada de Transformer, baseada em **mecanismos de atenção**. Os mecanismos de atenção substituiriam as RNNs (redes neurais recorrentes)
           na tarefa de transformar dados sequenciais.""")
col1.markdown("""A substituição das RNNs pelos Transformers em tarefas de PLN (Processamento de Linguagem Natural) se tornou
              praticamente inevitável, visto que os mecanismos de atenção diminuem a perda de informação das RNNs e também é mais eficiente, mesmo que
              ambas arquiteturas se baseiem na configuração **encoder-decoder**.""")

st.subheader("As camadas")
st.markdown("Uma implementação do Modelo Transformer está desenvolvida e explicada no seguinte notebook: [Google Colab](https://colab.research.google.com/drive/1t4PA-l8Wu_vRANIgjkl089aQe76J1Vdu?usp=sharing)")
st.markdown('''Um modelo transformer, em seu alto nível, é formado por um *Encoder* (Codificador) e um *Decoder* (Decodificador).
            O Encoder é formado por diversas camadas de codificação e o mesmo ocorre com o Decoder. 
            Esses dois tipos de camada, `EncoderLayer` e `DecoderLayer` utilizam dos mecanismos de atenção para suas respectivas tarefas.''')
st.image("https://media.datacamp.com/legacy/v1704797298/image_199786868d.png")
st.markdown('''Antes do Encoder e do Decoder, existe o pré processamento da entrada, que garante que a **linguagem natural** receberá uma
            **representação matemática**, que, inclusive, determina a posição das palavras na sequência e que o modelo "entenderá"''')
st.markdown('''### Encoder''')
st.markdown('''O Encoder é responsável por **contextualizar** os tokens de entrada, ele captura o contexto de cada token em relação à sequência.
            Para que isso seja possível, é preciso que o modelo tenha atenção em diversos pontos da cadeia de tokens, por isso, utiliza um mecanismo de atenção de "várias
            cabeças". A saída do Encoder é um conjunto de vetores que representam a entrada com diferentes contextos.''')
st.markdown('''### Decoder''')
st.markdown('''O Decoder tem como função principal a geração de sequências de tokens, que se tornarão textos em linguagem natural. Ele também recebe uma entrada de embeddings dos tokens e isso é utilizado para uma das
            camadas de atenção, que as palavras não são influenciadas por palavras subsequentes mas sim, anteriores. A segunda camada de atenção da `DecoderLayer` possibilita a identificação e a "atenção" às partes mais 
            relevantes da saída do Encoder.''')
st.markdown("""
### Multi-Head Attention
- **Objetivo**: Permitir que o modelo aprenda diferentes tipos de relações entre os tokens simultaneamente.
- **Funcionamento**:
  1. **Self-Attention**:
     - Cada token da sequência é transformado em três vetores:
       - **Query (Q)**: O que estou procurando.
       - **Key (K)**: O que ofereço.
       - **Value (V)**: As informações relevantes que carrego.
     - A atenção é calculada como uma pontuação entre Q e K para determinar a relevância entre os tokens.
     - Os valores (V) são combinados com base nas pontuações de atenção.
  2. **Multi-Head**:
     - O mecanismo de atenção é executado várias vezes em paralelo (múltiplas cabeças).
     - Cada cabeça foca em diferentes padrões ou relações nos dados.
     - As saídas de todas as cabeças são concatenadas e processadas por uma camada linear.
""")
st.latex(r"\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V")
st.markdown('''As matrizes Q, K e V são obtidas pela multiplicação do embedding de entrada pelos pesos **aprendíveis**''')
st.markdown("""
### Feedforward Network
- **Objetivo**: Refinar as representações processadas pela atenção.
- **Estrutura**:
  1. Camada totalmente conectada (linear) que expande dimensionalmente o vetor.
  2. Função de ativação (geralmente ReLU) para introduzir não-linearidade.
  3. Outra camada linear que reduz a dimensionalidade de volta ao tamanho original.
""")

st.title("Transformer para NLP")
st.markdown('''No artigo oficial, o Transformer foi posto em teste na tarefa de tradução, nas linguagens Inglês-Alemão e 
            Inglês - Francês.''')
st.markdown('''Neste projeto, o Transformer foi utilizado para a construção de um tradutor na linguagem Português-Inglês, 
            utilizando um dataset **pequeno** bilíngue do [tensorflow](https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate?hl=pt-br#ted_hrlr_translatept_to_en). 
            Foi obtida uma acurácia de aproximadamente 68% durante o treinamento.''')
st.markdown('''O modelo transformer é base para todos os grandes modelos de linguagem (LLMs) da atualidade. Sua arquitetura foi modificada para que novas
            tarefas de processamento de linguagem natural pudessem ser realizadas por ele de forma satisfatória. Por exemplo, os modelos da família GPT utilizam
            uma arquitetura decoder-only, o que faz mais sentido para a tarefa de **geração de texto**. Outros modelos, treinados para sumarizar textos, por exemplo,
            geralmente utilizam uma arquitetura encoder-only.''')