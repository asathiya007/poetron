# Poetron

Poetron is an AI system that uses a custom generative pretrained transformer (GPT) language model to write short poems.

See the `poetron.ipynb` notebook for a demo.

The Poetron language model is trained on data from these datasets:
- Haiku Dataset by Harshit Jhalani: [https://www.kaggle.com/datasets/hjhalani30/haiku-dataset](https://www.kaggle.com/datasets/hjhalani30/haiku-dataset). This dataset is licensed under the Attribution 4.0 International license, more info is available here: [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/). The data is preprocessed (see code for more details) before it is used to train the language model.
- Haiku Dataset by bfbarry: [https://www.kaggle.com/datasets/bfbarry/haiku-dataset](https://www.kaggle.com/datasets/bfbarry/haiku-dataset)

Poetron's implementation is based on the following sources.
- "Let's build GPT: from scratch, in code, spelled out" video by Andrej Karpathy: [https://www.youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Google Colab notebook for "Let's build GPT: from scratch, in code, spelled out" video by Andrej Karpathy: [https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)
- "Attention is All You Need" paper by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin: [https://arxiv.org/pdf/1706.03762](https://arxiv.org/pdf/1706.03762)
