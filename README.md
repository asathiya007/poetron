# Poetron

Poetron is an AI system that uses a custom generative pretrained transformer (GPT) language model to write short poems.

See the `poetron.ipynb` notebook for a demo.

The Poetron language model is trained on data from these datasets:
- Haiku Dataset by Harshit Jhalani: [https://www.kaggle.com/datasets/hjhalani30/haiku-dataset](https://www.kaggle.com/datasets/hjhalani30/haiku-dataset). This dataset is licensed under the Attribution 4.0 International license, more info is available here: [https://creativecommons.org/licenses/by/4.0/][https://creativecommons.org/licenses/by/4.0/]. The data is preprocessed (see code for more details) before it is used to train the language model.
- Haiku Dataset by bfbarry: [https://www.kaggle.com/datasets/bfbarry/haiku-dataset](https://www.kaggle.com/datasets/bfbarry/haiku-dataset)

Poetron's implementation is based on the following sources.
- "Let's build GPT: from scratch, in code, spelled out" video by Andrej Karpathy: [https://www.youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Google Colab notebook for "Let's build GPT: from scratch, in code, spelled out" video by Andrej Karpathy: [https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbkYxc3hBRThQWUJrQWdpZEE1MWJjWVNFMml5Z3xBQ3Jtc0tuN2h1c2gtNDhpOE16SEVwUTZMb2F5RTFDTGN2ZGIzRVJlelZNa2xSX2c1enpFWHR6TDM5UVJ5bktMamhJdnpQa0RCNjVObkpKcXBBX3cxcWtKXzlpNE5JWGlSSTYzOTJMbnFMam5menlyQlZGdHF6WQ&q=https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-%3Fusp%3Dsharing&v=kCc8FmEb1nY](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbkYxc3hBRThQWUJrQWdpZEE1MWJjWVNFMml5Z3xBQ3Jtc0tuN2h1c2gtNDhpOE16SEVwUTZMb2F5RTFDTGN2ZGIzRVJlelZNa2xSX2c1enpFWHR6TDM5UVJ5bktMamhJdnpQa0RCNjVObkpKcXBBX3cxcWtKXzlpNE5JWGlSSTYzOTJMbnFMam5menlyQlZGdHF6WQ&q=https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-%3Fusp%3Dsharing&v=kCc8FmEb1nY)
- "Attention is All You Need" paper by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin: [https://arxiv.org/pdf/1706.03762](https://arxiv.org/pdf/1706.03762)
