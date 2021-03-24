# Week 3 - Sequence models

- Relative ordering (sequence of words) matters for the meaning of a sentence.
- For this we'll use specialized neural networks:
    - **RNN** - Recurrent Nerual Networks
        - Context is preserved from time to time. But it might get lost in longer sentences.
    - **GRU** - Gated Recurrent Units
        - Gates are used for controlling the flow of information in the network. 
        - Gates are capable of learning which inputs in the sequence are important and store their information in the memory unit.
    - **LSTM** - Long Short-Term Memory
        - LSTMs have cell states and they carry the context.