# LSTMLanguageModel

**Recurrent Neural Network (RNN):**

```
      -----
     |     |
 x(t) | RNN | --> h(t) --> y(t)
     |     |
      -----
       ^
       |
       h(t-1)
```

* **x(t):**  Input at time step t (e.g., a word in a sentence)
* **RNN:**  The recurrent neural network cell (performs calculations)
* **h(t):**  Hidden state at time step t (stores memory)
* **y(t):**  Output at time step t (e.g., predicted next word)
* **h(t-1):** Hidden state from the previous time step (passed as input to the current step)

**Long Short-Term Memory (LSTM):**

```
      --------------------
     |                    |
 x(t) | LSTM               | --> h(t) --> y(t)
     | (with gates:        | 
     |  forget, input,     |
     |  output)            |
      --------------------
       ^  ^
       |  |
       c(t-1) h(t-1) 
```

* **c(t-1):** Cell state from the previous time step (additional memory)
* **Gates:**  Control the flow of information in and out of the cell state (forget gate, input gate, output gate)

**Gated Recurrent Unit (GRU):**

```
      -----------------
     |                 |
 x(t) | GRU             | --> h(t) --> y(t)
     | (with gates:     | 
     |  reset, update)  |
      -----------------
       ^
       |
       h(t-1)
```

* **Gates:**  Control the flow of information, but with fewer gates than LSTM (reset gate, update gate)


**Key Points:**

* The arrows indicate the flow of information.
* The hidden state (h) is the core of these models, carrying memory forward.
* The LSTM and GRU have additional mechanisms (gates) to regulate information flow and improve their ability to capture long-term dependencies.

