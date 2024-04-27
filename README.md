# federated-learning (DOING)
Small project to better understand: https://arxiv.org/pdf/1602.05629 McMahan et al., “Communication-Efficient Learning of Deep Networks from Decentralized Data”, 2016


**Arguments and Parameters of FedAvg**

- **Number of Clients (K)**: Total number of clients that own local datasets.
- **Fraction of Clients (C)**: Fraction of clients that are randomly selected to participate in each round of training.
- **Local Epochs (E)**: Number of epochs each client should train on its local data before sending updates back to the server.
- **Batch Size (B)**: The size of the mini-batches used for local training on each client.
- **Learning Rate (η)**: The learning rate used by clients for local optimization.
