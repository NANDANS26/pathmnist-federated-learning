# PHASE 0: FEDERATED ALGORITHMS AUDIT (FedAvg vs. FedProx) (REVISED)

**Project:** Federated Learning on PathMNIST (FedAvg vs. FedProx)  
**Fellowship Program:** IndiaAI Fellowship 2026–27 Screening Submission  
**Target Codebase:** `third_party/PFLlib`  
**Audit Status:** Technical Correction & Research Validation Completed  
**Date:** November 6, 2026  

---

## 1. Executive Summary

This audit deeply examines the actual implementation of **Federated Averaging (FedAvg)** and **Federated Proximal (FedProx)** within PFLlib. The analysis inspects class hierarchies, method dispatch, parameter aggregation mathematics, optimization routines, and algorithmic differences between the two frameworks.

---

## 2. Federated Averaging (FedAvg) Deep Dive

### 2.1 File Map & Class Architecture `[VERIFIED]`
- **Server:** `system/flcore/servers/serveravg.py` -> `class FedAvg(Server)`
- **Base Server:** `system/flcore/servers/serverbase.py` -> `class Server(object)`
- **Client:** `system/flcore/clients/clientavg.py` -> `class clientAVG(Client)`
- **Base Client:** `system/flcore/clients/clientbase.py` -> `class Client(object)`
- **Entry point:** `system/main.py` -> lines 186–191

### 2.2 Complete Training Lifecycle `[VERIFIED]`
In `serveravg.py`, the training loop executes as follows:
```python
def train(self):
    for i in range(self.global_rounds + 1):
        s_t = time.time()
        self.selected_clients = self.select_clients() # 1. Client selection
        self.send_models()                            # 2. Broadcast global model
        
        if i % self.eval_gap == 0:
            self.evaluate()                           # 3. Model evaluation
            
        for client in self.selected_clients:
            client.train()                            # 4. Local client training
            
        self.receive_models()                         # 5. Collect client updates
        self.aggregate_parameters()                   # 6. Aggregate global parameters
        self.Budget.append(time.time() - s_t)
```

#### Step-by-Step Execution Sequence:
1. **Server Initialization (`Server.__init__` in `serverbase.py:13-65`):**
   - Copies `args.model` to `self.global_model = copy.deepcopy(args.model)`.
   - Initializes client objects via `self.set_clients(clientAVG)`.
   - Prepares tracking arrays: `rs_test_acc`, `rs_test_auc`, `rs_train_loss`.
2. **Client Initialization (`Client.__init__` in `clientbase.py:17-53`):**
   - Instantiates a local model copy `self.model = copy.deepcopy(args.model)`.
   - Instantiates standard PyTorch SGD: `self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)`.
   - Instantiates cross-entropy loss: `self.loss = nn.CrossEntropyLoss()`.
3. **Client Selection (`Server.select_clients` in `serverbase.py:94-101`):**
   - Samples $K = \text{int}(N \times \text{join\_ratio})$ clients uniformly without replacement using `np.random.choice(self.clients, self.current_num_join_clients, replace=False)`.
4. **Global Parameter Distribution (`Server.send_models` in `serverbase.py:103-113`):**
   - Loops over all clients and synchronizes weights: `client.set_parameters(self.global_model)`.
   - `client.set_parameters` performs in-place tensor cloning:
     ```python
     for new_param, old_param in zip(model.parameters(), self.model.parameters()):
         old_param.data = new_param.data.clone()
     ```
5. **Evaluation Timing (`Server.evaluate` in `serverbase.py:228-254`):**
   - Notice: `evaluate()` occurs **BEFORE** `client.train()` in round $i$. Thus, at round $i$, the evaluation measures the performance of the aggregated model from round $i-1$ (and at round $0$, it measures the untrained initial model). The final model after round `global_rounds` aggregation is not evaluated without a post-loop call.
6. **Local Dataset Loading (`Client.load_train_data` in `clientbase.py:55-59`):**
   - Calls `read_client_data(self.dataset, self.id, is_train=True)`.
   - Constructs a `DataLoader(train_data, batch_size, drop_last=True, shuffle=True)`.
7. **Local Client Optimization (`clientAVG.train` in `clientavg.py:12-44`):**
   - Loops through `self.local_epochs` over the local mini-batches.
   - For each $(x, y)$, computes forward pass `output = self.model(x)`, loss `self.loss(output, y)`, zeroes gradients, backpropagates `loss.backward()`, and steps `self.optimizer.step()`.
8. **Client Model Update Collection (`Server.receive_models` in `serverbase.py:114-137`):**
   - Simulates client dropout if configured. Extracts sample counts: $n_k = \text{client.train\_samples}$.
   - Computes normalized sample weights:
     $$p_k = \frac{n_k}{\sum_{j \in S_t} n_j}$$
     Stored in `self.uploaded_weights`.
9. **Global Parameter Aggregation (`Server.aggregate_parameters` in `serverbase.py:138-147`):**
   - Initializes global model parameters to zero:
     ```python
     self.global_model = copy.deepcopy(self.uploaded_models[0])
     for param in self.global_model.parameters():
         param.data.zero_()
     for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
         self.add_parameters(w, client_model)
     ```
   - In `Server.add_parameters` (`serverbase.py:148-151`):
     $$\theta_{global}^{t+1} = \sum_{k \in S_t} p_k \theta_k^{t+1}$$

---

## 3. Federated Proximal (FedProx) Deep Dive

### 3.1 File Map & Class Architecture `[VERIFIED]`
- **Server:** `system/flcore/servers/serverprox.py` -> `class FedProx(Server)`
- **Client:** `system/flcore/clients/clientprox.py` -> `class clientProx(Client)`
- **Optimizer:** `system/flcore/optimizers/fedoptimizer.py` -> `class PerturbedGradientDescent(Optimizer)`
- **Entry point:** `system/main.py` -> lines 204–206

### 3.2 Mathematical Formulation & Algorithmic Design `[INFERRED]`
The theoretical formulation of FedProx (Li et al., 2020) solves a regularized local objective on client $k$:
$$\min_w h_k(w; w_t) = F_k(w) + \frac{\mu}{2} \|w - w_t\|_2^2$$
where:
- $F_k(w) = \frac{1}{n_k} \sum_{i=1}^{n_k} \ell(w; x_{k,i}, y_{k,i})$ is the local empirical risk.
- $w_t$ is the global model snapshot broadcast at the beginning of communication round $t$.
- $\mu \ge 0$ is the proximal penalty coefficient.

The gradient of this local objective with respect to $w$ is:
$$\nabla h_k(w; w_t) = \nabla F_k(w) + \mu (w - w_t)$$

The local proximal gradient update step with step size $\eta$ is:
$$w \leftarrow w - \eta \left( \nabla F_k(w) + \mu (w - w_t) \right)$$

### 3.3 Verification of PFLlib Implementation `[VERIFIED]`

In PFLlib, FedProx is implemented across two key files:
1. `system/flcore/clients/clientprox.py`:
   - Holds an explicit reference to the broadcast global parameters:
     ```python
     self.global_params = copy.deepcopy(list(self.model.parameters()))
     ```
   - Instantiates the custom optimizer:
     ```python
     self.optimizer = PerturbedGradientDescent(
         self.model.parameters(), lr=self.learning_rate, mu=self.mu)
     ```
   - In `set_parameters(self, model)` (called every round by `send_models()`):
     ```python
     def set_parameters(self, model):
         for new_param, global_param, param in zip(model.parameters(), self.global_params, self.model.parameters()):
             global_param.data = new_param.data.clone()
             param.data = new_param.data.clone()
     ```
     *Verification:* Both the client's working parameters (`param`) and the frozen global reference (`global_param`) are correctly refreshed each round with deep-cloned tensors.
   - During local mini-batch training (`clientprox.py:48-50`):
     ```python
     self.optimizer.zero_grad()
     loss.backward()
     self.optimizer.step(self.global_params, self.device)
     ```

2. `system/flcore/optimizers/fedoptimizer.py:64-75`:
   ```python
   class PerturbedGradientDescent(Optimizer):
       def __init__(self, params, lr=0.01, mu=0.0):
           default = dict(lr=lr, mu=mu)
           super().__init__(params, default)

       @torch.no_grad()
       def step(self, global_params, device):
           for group in self.param_groups:
               for p, g in zip(group['params'], global_params):
                   g = g.to(device)
                   d_p = p.grad.data + group['mu'] * (p.data - g.data)
                   p.data.add_(d_p, alpha=-group['lr'])
   ```
   *Mathematical Verification:*
   - `d_p = p.grad.data + group['mu'] * (p.data - g.data)`
   - `p.data.add_(d_p, alpha=-group['lr'])`
   This is mathematically identical to:
   $$w \leftarrow w - \text{lr} \cdot [\nabla F_k(w) + \mu (w - w_{\text{global}})]$$
   The parameter update rule in PFLlib is **100% mathematically correct** with respect to the FedProx paper!

### 3.4 Discrepancy Found in PFLlib Metric Reporting `[VERIFIED]`
While the gradient step is exact, our audit uncovered a discrepancy in how PFLlib evaluates the reported local loss in `clientprox.py:84-86`:
```python
gm = torch.cat([p.data.view(-1) for p in self.global_params], dim=0)
pm = torch.cat([p.data.view(-1) for p in self.model.parameters()], dim=0)
loss += 0.5 * self.mu * torch.norm(gm - pm, p=2)
```
*Mathematical Defect:*
The proximal term is $\frac{\mu}{2} \|w - w_t\|_2^2$ (squared $L_2$ norm). PFLlib computes `torch.norm(gm - pm, p=2)` without squaring. This affects only the printed training loss scalar in logs, not the parameter updates, but it is an important flaw to rectify in our independent implementation.

---

## 4. Key Differences: FedAvg vs. FedProx in PFLlib `[VERIFIED]`

| Dimension | FedAvg (`serveravg.py` / `clientavg.py`) | FedProx (`serverprox.py` / `clientprox.py`) |
|---|---|---|
| **Local Objective** | Empirical Cross-Entropy: $\min_w F_k(w)$ | Proximal-regularized: $\min_w F_k(w) + \frac{\mu}{2} \|w - w_t\|_2^2$ |
| **Optimizer** | Standard PyTorch `torch.optim.SGD` | Custom `PerturbedGradientDescent` |
| **Global Memory Overhead** | Only current working parameters | Working parameters + cached clone of $w_t$ per client |
| **Gradient Term** | $\nabla F_k(w)$ | $\nabla F_k(w) + \mu(w - w_t)$ |
| **Behavior on Noisy Client** | Unconstrained local overfitting to noisy labels; causes large gradient divergence | Proximal anchor penalizes divergence from $w_t$, constraining noisy update norms |
| **Aggregation Formula** | Normalized sample-weighted: $\sum p_k \theta_k$ | Identical normalized sample-weighted: $\sum p_k \theta_k$ |
| **Model Architecture in `main.py`** | Split into `BaseHeadSplit` (lines 187–189) | Left as monolithic `args.model` (lines 204–205) |

---

## 5. Reusable Components vs. Necessary Adaptations

### Reusable Concepts from PFLlib `[PROPOSED]`:
1. `PerturbedGradientDescent` in-place tensor addition pattern: `p.data.add_(d_p, alpha=-lr)`.
2. Parameter broadcasting and zero-initialized accumulation aggregation pattern.
3. Client sampling and communication round sequencing.

### Necessary Adaptations for our Independent Implementation `[PROPOSED]`:
1. **Model Consistency:** Eliminate the asymmetrical `BaseHeadSplit` wrapper present in PFLlib's FedAvg. Both algorithms must train the identical monolithic lightweight CNN.
2. **Correct Metric Calculation:** Calculate the true squared proximal loss: `0.5 * mu * torch.sum((pm - gm) ** 2)`.
3. **Evaluation Alignment:** Add post-aggregation round validation so that metrics reflect the aggregated global model at round $t$ directly.
4. **Update Norm Diagnostics:** Extract $\Delta_k = w_k - w_{\text{global}}$ before aggregation and compute $\|\Delta_k\|_2$.
5. **Pairwise Cosine Similarity & Similarity to Sample-Weighted Mean:** Calculate cosine similarities across all client update vectors and against the sample-weighted mean update after each round.
