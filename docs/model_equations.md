# $$\frac{dr}{dt} = \frac{k_1}{\sigma^2} U^T \frac{\partial f}{\partial x} (I - f(Ur)) + \frac{k_1}{\sigma_{td}^2} (r^{td} - r) - \frac{k_1}{2} g'(r)$$

### **The Variables**

- **$\frac{dr}{dt}$**: The rate of change of the neural activity (firing rate) over time.
    
- **$r$**: A vector representing the current firing rates of the neurons (also referred to as the network's internal estimate of the "causes" of the image).
    
- **$k_1$**: A positive rate constant that controls how fast the neurons update their firing rates (akin to a learning rate, but for neural activation rather than synaptic weights).
    
- **$\sigma^2$**: The variance of the noise in the bottom-up input. It dictates how much "trust" the network places in the raw sensory input.
    
- **$\sigma_{td}^2$**: The variance of the noise in the top-down prediction. It dictates how much "trust" the network places in the predictions coming from the higher cortical level.
    
- **$U$**: A matrix representing the feedforward synaptic weights of the neurons (or the "basis vectors" that encode features like oriented edges).
    
- **$U^T$**: The transpose of the synaptic weight matrix $U$. In the biological model, this represents the reciprocal feedback connections.
    
- **$I$**: The actual bottom-up input signal (e.g., the raw image patch or the activity from a lower cortical area).
    
- **$f$**: The neuronal activation function (typically a non-linear function like $tanh$, though the paper sometimes assumes a linear function where $f(x) = x$).
    
- **$x$**: A shorthand for $Ur$, the linear combination of the basis vectors.
    
- **$r^{td}$**: The "top-down" prediction of neural activity coming from the next higher level in the cortical hierarchy (e.g., V2 predicting V1).
    
- **$g(r)$**: A mathematical prior distribution applied to the neural activity. It acts as a regularization constraint.
    
- **$g'(r)$**: The derivative of that prior with respect to $r$.

# $$\frac{dU}{dt} = \frac{k_2}{\sigma^2} \frac{\partial f}{\partial x} (I - f(Ur)) r^T - k_2 \lambda U$$

### **The Variables**

- **$\frac{dU}{dt}$**: The rate of change of the synaptic weight matrix $U$ over time.
    
- **$k_2$**: A positive rate constant governing the learning rate for the synapses.
    
- **$r^T$**: The transpose of the neural activity vector. In a biological context, this represents the **presynaptic** activity.
    
- **$\lambda$**: A positive constant related to the variance of the prior distribution applied to the weights, acting as a weight decay parameter.
    

_(Note: $\sigma^2$, $\frac{\partial f}{\partial x}$, $I$, and $f(Ur)$ function exactly as they did in the state dynamics equation.)_
