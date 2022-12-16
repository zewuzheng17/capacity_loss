# capacity_loss

1.(8分)任取A的幂集中的一个元素集合C，即$\forall C \in P(A)$, 有$ C \subseteq A$ (2分)；由条件A是B的子集知 $C \subseteq A\subseteq C$，即C为B的子集（2分）；所以C属于B的幂集，$C\in P(B)$ $\color{red}{(2分)}$。至此，$P(A) \subseteq P(B)$ 。反之，若A的幂集是B的幂集的子集，一定有A是B的子集（2分）。

2. (8分) 假设A的子集$X_1$和$X_2$有$X_1\neq X_2$，我们要证$S_f(X_1)\neq S_f(X_2)$。首先，存在$a\in X_1-X_2$，即$f(a)\in S_f(X_1)$。假设$f(a)\in S_f(X_2)$，则存在$a'\in X_2$使得$f(a)=f(a')$，因f为单射函数知$a=a'$，但这与$a\in X_1-X_2$矛盾。故$f(a)\in S_f(X_1)-S_f(X_2)，S_f(X_1)\neq S_f(X_2)$。
