# Project Update

## 1. Overview
I have tested several methods to predict whether a stock’s next return will be positive or negative, using historical returns, volumes, and categorical features (e.g., sector or industry). Each approach aimed to capture temporal and cross-sectional (group) information from the market, with varying levels of success.

> **Note**: Overfitting can be mitigated by standard techniques (dropout, weight decay, etc.), but main improvements were done on leveraging grouping methods and “date-transversal” signals.

---

## 2. Tested Approaches

### 2.1 Simple MLP
- **Method**: A basic multi-layer perceptron taking in historical returns, volumes, and one-hot categorical features.  
- **Key Insight**: Provided a modest performance lift (~51.5% accuracy), indicating that recent return patterns and basic categorical data offer some predictive power.

### 2.2 Correlation-Based Group Signals
- **Method**:  
  - Computed correlation matrices and derived global or group-level signals (eigenvector-based weighted returns).  
  - Combined these features with the MLP to incorporate market-wide and group behavior.  
- **Key Insight**: Though conceptually appealing, it did not consistently improve evaluation accuracy, likely due to misalignment between short-term predictions and larger group trends.

### 2.3 LSTM + Dense Hybrid
- **Method**:  
  - Used an LSTM for temporal sequences of returns, then merged its output with dense features (volumes, global/group signals).  
  - Correlation matrices were updated via exponentially weighted methods.  
- **Key Insight**: Showed strong training performance (~60% training accuracy) but struggled on unseen data, highlighting the complexity of temporal dependencies and potential mismatch in real-world shifts.

### 2.4 Node2Vec on Correlation Graph
- **Method**:  
  - Treated each stock as a node in a graph, where edge weights came from the correlation matrix.  
  - Learned node embeddings via Node2Vec, then for each stock calculated the average “RET_1” over its *k* nearest neighbors (in embedding space).  
- **Key Insight**: Provides a novel way to integrate similarity across stocks by capturing more nuanced structural relationships. The derived neighbor-based features could enrich downstream prediction tasks.

---

## 3. Expected Improvements and Next Steps

1. **Enhanced Grouping & Date-Transversal Information**  
   - Dynamically leverage arbitrary or more granular grouping structures (beyond standard sectors or hierarchical clustering).  
   - Incorporate signals aggregated across a date—for instance, learning from patterns that span multiple stocks on the same trading day.

2. **Focus on Feature Engineering**  
   - Greater performance gains are expected by identifying and integrating effective features (e.g., refined correlation signals, additional market indicators).  
   - The choice of complex neural network architectures versus simpler models appears less critical at the current stage.

3. **Rapid Iterations via Classification Trees**  
   - Using classification trees (e.g., random forests, gradient boosting) can quickly gauge the utility of new features.  
   - These models are more efficient to train and interpret, speeding up the iterative feature engineering process.

---

## 4. Conclusion
Our experiments indicate that while neural networks (MLPs, LSTMs) and advanced graph-based methods (Node2Vec) can capture rich structures, simpler approaches often suffice to benchmark new feature ideas. Going forward, the most promising avenue lies in refining how we group stocks and how we extract cross-sectional insights on each trading date, rather than solely tuning model architectures.
