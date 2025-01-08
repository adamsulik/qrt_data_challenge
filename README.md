# Project Description

## 1. Overview
I have tested several methods to predict whether a stock’s next return will be positive or negative, using historical returns, volumes, and categorical features (e.g., sector or industry). Each approach aimed to capture temporal and cross-sectional (group) information from the market, with varying levels of success.

> **Note**: Overfitting can be mitigated by standard techniques (e.g., dropout, weight decay). However, main improvements can be achieved on leveraging improved grouping methods and “date-transversal” signals, rather than on complex architectures.

---

## 2. Tested Approaches

### 2.1 Simple MLP
- **Method**: A basic multi-layer perceptron taking in historical returns, volumes, and one-hot categorical features.  
- **Key Insight**: Achieved a modest performance lift (~51.5% accuracy), indicating that recent return patterns and basic categorical data offer some predictive power.

### 2.2 Correlation-Based Group Signals
- **Method**:  
  - Computed correlation matrices and derived a global or group-level signal using the eigenvectors.  
  - **Important Detail**: For each date-stock pair, an additional feature was added—a weighted average of `RET_1` based on the first eigenvector of the correlation matrix (i.e., the eigenvector defines the weights).  
  - Combined these features with the MLP to incorporate market-wide and group behavior.  
- **Key Insight**: Though conceptually appealing, it did not consistently improve evaluation accuracy—possibly due to misalignment between short-term predictions and the larger group trends.

### 2.3 LSTM + Dense Hybrid
- **Method**:  
  - Used an LSTM for temporal sequences of returns, leveraging the **order** of values in the sequence (which is lost in other approaches).  
  - Merged the LSTM output with dense features (volumes, global/group signals).  
  - Updated correlation matrices via exponentially weighted methods.  
- **Key Insight**: Showed strong training performance (~60% accuracy) but struggled on unseen data, highlighting the complexity of temporal dependencies and potential mismatch in real-world market shifts.

### 2.4 Node2Vec on Correlation Graph
- **Method**:  
  - Treated each stock as a node in a graph, where edge weights came from the correlation matrix.  
  - Learned node embeddings via Node2Vec, then for each stock calculated the average `RET_1` over its *k* nearest neighbors in embedding space.  
- **Key Insight**: Provides a novel way to incorporate similarity across stocks by capturing more nuanced structural relationships. The derived neighbor-based features could enrich downstream prediction tasks.

---

## 3. Expected Improvements and Next Steps

1. **Enhanced Grouping & Date-Transversal Information**  
   - Use or refine existing group labels (e.g., **SECTOR**) and compute per-group correlation-based features (e.g., weighted average of eigenvector-based returns for each sector separately).  
   - Dynamically leverage arbitrary or more granular group structures (beyond standard sectors or hierarchical clustering).  
   - Incorporate signals aggregated across a date—for instance, learning from patterns that span multiple stocks on the same trading day.

2. **Focus on Feature Engineering**  
   - Greater performance gains are expected by identifying and integrating effective features (e.g., refined correlation signals, additional market indicators).  
   - Complex neural network architectures vs. simpler models appears less critical at this stage.

3. **Rapid Iterations via Classification Trees**  
   - Classification trees (e.g., random forests, gradient boosting) can quickly gauge the utility of new features.  
   - These models are more efficient to train and interpret, speeding up the iterative feature engineering process.

---

## 4. Conclusion
Our experiments show that while neural networks (MLPs, LSTMs) and advanced graph-based methods (Node2Vec) can capture rich market structures, simpler approaches often suffice to benchmark new feature ideas. Going forward, the most promising avenue lies in:
- Refining how we group stocks (including sector-level embeddings).  
- Extracting cross-sectional insights on each trading date via correlation-based or eigenvector-based signals.  
- Iterating swiftly with classification trees to validate new feature concepts before refining model architectures.
