# Project Update

## 1. Overview
We have tested several methods to predict whether a stock’s next return will be positive or negative, using historical returns, volumes, and categorical features (e.g., sector or industry). Each approach aimed to capture temporal and cross-sectional (group) information from the market, with varying levels of success.

> **Note**: Overfitting can be mitigated by standard techniques (dropout, weight decay, etc.), but the central focus here is on leveraging improved grouping methods and “date-transversal” signals.

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


[//]: # (# Chain of Thought Note)

[//]: # ()
[//]: # (## 1. Initial Problem Framing)

[//]: # (- **Objective**: Predict whether the next return of a given stock is positive or negative.  )

[//]: # (- **Data**: Historical returns, volumes, and categorical features &#40;sector, industry, etc.&#41;.  )

[//]: # (- **Challenge**:  )

[//]: # (  - Effectively capture temporal patterns while avoiding overfitting.  )

[//]: # (  - Leverage market-wide signals and correlated group behaviors to enhance prediction.  )

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## 2. First Approach: Simple Neural Network)

[//]: # (- **Method**:  )

[//]: # (  - A basic Multi-Layer Perceptron &#40;MLP&#41; using historical returns and volumes as input.  )

[//]: # (  - Incorporated one-hot-encoded sector or industry features.  )

[//]: # (- **Outcome**:  )

[//]: # (  - Achieved ~51.5% accuracy, surpassing a random baseline.  )

[//]: # (  - Likely succeeded due to direct correlations between recent returns and near-future direction.  )

[//]: # (  - However, it did not fully exploit richer structures or inter-stock interactions.  )

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## 3. Second Approach: Incorporating Correlation Structures & Global/Group Signals)

[//]: # (- **Method**:  )

[//]: # (  - Added market-wide and group-level signals using correlation structures:  )

[//]: # (    1. **Global Market Signal**:  )

[//]: # (       - Computed a correlation matrix across all stocks for each date.  )

[//]: # (       - Extracted the principal eigenvector to calculate a weighted average daily return.  )

[//]: # (    2. **Group-Level Signals**:  )

[//]: # (       - Hierarchically grouped stocks using the correlation matrix.  )

[//]: # (       - Computed eigenvector-based weighted returns for each group.  )

[//]: # (  - Combined these additional features with the MLP model.  )

[//]: # (- **Outcome**:  )

[//]: # (  - Improved training convergence but worsened evaluation accuracy.  )

[//]: # (  - Possible reasons:  )

[//]: # (    - Overfitting to the training set due to complex feature engineering.  )

[//]: # (    - Mismatch between training and evaluation distributions.  )

[//]: # (    - Group signals may not align with short-term directional trends.  )

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## 4. Third Approach: LSTM + Dense Neural Network Hybrid)

[//]: # (- **Method**:  )

[//]: # (  - Used an LSTM to capture the temporal sequence of returns explicitly:  )

[//]: # (    - Input: Historical returns passed into the LSTM.  )

[//]: # (    - Combined the LSTM's embedding with additional dense inputs: volumes, global averages, group averages.  )

[//]: # (  - Correlation matrices were computed using exponentially weighted moving averages &#40;75% of weight in the first 5 entries&#41;.  )

[//]: # (- **Outcome**:  )

[//]: # (  - Achieved ~60% training accuracy, indicating the model effectively captured temporal structures.  )

[//]: # (  - However, evaluation performance was lower than the simpler MLP approach.  )

[//]: # (  - Possible reasons:  )

[//]: # (    - Overfitting to the training set.  )

[//]: # (    - Sensitivity to shifts in return dynamics not represented in training.  )

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## 5. Reflections and Next Steps)

[//]: # (- **Key Observations**:  )

[//]: # (  - Complex models and features &#40;e.g., LSTM, correlation matrices&#41; risk overfitting or capturing noise.  )

[//]: # (  - Simpler models can outperform if they generalize better to evaluation scenarios.  )

[//]: # (- **Future Directions**:  )

[//]: # (  - Regularization techniques &#40;e.g., dropout, weight decay&#41; and robust cross-validation to mitigate overfitting.  )

[//]: # (  - Simplify feature engineering to focus on the most reliable signals.  )

[//]: # (  - Explore hybrid approaches that balance simplicity with market structure insights.)

[//]: # (  - Explore correlation related features like extended halflife to capture the long term correlation between stocks.)

[//]: # (  - What is lost is grouping over the date - a date transversal learnable information; this would be beneficial for improvement.)

[//]: # (---)

[//]: # ()
[//]: # (## 6. Conclusion)

[//]: # (- While advanced architectures &#40;LSTM + dense networks&#41; and group-based features show promise in theory, the simpler MLP still performs best on the evaluation dataset.  )

[//]: # (- Future efforts should emphasize balancing complexity with generalization and ensuring robust validation strategies.)
