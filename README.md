# Chain of Thought Note

## 1. Initial Problem Framing
- **Objective**: Predict whether the next return of a given stock is positive or negative.  
- **Data**: Historical returns, volumes, and categorical features (sector, industry, etc.).  
- **Challenge**:  
  - Effectively capture temporal patterns while avoiding overfitting.  
  - Leverage market-wide signals and correlated group behaviors to enhance prediction.  

---

## 2. First Approach: Simple Neural Network
- **Method**:  
  - A basic Multi-Layer Perceptron (MLP) using historical returns and volumes as input.  
  - Incorporated one-hot-encoded sector or industry features.  
- **Outcome**:  
  - Achieved ~51.5% accuracy, surpassing a random baseline.  
  - Likely succeeded due to direct correlations between recent returns and near-future direction.  
  - However, it did not fully exploit richer structures or inter-stock interactions.  

---

## 3. Second Approach: Incorporating Correlation Structures & Global/Group Signals
- **Method**:  
  - Added market-wide and group-level signals using correlation structures:  
    1. **Global Market Signal**:  
       - Computed a correlation matrix across all stocks for each date.  
       - Extracted the principal eigenvector to calculate a weighted average daily return.  
    2. **Group-Level Signals**:  
       - Hierarchically grouped stocks using the correlation matrix.  
       - Computed eigenvector-based weighted returns for each group.  
  - Combined these additional features with the MLP model.  
- **Outcome**:  
  - Improved training convergence but worsened evaluation accuracy.  
  - Possible reasons:  
    - Overfitting to the training set due to complex feature engineering.  
    - Mismatch between training and evaluation distributions.  
    - Group signals may not align with short-term directional trends.  

---

## 4. Third Approach: LSTM + Dense Neural Network Hybrid
- **Method**:  
  - Used an LSTM to capture the temporal sequence of returns explicitly:  
    - Input: Historical returns passed into the LSTM.  
    - Combined the LSTM's embedding with additional dense inputs: volumes, global averages, group averages.  
  - Correlation matrices were computed using exponentially weighted moving averages (75% of weight in the first 5 entries).  
- **Outcome**:  
  - Achieved ~60% training accuracy, indicating the model effectively captured temporal structures.  
  - However, evaluation performance was lower than the simpler MLP approach.  
  - Possible reasons:  
    - Overfitting to the training set.  
    - Sensitivity to shifts in return dynamics not represented in training.  

---

## 5. Reflections and Next Steps
- **Key Observations**:  
  - Complex models and features (e.g., LSTM, correlation matrices) risk overfitting or capturing noise.  
  - Simpler models can outperform if they generalize better to evaluation scenarios.  
- **Future Directions**:  
  - Regularization techniques (e.g., dropout, weight decay) and robust cross-validation to mitigate overfitting.  
  - Simplify feature engineering to focus on the most reliable signals.  
  - Explore hybrid approaches that balance simplicity with market structure insights.
  - Explore correlation related features like extended halflife to capture the long term correlation between stocks.
---

## 6. Conclusion
- While advanced architectures (LSTM + dense networks) and group-based features show promise in theory, the simpler MLP still performs best on the evaluation dataset.  
- Future efforts should emphasize balancing complexity with generalization and ensuring robust validation strategies.
