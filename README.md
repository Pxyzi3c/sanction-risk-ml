# Sanction risk ml
Data science extension repository of my first end-to-end project.

**Objective:** Leverage OFAC screening data to detect high-risk patterns, optimize name matching thresholds, and provide insights for proactive risk detection.

**Detailed Documentation:** [Sanction Intelligence System for Proactive Risk Detection](https://www.notion.so/Sanction-Intelligence-System-for-Proactive-Risk-Detection-2010742294af80e89652e83e7a3f2f1d?source=copy_link)

------------------------------------------------------------------------------------------------------
## Model Training and Evaluation
To effectively detect high-risk patterns within OFAC screening data and optimize our name matching thresholds, we trained and evaluated several classification models. Given the significant class imbalance in our dataset (a very small percentage of "is_match" instances), standard accuracy metrics can be misleading. Therefore, model selection and performance assessment primarily relied on the F1-score and AUC-ROC (Area Under the Receiver Operating Characteristic Curve) for their robustness in such scenarios.

### Classifiers Employed
We trained three distinct classification models:
1. **Logistic Regression:** A linear model often used as a baseline for binary classification. To address class imbalance, it was configured with class_weight="balanced".
2. **Random Forest Classifier:** An ensemble tree-based method known for its robustness and ability to handle non-linear relationships. It was also configured with class_weight="balanced" to account for the minority class.
3. **XGBoost Classifier (Extreme Gradient Boosting):** A highly optimized and powerful gradient boosting framework, often achieving state-of-the-art results. It was intended to be configured with scale_pos_weight (the equivalent of class_weight="balanced" for XGBoost) to handle the imbalance, though its implementation needed specific attention during development to ensure this was active.

### Chosen Model and Performance Findings
Based on the F1-score and AUC-ROC metrics for the positive class (high-risk match), the XGBoost Classifier emerged as the chosen model.

### Key Findings:
The XGBoost model, after tuning and (presumably) correctly configuring scale_pos_weight, achieved perfect classification performance on the test set. All metrics, including precision, recall, and F1-score for both the majority (Class 0, non-match) and minority (Class 1, match) classes, reached 1.000. Correspondingly, the AUC-ROC score is also 1.00.