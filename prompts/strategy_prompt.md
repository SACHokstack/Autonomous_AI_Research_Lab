Section 1: Problem Context (Static - same every time)
# Research Problem

**Task:** Hospital Readmission Prediction (30-day readmission risk)

**Dataset:** TableShift diabetes_readmission
- Features: Patient demographics, admission details, medical history
- Target: Binary (readmitted within 30 days: yes/no)
- Training domains: Admission sources [Physician Referral, Clinic Referral, Transfer]
- Test domain (OOD): Emergency Room (ER)

**The Challenge:** Models trained on non-ER admissions fail when deployed on ER patients.

**Why This Matters:** 
- ER patients have different characteristics (more acute, sicker)
- Real-world deployment requires robustness across all admission types
- Poor OOD performance = missed readmissions = patient harm + hospital penalties

**Evaluation Metrics:**
- ID Accuracy: Performance on training domains
- OOD Accuracy: Performance on held-out ER domain
- Robustness Gap: |ID - OOD| (lower is better)
- Target: OOD ≥ 85% with gap < 5%

Section 2: Current Experimental Results (Dynamic - updates each cycle)
# Experimental Results (Cycle 1)

## Completed Experiments

| Strategy | ID Acc | OOD Acc | Gap | Notes |
|----------|--------|---------|-----|-------|
| baseline_lr | 82.3% | 67.1% | 15.2% | Simple logistic regression |
| importance_sampling | 81.7% | 74.5% | 7.2% | KDE-based reweighting |
| group_dro | 79.8% | 76.2% | 3.6% | Min-group optimization |
| focal_loss | 83.1% | 70.3% | 12.8% | γ=2.0, α=0.25 |
| domain_mixup | 80.5% | 72.8% | 7.7% | α=0.3 |

## Key Observations

**Best OOD performance:** group_dro (76.2%)
**Smallest gap:** group_dro (3.6%)
**Highest ID:** focal_loss (83.1%)

**Patterns:**
- Group DRO shows promise for robustness (small gap)
- But OOD performance still below 85% target
- Focal loss helps ID but hurts OOD (overfits to source domains)
- Importance sampling moderately effective

**Domain Analysis:**
- ER domain has: 23% higher acuity scores, 31% more comorbidities
- Baseline completely fails on ER patients with >3 comorbidities (42% acc)

Section 3: Relevant Techniques (Static - curated list)
# Robustness Techniques Reference

## Sample Reweighting
- **Importance Sampling:** Reweight training samples to match test distribution
- **Propensity Scores:** Inverse probability weighting based on domain likelihood
- **Covariate Shift Adaptation:** Density ratio estimation

## Robust Training Objectives
- **Group DRO:** Minimize worst-group loss (Sagawa et al. 2020)
- **Distributionally Robust Optimization:** Optimize under worst-case distribution
- **Focal Loss:** Down-weight easy examples, focus on hard cases
- **Class-Balanced Loss:** Account for label imbalance

## Domain Generalization
- **Domain-Invariant Features:** Learn features that work across domains
- **Domain Mixup:** Interpolate samples from different domains
- **Adversarial Training:** Remove domain-specific information
- **Meta-Learning:** Train on domain shifts explicitly

## Regularization & Architecture
- **L1/L2 Regularization:** Prevent overfitting to source domains
- **Dropout:** Force model not to rely on domain-specific features
- **Early Stopping:** Stop before overfitting to source
- **Feature Selection:** Use only causally-relevant features

## Ensemble Methods
- **Domain-Specific Models + Meta-Learner:** Train separate models, combine predictions
- **Stacking with Domain Features:** Use domain as explicit input to meta-model
- **Uncertainty Weighting:** Weight predictions by confidence

Section 4: Your Task (Static - instructions)
# Your Task

Based on the experimental results above, propose 3-5 NEW training strategies to test.

**Requirements:**
1. Each strategy should address observed weaknesses
2. Explain your reasoning (why will this help?)
3. Be specific about hyperparameters
4. Consider combinations of techniques

**Output Format:**
Return a JSON list of strategy configs:

{
  "hypotheses": [
    {
      "name": "descriptive_name",
      "reasoning": "Why this might work based on results...",
      "strategy_type": "reweighting|robust_loss|ensemble|regularization|domain_gen",
      "config": {
        "method": "specific_method",
        "params": {
          "param1": value1,
          "param2": value2
        }
      },
      "expected_improvement": "What metric should improve and why"
    }
  ]
}

**Example:**
{
  "hypotheses": [
    {
      "name": "group_dro_with_regularization",
      "reasoning": "Group DRO had smallest gap (3.6%) but OOD still low (76.2%). Adding L2 regularization may prevent overfitting while maintaining robustness.",
      "strategy_type": "robust_loss",
      "config": {
        "method": "group_dro",
        "params": {
          "step_size": 0.01,
          "l2_penalty": 0.01
        }
      },
      "expected_improvement": "OOD should reach 80%+ while maintaining <5% gap"
    }
  ]
}