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
- **Primary Target:** Worst Group Accuracy (WGA) on OOD - accuracy of the worst-performing subgroup (Sex × ER history)
- **Secondary Target:** Overall OOD Accuracy - performance on held-out ER domain
- **Constraint:** Robustness Gap |ID - OOD| should remain small
- **Success Criterion:** A strategy is only considered better if it improves WGA and does not significantly hurt OOD accuracy.

**Why WGA Matters:**
- Models can achieve high overall accuracy while failing on minority subgroups.
- Clinical fairness requires good performance across all patient demographics.
- Worst-group performance determines real-world reliability.

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
1. **Primary Goal:** Improve Worst Group Accuracy (WGA) on OOD.
2. **Secondary Goal:** Optimize OOD accuracy and keep ID–OOD gap small.
3. A strategy is only considered better if it improves WGA and does not significantly hurt OOD.
4. Explain your reasoning (why will this help WGA specifically?).
5. Be specific about hyperparameters.
6. Consider combinations of techniques.

**Critical Rules for Strategy Design:**
1. **Group DRO Naming Convention:** If the strategy name contains "group_dro", you MUST set `"use_group_dro": true` in the config.
2. **Fairness Focus:** At least HALF of your proposals should be explicitly fairness/robustness-oriented.
   - Examples: "group_dro_*", "class_balanced_*", "importance_sampling_*", "worst_group_*"
3. **Aggressive Exploration:** Propose strategies that:
   - Aggressively reweight or resample worst-performing groups.
   - Try stronger regularization (vary `l2_C` parameter from 0.1 to 10).
   - Experiment with `sample_frac` (0.3 to 1.0) to shift focus towards hard groups.
   - Use `undersample_majority` to balance classes.
4. **Avoid Baseline-Like Strategies:** Do not propose many similar vanilla strategies. Be bold in targeting group fairness.

**Output Format:**
You must output a JSON list of StrategyConfig objects. Each StrategyConfig must include:

```json
{
  "name": "descriptive_name",
  "sample_frac": 0.3-1.0,
  "undersample_majority": true/false,
  "l2_C": 0.1-10.0,
  "use_group_dro": true/false,
  "class_weight": null/"balanced",
  "reg_strength": "weak"/"normal"/"strong"
}
```

**IMPORTANT:** If strategy name contains "group_dro", set `"use_group_dro": true`

**Example:**
```json
[
  {
    "name": "group_dro_with_strong_regularization",
    "sample_frac": 1.0,
    "undersample_majority": false,
    "l2_C": 0.1,
    "use_group_dro": true,
    "class_weight": null,
    "reg_strength": "strong"
  },
  {
    "name": "aggressive_class_balanced_with_undersampling",
    "sample_frac": 0.5,
    "undersample_majority": true,
    "l2_C": 1.0,
    "use_group_dro": false,
    "class_weight": "balanced",
    "reg_strength": "normal"
  },
  {
    "name": "group_dro_with_minority_focus",
    "sample_frac": 0.7,
    "undersample_majority": true,
    "l2_C": 0.5,
    "use_group_dro": true,
    "class_weight": null,
    "reg_strength": "normal"
  }
]
```

**Reasoning:** Focus on strategies that directly address worst-group performance through reweighting, resampling, or group-aware training.
