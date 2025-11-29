import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm, logistic
from scipy.integrate import quad
import matplotlib.pyplot as plt

st.markdown("""
    ## "LIFE ISN'T FAIR!"
    This tool simulates a hiring algorithm that evaluates applicants from three demographic groups (A, B, C) using a single, uniform threshold. Despite treating everyone "equally," the system produces drastically different outcomes across groups.

    **The simulation models:**
    1. **True Skill (Q)**: Each applicant's actual ability (unobservable in real life)
    2. **Observed Score**: Noisy measurements (like test scores) that approximate true skill IRL
    3. **Evaluation Bias (ε)**: Systematic noise that affects groups differently
        - Group A: High bias (disadvantaged communities with less access)
        - Group B: Moderate bias
        - Group C: Low bias (privileged groups with structural advantages)
    4. **Hiring Decision**: Binary accept/reject based on a fixed threshold (τ)
    
            
    Adjust these simulation and group-level parameters (+more) in the sidebar!
    """)

st.markdown("""
    ### Table of contents
    - *Fairness metrics*: measure disparate impact through false negative rates, false positive rates, acceptance rates, and hire quality
    - *Bayesian posterior analysis*: see how different groups face vastly different selection pressures despite identical thresholds
    - *Distribution visualization*: see the gap between true skill and biased observations, revealing where talent is missed
    - *Bootstrapping*: quantify uncertainty in fairness metrics with confidence intervals
    - *Threshold sensitivity*: analyze fairness trade-offs across different hiring thresholds
    - **Logistic Regression** (just for fun...): train a simple LR model on biased data and watch it learn to perpetuate those biases!
    """)

st.markdown("""     
    Adjust the parameters, run the analyses, and see for yourself: find a threshold that's fair to all groups! *(SPOILER AT END/BOTTOM)*
    """)

st.markdown("---")   

# Define sidebar sliders!
st.sidebar.header("Simulation parameters")

N_APPLICANTS = st.sidebar.slider(
    "Applicant population (N)",
    min_value=10_000,
    max_value=200_000,
    value=100_000,
    step=10_000,
    help="Total simulated applicants (split approximately equally across groups)"
)

Q_TARGET = st.sidebar.slider(
    "True qualification skill target (Q_target)",
    min_value=3.0,
    max_value=9.0,
    value=7.0,
    step=0.1,
    help="Minimum true Q that we *consider* to be objectively qualified"
)

HIRING_THRESHOLD_TAU = st.sidebar.slider(
    "Hiring threshold (τ on observed score)",
    min_value=3.0,
    max_value=10.0,
    value=7.5,
    step=0.1,
    help="Observable score threshold used to decide hires"
)

st.sidebar.markdown("### Group-level parameters")
st.sidebar.markdown("**True Qualification (Q)**")
GROUP_A_MEAN = st.sidebar.slider("Group A", 3.0, 9.0, 5.0, 0.1,
                                 help="Unobservable true qualification of applicant we want to estimate")
GROUP_B_MEAN = st.sidebar.slider("Group B", 3.0, 9.0, 6.5, 0.1)
GROUP_C_MEAN = st.sidebar.slider("Group C", 3.0, 9.0, 8.0, 0.1)

st.sidebar.markdown("**Evaluation Bias (ε)**")
GROUP_A_BIAS = st.sidebar.slider("Group A", 0.1, 2.0, 1.0, 0.1, 
                                   help="Higher = more subjective/biased evaluation")
GROUP_B_BIAS = st.sidebar.slider("Group B", 0.1, 2.0, 0.7, 0.1)
GROUP_C_BIAS = st.sidebar.slider("Group C", 0.1, 2.0, 0.5, 0.1)

GROUP_QUALIFICATION_STDEV = 1.0 # intrinsic variability of true Q
OBSERVABLE_SCORE_STDEV = 0.8 # extra measurement/rounding bias (E)

GROUP_QUALIFICATION_MEANS = {'A': GROUP_A_MEAN, 'B': GROUP_B_MEAN, 'C': GROUP_C_MEAN}
GROUP_BIAS_LEVELS = {'A': GROUP_A_BIAS, 'B': GROUP_B_BIAS, 'C': GROUP_C_BIAS}

SEED = st.sidebar.number_input("Random seed (for reproducibility)", min_value=0, value=42, step=1)

# generate dataframe
@st.cache_data
def generate_synthetic_data(N, q_means, bias_levels, q_target, tau, seed=0):
    randomnum = np.random.default_rng(seed) # random number generator
    # split groups evenly (remainder assigned to A)
    groups = np.repeat(['A', 'B', 'C'], N // 3)
    remainder = N - len(groups)
    if remainder > 0:
        groups = np.concatenate([groups, np.array(['A'] * remainder)])
    randomnum.shuffle(groups)
    
    applicants = []
    for group in ['A', 'B', 'C']:
        # find all people in this group
        group_indices = np.where(groups == group)[0]
        num_people_in_group = len(group_indices)
        
        # Generate TRUE SKILL (what we want to estimate): applicant's actual ability/qualification, UNOBSERVABLE in real life
        true_skill_mean = q_means[group] # groups have different average skills (structural inequality), set in sidebar sliders
        true_skill = randomnum.normal(loc=true_skill_mean, 
                                      scale=GROUP_QUALIFICATION_STDEV, 
                                      size=num_people_in_group)
        
        # Label who is ACTUALLY qualified (if we had perfect information)
        actually_qualified = true_skill > q_target  # boolean: True if person would succeed
        
        # Generate TEST SCORE (imperfect proxy for true skill): like a GPA or standardized test - tries to measure skill but has BIAS
        observed_score = randomnum.normal(loc=true_skill, # Centered around true skill
                                      scale=OBSERVABLE_SCORE_STDEV, # ...but with measurement error (noisy)
                                      size=num_people_in_group)
        
        # Add evaluation BIAS (different for each group): represents subjective bias in interviews, recommendations, etc.
        # groups with less access/privilege get MORE BIAS (higher s = more randomness)
        bias_BIAS_level = bias_levels[group]
        evaluation_bias = logistic.rvs(loc=0, 
                                       scale=bias_BIAS_level, 
                                       size=num_people_in_group, 
                                       random_state=randomnum)
        
        # Calculate FINAL OBSERVED SCORE (what the algorithm sees) = test_score + random bias/BIAS from the evaluation process
        final_score = observed_score + evaluation_bias
        
        # Make HIRING DECISION based on final score
        # recall we can ONLY see final_score, not true_skill!
        hired = (final_score > tau).astype(int) # 1 = hired, 0 = rejected
        
        # Store all the data for this group
        group_data = pd.DataFrame({
            'group': group,
            'true_skill': true_skill, # unobservable ground truth
            'observed_score': observed_score, # noisy evaluation
            'final_score': final_score, # what we actually see
            'true_qualified': actually_qualified, # hidden ground truth label
            'hire': hired # hiring decision based on observed
        })
        applicants.append(group_data)
    
    return pd.concat(applicants).reset_index(drop=True)

# regenerate button to clear cache
if st.sidebar.button("Generate new data", type="primary"):
    st.cache_data.clear()

with st.spinner("Generating simulation data..."):
    population_df = generate_synthetic_data(
        N_APPLICANTS, GROUP_QUALIFICATION_MEANS, GROUP_BIAS_LEVELS, Q_TARGET, HIRING_THRESHOLD_TAU, seed=SEED
    )

def calculate_fairness_metrics(df):
    metrics = {}
    for group, group_df in df.groupby('group'):
        total_applicants = len(group_df)

        # 4 possible outcomes in our confusion matrix:
        TP = ((group_df['hire'] == 1) & (group_df['true_qualified'] == True)).sum() # hire someone, they are qualified (good)
        FN = ((group_df['hire'] == 0) & (group_df['true_qualified'] == True)).sum() # REJECTED somone, they are qualified
        FP = ((group_df['hire'] == 1) & (group_df['true_qualified'] == False)).sum() # hired somone, they are NOT qualified
        TN = ((group_df['hire'] == 0) & (group_df['true_qualified'] == False)).sum() # REJECTED someone, they are NOT qualified (good)

        # false negative rate (FNR) = FN / (TP + FN): among truly qualified, fraction missed
        total_qualified = TP + FN
        if total_qualified > 0:
            FNR = FN / total_qualified
        else:
            FNR = np.nan 

        # false positive rate (FPR) = FP / (FP + TN): among truly unqualified, fraction mistakenly hired
        total_qualified = FP + TN
        if total_qualified > 0:
            FPR = FP / total_qualified 
        else:
            FPR = np.nan

        # predictive parity = TP / (TP + FP): quality of our hires (out of everyone hired, how many are qualified)
        total_hired = TP + FP
        if total_hired > 0:
            hire_quality = TP / total_hired 
        else:
            hire_quality = np.nan

        # overall acceptance rate = hired / total applicants
        if total_applicants > 0:
            acceptance_rate = total_hired / total_applicants 
        else:
            acceptance_rate = np.nan 

        # store all metrics for this group
        metrics[group] = {
            'acceptance rate': acceptance_rate,
            'false negative rate': FNR,
            'false positive rate': FPR,
            'hire quality': hire_quality,
            'num total': total_applicants
        }
    return pd.DataFrame(metrics).T

# Bayesian posterior analysis using numerical integration
def compute_posterior_distribution(df, group, tau, q_mean, q_std, bias_levels):
    # among the people we hired, what is their true skill level likely to be?
    # compute P(Q | Hire=1, group) = probability of true qualification (Q) given we hire an applicant from one of the groups (joint)

    group_data = df[df['group'] == group]
    hired = group_data[group_data['hire'] == 1]
    
    if len(hired) == 0:
        return None, None, None
    
    # EMPERICAL: just look at the actual data, what's the average true skill of hired people?
    empirical_mean = hired['true_skill'].mean()
    empirical_std = hired['true_skill'].std()
    
    # THEORETICAL:
    # applying Bayes: P(Q | Hire=1, group) = P(Q | Hire=1, group) = P(Hire=1 | Q, group) * P(Q | group) / P(Hire=1 | group)
        # P(Q | group) PRIOR, Normal(q_mean, q_std)
        # P(Hire=1 | Q, group) is the LIKELIHOOD of hiring (observable score E + epislon exceeding tau, our hiring threshold) given a true qualification Q
        # P(Hire=1 | group) is the overall hiring rate for that group 

    # P(Q | group)
    def prior(true_skill):
        return norm.pdf(true_skill, loc=q_mean, scale=q_std)
    
    # P(Hire=1 | Q, group)
    def likelihood(true_skill):
        def probability_calculation(observed_score):
            prob = norm.pdf(observed_score, loc=true_skill, scale=OBSERVABLE_SCORE_STDEV)
            # hiring rule: final_score = observed_score + bias > tau, or bias > (tau - observed_score)
            # since bias ~ Logistic(0, bias_levels): P(bias > tau - observed_score) = 1 - CDF(tau - observed_score)
            # P(epsilon > tau - e) where epsilon ~ Logistic(0, bias_levels)
            hire = 1 - logistic.cdf(tau - observed_score, loc=0, scale=bias_levels)
            # P(hired AND observed_score | true_skill) = P(hired | observed_score) * P(observed_score | true_skill)
            return prob * hire
        
        # LOTP: P(hired | true_skill) = ∫ P(hired | test_score) × P(test_score | true_skill) d(test_score)                                     ↑        #
        # i.e. we are summing over ALL possible test_scores (weighted by their probability)
        total_probability, _ = quad(probability_calculation, # note: quad = numerical integration function from scipy.integrate
                               true_skill - 5*OBSERVABLE_SCORE_STDEV, # lower bound
                               true_skill + 5*OBSERVABLE_SCORE_STDEV) # upper bound
        return total_probability

    # similarly, compute evidence P(Hire=1) by normalizing over Q true skills
    def evidence_integrand(true_skill):
        return likelihood(true_skill) * prior(true_skill)
    # integrate over reasonable range of Q
    evidence, _ = quad(evidence_integrand, q_mean - 5*q_std, q_mean + 5*q_std)
    
    def posterior(true_skill):
        if evidence > 0:
            # P(Q | Hire=1) = P(Hire=1 | Q) × P(Q) / P(Hire=1) = likelihood × prior / evidence
            return likelihood(true_skill) * prior(true_skill) / evidence
        return 0
    
    # compute posterior mean (avg true skill of people hired)
    # posterior mean > prior mean = hired people better than average
    # < = hired worse than average people
    def posterior_mean_integrand(true_skill):
        return true_skill * posterior(true_skill)
    
    # add up all the weighted contributions across all possible skill levels (skill * probability, as calculated in function posterior_mean_integrand)
    theoretical_mean, _ = quad(posterior_mean_integrand, q_mean - 5*q_std, q_mean + 5*q_std)
    
    # if our math is correct, theoretical_mean should be close to empirical_mean!
    return empirical_mean, empirical_std, theoretical_mean

# WRAPPER FUNCTION: generates datatable comparing all groups 
def bayesian_posterior_analysis(df, tau, q_means, bias_levels):
    results = {} 

    # analyze each group separately (A, B, C)
    for group, group_df in df.groupby('group'):

        # calculate PRIOR 
        prior_mean = group_df['true_skill'].mean() # avg skill of everyone who applies 
        prior_std = group_df['true_skill'].std() # spread of skill level

        # calculate POSTERIOR (look at only the people hired)
        hired_Q = group_df[group_df['hire'] == 1]['true_skill'] 

        # Get three estimates of "average skill of hired people":
        emperical_mean, emperical_std, theoretical_mean = compute_posterior_distribution(
            df, group, tau, 
            q_means[group], # group's average skills
            GROUP_QUALIFICATION_STDEV, 
            bias_levels[group] 
        )
        # calculate SELECTION STRENGTH, i.e. how much better are hired people than the average applicant?
        if emperical_mean is not None:
            selection_strength = emperical_mean - prior_mean
            # + = we hired above-average people ✓
            # 0 = we hired randomly (no selection)
            # - = we hired below-average people (shouldn't happen!)
        else:
            selection_strength = np.nan

        results[group] = {
            'Prior Mean Q': prior_mean, # avg skill of ALL applicants
            'Prior Std Q': prior_std,
            'Posterior Mean Q (empirical)': emperical_mean,  # avtual avg of HIRED (from data)
            'Posterior Std Q (empirical)': emperical_std if emperical_std is not None else np.nan,
            'Posterior Mean Q (Bayes)': theoretical_mean if theoretical_mean is not None else np.nan, # predicted avg (from math)
            'N hired': len(hired_Q),  
            'Selection Strength': selection_strength 
        }
    # convert to a table w groups as rows
    return pd.DataFrame(results).T

# bootstrap resampling: simulate "1000 parallel universes" where we re-run the hiring process with slightly different applicant pools
def bootstrap_fairness_metrics(df, n_bootstrap=1000, sample_size=None, seed=42):
    # we want to meaasure the UNCERTAINTY in our fairness metrics
    # WHY?? our metrics (FNR, FPR, etc.) are based on ONE random sample of applicants. if we ran the hiring process again with different applicants, would we get the same metrics?
    
    random_generator = np.random.default_rng(seed)
    
    # if not specified resample same size as og dataset
    if sample_size is None:
        sample_size = len(df)
    
    # store results from each bootstrap sample
    all_bootstrap_samples = {
        # structure: {group A: {FNR: [0.78, 0.79, 0.77, ...], FPR: [...], ...}, ...}
        group: {'FNR': [], 'FPR': [], 'HQ': [], 'acceptance': []} 
        for group in ['A', 'B', 'C']
    }
    
    # run 1000 simulated hiring processes
    for i in range(n_bootstrap):
        # resample by randomly picking applicants w REPLACEMENT (allows duplicates)
        resampled_applicants = df.sample(n=sample_size, replace=True)
        
        # calculate/save fairness metrics for THIS resampled dataset
        metrics_for_this_sample = calculate_fairness_metrics(resampled_applicants)
        
        for group in ['A', 'B', 'C']:
            if group in metrics_for_this_sample.index:
                all_bootstrap_samples[group]['FNR'].append(metrics_for_this_sample.loc[group, 'false negative rate'])
                all_bootstrap_samples[group]['FPR'].append(metrics_for_this_sample.loc[group, 'false positive rate'])
                all_bootstrap_samples[group]['HQ'].append(metrics_for_this_sample.loc[group, 'hire quality'])
                all_bootstrap_samples[group]['acceptance'].append(metrics_for_this_sample.loc[group, 'acceptance rate']
                )
    
    # calculate confidence intervals from the distributions    
    # tells us "we're X% confident the TRUE FNR for group A is between Y and Z"
    confidence_intervals = {}
    
    for group in ['A', 'B', 'C']:
        confidence_intervals[group] = {}
        
        for metric_name in ['FNR', 'FPR', 'HQ', 'acceptance']:
            # get all 1000 values for this metric
            metric_values = np.array(all_bootstrap_samples[group][metric_name])
            metric_values = metric_values[~np.isnan(metric_values)] # remove any NaN values
            
            # calculate summary statistics
            mean_value = np.mean(metric_values)

            lower_bound = np.percentile(metric_values, 2.5) # 2.5th percentile (cut off bottom 2.5%)
            upper_bound = np.percentile(metric_values, 97.5) # 97.5th percentile (cut off top 2.5%)

            confidence_intervals[group][f'{metric_name}_mean'] = mean_value
            confidence_intervals[group][f'{metric_name}_ci_lower'] = lower_bound
            confidence_intervals[group][f'{metric_name}_ci_upper'] = upper_bound
    
    return all_bootstrap_samples, confidence_intervals

# analyze how fairness metrics change across different threshold values.
def threshold_analysis(df, q_target, tau_range=np.linspace(3, 10, 50)):
    results = {g: {'tau': [], 'acceptance': [], 'FNR': [], 'FPR': [], 'HQ': []} 
               for g in ['A', 'B', 'C']}
    
    for tau in tau_range:
        # recompute hiring decisions with new threshold
        df_temp = df.copy()
        df_temp['hire'] = (df_temp['final_score'] > tau).astype(int)
        metrics = calculate_fairness_metrics(df_temp)
        
        # update group data
        for group in ['A', 'B', 'C']:
            if group in metrics.index:
                results[group]['tau'].append(tau)
                results[group]['acceptance'].append(metrics.loc[group, 'acceptance rate'])
                results[group]['FNR'].append(metrics.loc[group, 'false negative rate'])
                results[group]['FPR'].append(metrics.loc[group, 'false positive rate'])
                results[group]['HQ'].append(metrics.loc[group, 'hire quality'])

    return results

# plot fairness metrics
st.subheader("Fairness metrics")

# description
st.markdown("""
*Measure how our hiring algorithm treats different groups using standard fairness criteria using 4 key metrics (explained below).*
            
*Why it matters*: Even with identical thresholds, different groups can experience vastly different error rates, revealing where bias enters the system and who bears the cost (spoiler: you can't optimize all of them simultaneously!)
""")

fairness_df = calculate_fairness_metrics(population_df)
st.dataframe(fairness_df.style.format("{:.4f}"), use_container_width=True)

st.markdown("""
**4 METRICS:**
- **Acceptance Rate**: % of applicants hired from each group
  - Higher = easier to get hired from this group
- **False Negative Rate (FNR)**: % of qualified people we wrongly rejected
  - Higher = more missed talent (e.g. group A: 0.50 = missed 50% of qualified applicants!)
- **False Positive Rate (FPR)**: % of unqualified people we wrongly hired
  - Higher = more bad hires slipping through (e.g. group C: 0.20 = 20% of unqualified got hired)
- **Hire Quality**: % of our hires who are actually qualified
  - Higher = better quality hires (0.85 = 85% of Group A hires are qualified)
""")

# acceptance rates bar plot
fig1, ax1 = plt.subplots(figsize=(7, 4))
fairness_df['acceptance rate'].plot(kind='bar', ax=ax1, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
ax1.set_title("Acceptance rate by group")
ax1.set_ylabel("Rate")
ax1.set_ylim(0, 1)
ax1.set_xticklabels(fairness_df.index, rotation=0)
ax1.grid(axis='y', alpha=0.3)
st.pyplot(fig1) 

st.subheader("Bayesian posterior (selection) analysis")
# description
st.markdown("""
*Reveals the hidden selection bias through prior vs posterior comparison: compares who applies vs. who actually gets hired to expose disparate impact.*

*Why it matters:* Even with identical hiring thresholds, different groups face vastly different selection pressures. This analysis quantifies how much "better than average" someone needs to be to get hired from each group by. Therefore, the goal of this analysis is to:
- Show which groups face stricter filtering (higher selection strength = harder bar to clear)
- Compare our theoretical predictions (Bayes math) against actual outcomes (empirical data)
- Reveal structural unfairness: e.g. Group A might need to be +2.0 points above average to get hired, while Group C only needs +0.5
""")

# run computation (pure function)
posterior_df = bayesian_posterior_analysis(population_df, HIRING_THRESHOLD_TAU, 
                                            GROUP_QUALIFICATION_MEANS, GROUP_BIAS_LEVELS)
# display the table + variable descriptions
st.dataframe(posterior_df.style.format("{:.4f}"), use_container_width=True)
st.markdown("""
**VARIABLE MEANINGS:**
- **Prior Mean Q**: average skill of all applicants (before hiring)
- **Posterior Mean Q (empirical)**: average skill of people we hired (actual data)
- **Posterior Mean Q (Bayes)**: average skill of hired people (predicted by math)
- **Selection Strength**: how much better are hires vs. average applicant?
  - higher = stricter filter (e.g. Group A: +1.2 means 1.2 points above average needed)
  - lower = easier filter (e.g. Group C: +0.7 means only 0.7 points above average needed)
""")

# bar plot of prior vs posterior means
fig2, ax2 = plt.subplots(figsize=(7, 4))
x = np.arange(len(posterior_df))
width = 0.25
ax2.bar(x - width, posterior_df['Prior Mean Q'], width, label='Skill of all applicants (avg)', alpha=0.7)
ax2.bar(x, posterior_df['Posterior Mean Q (empirical)'], width, label='Skill of hired people (atual)', alpha=0.7)
ax2.bar(x + width, posterior_df['Posterior Mean Q (Bayes)'], width, label='Skill of hired people (predicted)', alpha=0.7)
ax2.axhline(y=Q_TARGET, color='red', linestyle='--', linewidth=2, label=f'Qualification threshold = {Q_TARGET}')
ax2.set_xticks(x)
ax2.set_xticklabels(posterior_df.index)
ax2.set_ylabel("True skill level")
ax2.set_title("Selection effect")
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
st.pyplot(fig2)

st.markdown("""*Note unfairness*: different groups face different selection pressures despite same threshold!""")

# distributions
st.subheader("True skill vs. Observed score distributions")
st.markdown("""
*Visualizes the gap between reality and what the algorithm sees—revealing how bias distorts hiring decisions.*

**LEGEND:**
- **Brown histogram**: True hidden skill (unobservable in real life)
- **Orange histogram**: Final scores after bias is added (what we actually see and use for hiring)
- **Red line**: Qualification threshold (7.0) - people above this are truly qualified
- **Green line**: Hiring threshold (7.5) - people above this get hired
""")

fig3, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, group_name in enumerate(['A', 'B', 'C']):
    group_data = population_df[population_df['group'] == group_name]
    axes[i].hist(group_data['true_skill'], bins=50, alpha=0.5, label='True Skill (unobserved)', color='blue')
    axes[i].hist(group_data['final_score'], bins=50, alpha=0.5, label='Observed Score (biased)', color='orange')
    axes[i].axvline(Q_TARGET, color='red', linestyle='--', linewidth=2, label=f'Skill target = {Q_TARGET}')
    axes[i].axvline(HIRING_THRESHOLD_TAU, color='green', linestyle='--', linewidth=2, label=f'Hiring threshold = {HIRING_THRESHOLD_TAU}')
    axes[i].set_title(f'group {group_name}')
    axes[i].set_xlabel("Score (1-10 scale)")
    axes[i].set_ylabel("Number of Applicants")
    # axes[i].legend()
    axes[i].grid(axis='y', alpha=0.3)
plt.tight_layout()
st.pyplot(fig3)

st.markdown("""
*Some observations:* note that we hire based on the orange (biased) distribution, not the blue (true skill)!
- **Group A**: orange spread far from blue -> high bias creates gap between true skill + observed score
- **Group C**: orange closely tracks blue -> low bias means observed scores reflect true skill better
""")

# summary metrics
st.subheader("Summary Stats")
col1, col2, col3 = st.columns(3)
for col, group_name in zip([col1, col2, col3], ['A', 'B', 'C']):
    group_data = population_df[population_df['group'] == group_name]
    with col:
        st.metric(f"Group {group_name}: Total Applicants", len(group_data))
        st.metric(f"Group {group_name}: Hired", int(group_data['hire'].sum()))
        st.metric(f"Group {group_name}: Actually Qualified", int(group_data['true_qualified'].sum()))

# display bootstrap analysis
st.subheader("Bootstrap analysis (uncertainty quantification)")
st.markdown("""
*Estimates the sampling distribution of fairness metrics through resampling to quantify uncertainty.* 
            
What problem is this solving? Well, our fairness metrics (FNR, FPR, etc.) are calculated from a single random sample of applicants. If we ran the hiring process again with a different applicant pool, would we get the same metrics? How confident should we be in these numbers?

**Steps**:
1. *Resample with replacement*: Randomly draw N applicants from our dataset (allowing duplicates)
2. *Recalculate metrics*: Compute FNR, FPR, hire quality, acceptance rate for this resampled dataset
3. *Repeat 1000 times*: Generate 1000 alternative "universes" with slightly different applicant pools
4. *Build distributions*: Each metric now has 1000 values forming an empirical sampling distribution
5. *Compute confidence intervals*: Use percentile method (2.5th and 97.5th percentiles) for 95% CIs

Without this, we only have point estimates (single numbers). Bootstrap tells us: "We're X% confident the true FNR for Group A is between Y and Z," revealing whether observed disparities are real or just noise from random sampling!
""")
st.markdown("*Computing 95% confidence intervals using 1000 bootstrap samples.*")

with st.spinner("Running bootstrap analysis (this may take a sec)..."):
    bootstrap_dist, ci_results = bootstrap_fairness_metrics(population_df, n_bootstrap=1000, seed=SEED)

# display confidence intervals
ci_df_data = {}
for group in ['A', 'B', 'C']:
    ci_df_data[group] = {
        'FNR (95% CI)': f"{ci_results[group]['FNR_mean']:.3f} [{ci_results[group]['FNR_ci_lower']:.3f}, {ci_results[group]['FNR_ci_upper']:.3f}]",
        'FPR (95% CI)': f"{ci_results[group]['FPR_mean']:.3f} [{ci_results[group]['FPR_ci_lower']:.3f}, {ci_results[group]['FPR_ci_upper']:.3f}]",
        'HQ (95% CI)': f"{ci_results[group]['HQ_mean']:.3f} [{ci_results[group]['HQ_ci_lower']:.3f}, {ci_results[group]['HQ_ci_upper']:.3f}]",
    }
ci_df = pd.DataFrame(ci_df_data).T
st.dataframe(ci_df, use_container_width=True)

st.markdown("""
**How to read this table**
- *Mean*: Best estimate of the true metric value across many hypothetical hiring processes
- *95% CI [lower, upper]*: We're 95% confident the true value falls in this range

*e.g.: If Group A has FNR = 0.482 [0.465, 0.499] and Group C has FNR = 0.152 [0.138, 0.166], these intervals don't overlap, proving Group A faces significantly higher false rejection rates, not by chance.*
""")

# plot bootstrap distributions
fig_boot, axes = plt.subplots(2, 2, figsize=(15, 10))
metrics_to_plot = [('FNR', 'false negative rate'), ('FPR', 'false positive rate'), 
                    ('HQ', 'hire quality'), ('acceptance', 'acceptance rate')]

for idx, (metric, title) in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]
    for group in ['A', 'B', 'C']:
        values = np.array(bootstrap_dist[group][metric])
        values = values[~np.isnan(values)]
        ax.hist(values, bins=30, alpha=0.5, label=f'group {group}')
    ax.set_xlabel(title)
    ax.set_ylabel('Frequency')
    ax.set_title(f'Bootstrap Distribution: {title}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
st.pyplot(fig_boot)

st.markdown("""
Each histogram shows the distribution of a metric across 1000 resampled datasets (1000 "parallel universes").
- X-axis: metric value (FNR, FPR, hire quality, or acceptance rate)
- Y-axis: frequency (how often we observed that value across 1000 bootstrap samples)
- Each colored distribution = one demographic group (A, B, C)

**Interpreting shape**: Tall, narrow peaks indicate a consistent metric across resamples, so it is a reliable estimate (and vice cersa for wider peaks). Separated peaks indicate that groups experience genuinely different outcomes (not just sampling noise).
            Overlapping distributions mean that the differences might be due to random chance.

""")

# display threshold analysis
st.subheader("Threshold analysis")
st.markdown("*How do fairness metrics change as we vary the hiring threshold τ? The impossible choice is visible in these graphs!*")

st.markdown("""
- Lower τ (move left): Group A acceptance improves, hire quality drops
- Higher τ (move right): hire quality improves, Group A acceptance plummets
- **Lines never converge**: NO SINGLE THRESHOLD achieves fairness for all groups

Each plot shows how a metric changes as hiring threshold τ varies (3 to 10). Try adjusting your hiring threshold (sidebar) and watch the red line move!
""")

with st.spinner("Analyzing threshold trade-offs..."):
    tau_range = np.linspace(3, 10, 50)
    threshold_results = threshold_analysis(population_df, Q_TARGET, tau_range)

# create threshold analysis plots
fig_thresh, axes = plt.subplots(2, 2, figsize=(15, 10))

# acceptance rate vs tau
ax = axes[0, 0]
for group in ['A', 'B', 'C']:
    ax.plot(threshold_results[group]['tau'], threshold_results[group]['acceptance'], 
            label=f'group {group}', linewidth=2)
ax.axvline(x=HIRING_THRESHOLD_TAU, color='red', linestyle='--', label='Current τ')
ax.set_xlabel('Hiring Threshold (τ)')
ax.set_ylabel('Acceptance Rate')
ax.set_title('Acceptance Rate vs Threshold')
ax.legend()
ax.grid(True, alpha=0.3)

# FNR vs tau
ax = axes[0, 1]
for group in ['A', 'B', 'C']:
    ax.plot(threshold_results[group]['tau'], threshold_results[group]['FNR'], 
            label=f'group {group}', linewidth=2)
ax.axvline(x=HIRING_THRESHOLD_TAU, color='red', linestyle='--', label='Current τ')
ax.set_xlabel('Hiring Threshold (τ)')
ax.set_ylabel('False Negative Rate')
ax.set_title('FNR vs Threshold (Missing Qualified Applicants)')
ax.legend()
ax.grid(True, alpha=0.3)

# FPR vs tau
ax = axes[1, 0]
for group in ['A', 'B', 'C']:
    ax.plot(threshold_results[group]['tau'], threshold_results[group]['FPR'], 
            label=f'group {group}', linewidth=2)
ax.axvline(x=HIRING_THRESHOLD_TAU, color='red', linestyle='--', label='Current τ')
ax.set_xlabel('Hiring Threshold (τ)')
ax.set_ylabel('False Positive Rate')
ax.set_title('FPR vs Threshold (Hiring Unqualified Applicants)')
ax.legend()
ax.grid(True, alpha=0.3)

# hiring quality (HQ) vs tau
ax = axes[1, 1]
for group in ['A', 'B', 'C']:
    ax.plot(threshold_results[group]['tau'], threshold_results[group]['HQ'], 
            label=f'group {group}', linewidth=2)
ax.axvline(x=HIRING_THRESHOLD_TAU, color='red', linestyle='--', label='Current τ')
ax.set_xlabel('Hiring Threshold (τ)')
ax.set_ylabel('Hire Quality')
ax.set_title('Hire Quality vs Threshold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig_thresh)

st.markdown("""
**LEGEND:**
- Red line: your current threshold setting
- Steeper curves: more sensitive to threshold changes

            """)

# logistic regression demo
st.markdown("### Logistic Regression")
st.markdown("""
*Train a simple LR model to predict qualification from biased scores, demonstrating how algorithms learn and perpetuate existing bias. 
            Can we predict if someone is `Actually_Qualified` using only their biased `Final_Score` (and therefore use this LR model to hire)?*

**About the model:**
- **Input (X)**: Continuous real number representing the biased observable score (what the algorithm sees)
- **Output (y)**: True qualification status, a binary 0 or 1 (what we're trying to predict!)
- *Learning Process*: Model adjusts its weights through **gradient descent** to minimize prediction error
    - i.e. learns P(qualified=1) = sigmoid(weight × final_score + bias)
- *Result*: sigmoid curve that converts scores -> probabilities of being qualified
            
Click on "Run LR" (change learning rate and epochs in the sidebar)!
""")

run_logistic_regression = st.checkbox("Run logistic regression", value=True)

# sigmoid function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# loss measured how wrong our predictions are (smaller = better)
def binary_cross_entropy_loss(y_true, y_predicted, epsilon=1e-12):
    # log(0) = -infinity
    y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon)
    
    # calculate loss for each prediction, then average
    # loss = -(1/n) Σ [y_true × log(y_pred) + (1-y_true) × log(1-y_pred)]
    loss = -np.mean(
        y_true * np.log(y_predicted) + # loss when y_true=1
        (1 - y_true) * np.log(1 - y_predicted) # loss when y_true=0
    )
    return loss

# train LR using gradient descent
def train_logistic_regression(features, labels, learning_rate=0.1, num_iterations=1000, verbose=False):
    # probability = sigmoid(weight × feature + bias)
    
    # init params to zero
    weight = 0.0
    bias = 0.0
    loss_history = []

    num_samples = len(features)

    for i in range(num_iterations):
        # step 1: make prediction
        # linear comb
        z = weight * features + bias
        # convert to probabilities using sigmoid(z)
        predicted_probabilities = sigmoid(z)
        
        # step 2: calculate loss
        current_loss = binary_cross_entropy_loss(labels, predicted_probabilities)
        loss_history.append(current_loss)
        
        # step 3: calculate gradients (chain rule):
            # dLoss/dweight = (1/n) Σ (predicted - actual) × feature
            # dLoss/dbias = (1/n) Σ (predicted - actual)
    
        # error = predicted - actual (positive if overestimated, negative if underestimated)
        prediction_error = predicted_probabilities - labels
        gradient_weight = np.mean(prediction_error * features)
        gradient_bias = np.mean(prediction_error)
        
        # step 4: update params
        # move in the OPPOSITE direction of the gradient (gradient descent!)
            # positive -> dec parameter
            # negative -> inc parameter
        weight = weight - learning_rate * gradient_weight
        bias = bias - learning_rate * gradient_bias
        
    return weight, bias, loss_history

# run LR
if run_logistic_regression:
    st.markdown("### Training Setup")
    
    st.markdown("""

    **Gradient Descent (learning algorithm)**: the model starts with random guesses (we initalize all w=0, b=0), then iteratively improves them as follows.
                
    Step 1: **Make a prediction:**
    - Compute: `z = w × final_score + b` (linear combination)
    - Apply sigmoid: `ŷ = 1/(1 + e^(-z))` (convert to probability between 0 and 1)
    - OUPUT: predicted probability that the label `Actually_Qualified = 1` (LR assumption)

    Step 2: **Calculate loss (error function)**:
    - Binary Cross-Entropy function: `Loss = −(1/n) Σ [y×log(ŷ) + (1−y)×log(1−ŷ)]`
        - This penalizes confident wrong predictions heavily

    Step 3: **Compute Gradients (Which Direction to Move)**:
    - `∂Loss/∂w = (1/n) Σ (ŷ − y) × final_score` (how much does changing w affect loss?)
    - `∂Loss/∂b = (1/n) Σ (ŷ − y)` (how much does changing b affect loss?)
    - (+) gradient = parameter too high, (-) = parameter too low

    Step 4: **Update parameters**:
    - `w_new = w_old − learning_rate × ∂Loss/∂w`
    - `b_new = b_old − learning_rate × ∂Loss/∂b`
        - move **opposite** the gradient direction (gradient descent!)
    """)

    # sample a subset (faster training)
    sample_size = 4000
    training_data = population_df.sample(n=sample_size, random_state=SEED)
    
    # extract features (X) and binary labels (y)
        # X = final_score (the biased observable score)
        # y = actually_qualified (0 = not qualified, 1 = qualified)
    X_features = training_data['final_score'].values
    y_labels = training_data['true_qualified'].astype(int).values
    
    st.write(f"Training on {sample_size} applicants (using a sample size so that it is a bit faster!!)")
    
    # hyperparameters
    learning_rate = st.sidebar.slider("Learning rate step size", 0.001, 1.0, 0.1, 0.001)
    num_iterations = st.sidebar.slider("Number of training iterations", 10, 5000, 500, 10)
    
    # train LR model!
    with st.spinner("Training logistic regression model..."):
        learned_weight, learned_bias, training_losses = train_logistic_regression(
            X_features, y_labels, 
            learning_rate=learning_rate, 
            num_iterations=num_iterations, 
            verbose=False
        )

    # show results 
    # description   
    st.markdown("### Learned Parameters")
    st.markdown("""
    **What these numbers mean:**
    - **Weight (w)**: how strongly `final_score` influences the prediction
        - (+)) w: Higher scores → higher probability of being qualified ✓
        - (-) |w|: Steeper sigmoid curve (more confident decisions)
    - **Bias (b)**: model's baseline "pessimism" or "optimism"
        - (-) b: Model assumes people are unqualified by default (needs high scores to predict qualified)
        - (+) b: Model is more lenient (predicts qualified more easily)
    
    *e.g.: if w=4 and b=-3, the model says: "each point of `final_score adds` +4 to my confidence, but I start very skeptical (−3), so you need a score around 8+ before I think you're qualified!"*
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Weight (w)", f"{learned_weight:.4f}")
        st.caption("How much does final_score affect the hiring prediction?")
    with col2:
        st.metric("Bias (b)", f"{learned_bias:.4f}")
        st.caption("Baseline prediction when final_score = 0")
    
    # plot training loss curve
    st.markdown("### Training Progress (Loss)")
    fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
    ax_loss.plot(training_losses, linewidth=2, color='#ff6b6b')
    ax_loss.set_xlabel("Iteration", fontsize=12)
    ax_loss.set_ylabel("Loss (Binary Cross-Entropy)", fontsize=12)
    ax_loss.set_title("Loss (Error) Decreases as Model Learns!", fontsize=14, fontweight='bold')
    ax_loss.grid(True, alpha=0.3)
    ax_loss.axhline(y=training_losses[-1], color='green', linestyle='--', 
                    label=f'Final Loss: {training_losses[-1]:.4f}')
    ax_loss.legend()
    st.pyplot(fig_loss)
        
    # evaluating accuracy
    st.markdown("### Model Performance")
    st.markdown("""
    **Calculating accuracy**:
    1. Convert probabilities to predictions: if ŷ > 0.5 → predict "qualified" (1), else predict "not qualified" (0)
    2. Compare to ground truth: check if prediction matches `actually_qualified` label
    3. `Accuracy = (# correct predictions) / (total predictions)`

    *Note that high accuracy ≠ fair! Model can be 90% accurate but still systematically discriminate against certain groups.*
    """)
    # make predictions on the training data
    predicted_probabilities = sigmoid(learned_weight * X_features + learned_bias)
    # convert probabilities to binary predictions (threshold = 0.5)
    predicted_labels = (predicted_probabilities > 0.5).astype(int)

    # calculate accuracy
    correct_predictions = (predicted_labels == y_labels)
    accuracy = correct_predictions.mean()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.1%}")
    with col2:
        st.metric("Correct Predictions", f"{correct_predictions.sum()} / {len(y_labels)}")
    with col3:
        st.metric("Final Loss", f"{training_losses[-1]:.4f}")
    
    # show learned decision boundary
    st.markdown("### Learned LR decision boundary")
    st.markdown("""
    *The model's learned relationship between biased scores and predicted qualification probability. This curve reveals what the model "thinks," or which scores it associates with qualification. If trained on biased data, this curve encodes that bias!*

    **About the curve!** (sigmoid = model's confidence function)
    - *Steep*: model is confident, small score changes flip predictions dramatically
    - *Gradual*: model is uncertain, needs bigger score differences to change predictions
    - **Inflection point** (where curve is steepest): the score where model is maximally uncertain (≈ 50% probability), called the decision boundary!
    """)
    fig_boundary, ax_boundary = plt.subplots(figsize=(10, 6))
    
    # plot the sigmoid curve
    score_range = np.linspace(X_features.min(), X_features.max(), 200)
    predicted_probs = sigmoid(learned_weight * score_range + learned_bias)
    
    ax_boundary.plot(score_range, predicted_probs, linewidth=3, color='blue', 
                     label='P(Qualified | final_score)')
    ax_boundary.axhline(y=0.5, color='red', linestyle='--', linewidth=2, 
                        label='Decision Threshold (50%)')
    ax_boundary.axvline(x=HIRING_THRESHOLD_TAU, color='green', linestyle='--', linewidth=2,
                        label=f'Hiring Threshold τ = {HIRING_THRESHOLD_TAU}')
    
    ax_boundary.set_xlabel("Final score (biased observable)", fontsize=12)
    ax_boundary.set_ylabel("Predicted probability of being qualified", fontsize=12)
    ax_boundary.set_title("Model's learned relationship", fontsize=14, fontweight='bold')
    ax_boundary.grid(True, alpha=0.3)
    ax_boundary.legend(fontsize=10)
    ax_boundary.set_ylim(-0.05, 1.05)
    
    st.pyplot(fig_boundary)
    
    st.markdown("""
    **LEGEND:**
    - X-axis: biased `final_score` (what the model sees)
    - Y-axis: model's predicted probability someone is qualified
    - Blue curve: learned sigmoid function
    - Red dashed line: 50% threshold (above = predict qualified, below = predict unqualified)
    - Green line: actual hiring threshold τ
    """)

st.markdown("---")    

st.markdown("""
### THE BIG IDEA.
**Equal treatment/threshold ≠ equal outcomes.** 
When groups have different levels of measurement bias (due to structural inequality), a single threshold creates a system where some groups need to be higher above average to get hired relative to more advantaged groups. Some groups are disproportionately rejected (and at higher error rates). 
            
Real-world hiring algorithms face the same problem: résumé screening tools, interview scoring systems, and predictive models all rely on noisy, biased proxies for true ability. This simulation is pure computation, but still reveals how even "objective" systems can be deeply unfair.
            
**So... is fairness impossible?** Hopefully, you discovered that there's no single threshold that's fair to all groups. WHY?
""")

st.markdown("""
#### The Fairness Impossibility Theorem!
As demonstrated by Hsu et al. (2022), when groups have different skill distributions AND face different measurement biases, you cannot simultaneously achieve:
- *Demographic Parity*: Equal acceptance rates (everyone hired at same %)
- *Equalized Odds*: Equal error rates (same FNR and FPR across groups)
- *Predictive Parity*: Equal hire quality (same % of hires actually qualified)
            
There is no "neutral" solution; only value judgments about who deserves protection from which types of mistakes (false rejections vs. false acceptances).
            
**Real hiring systems face this exact dilemma.** The question isn't "how do we make it fair?". It is "which definition of fair do we prioritize, and **who pays the price?**"
""")

st.markdown("""
**Disclaimer about LLM usage:** I used ChatGPT to help me with the streamlit framework, including syntax for interactive widgets (sliders, checkboxes, buttons), layout components (sidebar, columns, tabs), and the dataframes/visuals. It was especially helpful in debugging integration issues between scipy numerical methods and Streamlit rendering
            I also had it help me write intuitive descriptions that clarify mathematical concepts for non-technical users. All probability concepts (Bayesian inference, bootstrap, fairness metrics) were derived from CS 109 course material. I hand-derived my likelihood functions (aside from the integration in the code!), posterior derivations, gradient descent equations, bootstrapping logic, etc.!
""")