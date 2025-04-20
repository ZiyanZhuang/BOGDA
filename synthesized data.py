"""
Synthetic Dataset Generator for Pest Resistance Analysis
Author: Ziyan Zhuang, Qiao Sheng and Jingang Xie.
License: MIT
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, ks_2samp  # Statistical functions
from scipy.optimize import minimize  # Optimization
from sklearn.metrics import mean_squared_error  # Model evaluation
import matplotlib.pyplot as plt  # Visualization
import seaborn as sns  # Enhanced visualization
import os  # File system operations

# Configure numerical computation environment
np.seterr(all='raise')  # Convert warnings to exceptions for debugging
pd.options.mode.chained_assignment = None  # Disable pandas chained assignment warning


# -------------------- Data Loading & Cleaning -------------------- #
def safe_float_convert(x):
    """Safely convert string to float with locale awareness

    Args:
        x: Input value (string or number)

    Returns:
        float: Converted value or NaN if conversion fails
    """
    try:
        return float(str(x).replace(',', '.'))  # Handle European decimal commas
    except:
        return np.nan  # Return NaN for invalid values


def load_and_clean_data():
    """Intelligent data loader with automatic demo data fallback

    Returns:
        tuple: (df_a, df_b) where:
            df_a: LC50 experimental data
            df_b: Mortality observation data
    """
    try:
        # Attempt to load real experimental data
        df_a = pd.read_excel("A.xlsx", sheet_name="Sheet1")  # LC50 data
        df_b = pd.read_excel("B.xlsx", sheet_name="Sheet1")  # Mortality data
        print("Successfully loaded real datasets")

        # Data cleaning pipeline
        # Standardize LC50 string format (Chinese parentheses, spaces)
        df_a['LC50(CI95% )(mg/L)'] = df_a['LC50(CI95% )(mg/L)'].apply(
            lambda x: str(x).replace('）', ')').replace('（', '(').replace(' ', '').strip()
        )

        # Convert concentration values safely
        df_b['concentration(mg/l)'] = df_b['concentration(mg/l)'].apply(safe_float_convert)
        df_b = df_b.dropna(subset=['concentration(mg/l)'])  # Remove invalid rows

    except Exception as e:
        print(f"Data loading failed: {str(e)}, generating demo datasets...")
        df_a, df_b = generate_demo_data()  # Fallback to synthetic data

    return df_a, df_b


# -------------------- Demo Data Generation -------------------- #
def generate_demo_data():
    """Generate numerically stable demonstration datasets

    Returns:
        tuple: (demo_a, demo_b) synthetic datasets with:
            demo_a: Simulated LC50 measurements with confidence intervals
            demo_b: Simulated mortality observations
    """
    np.random.seed(42)  # For reproducible results

    # Generate Dataset A (LC50 measurements)
    base_lc50 = 0.1  # Baseline toxicity value
    demo_a = pd.DataFrame({
        'Generations': range(1, 21),  # 20 generations
        'LC50(CI95% )(mg/L)': [
            # Generate realistic LC50 values with confidence intervals
            f"{base_lc50 * (1.2 ** i):.3f}({base_lc50 * (1.2 ** i) * 0.8:.3f}-{base_lc50 * (1.2 ** i) * 1.2:.3f})"
            for i in range(20)
        ],
        # Generate slope values with standard errors
        'Slope±SE': [f"{1.5 + 0.1 * i:.1f}±0.2" for i in range(20)]
    })

    # Generate Dataset B (Mortality observations)
    concentrations = np.geomspace(0.1, 10, num=10)  # Logarithmic concentration range
    demo_b = pd.DataFrame({
        'generations': np.repeat(range(1, 21), 10),  # 10 concentrations per generation
        'concentration(mg/l)': np.tile(concentrations, 20),  # Repeat concentrations
        'insects': np.random.randint(25, 30, 200),  # Number of test subjects
        # Simulate mortality using binomial distribution
        'dead number': np.random.binomial(
            25,  # Number of trials
            np.clip(norm.cdf(np.random.normal(0, 0.5, 200)), 0.1, 0.9)  # Mortality probability
        )
    })

    return demo_a, demo_b


# -------------------- Core Calculation Functions -------------------- #
def calculate_params(row):
    """Calculate log-normal parameters from LC50 string with CI

    Args:
        row: DataFrame row containing LC50(CI95%) string

    Returns:
        tuple: (mu, sigma) parameters for log-normal distribution
    """
    lc50_str = row['LC50(CI95% )(mg/L)']
    try:
        # Parse confidence interval string
        parts = lc50_str.split('(')
        val = safe_float_convert(parts[0])  # Central value
        ci_part = parts[1].split(')')[0]  # Confidence interval part
        lower, upper = map(safe_float_convert, ci_part.split('-'))  # CI bounds

        # Calculate log-normal parameters
        mu = (np.log(lower) + np.log(upper)) / 2  # Mean of log values
        sigma = (np.log(upper) - np.log(lower)) / 3.92  # 95% CI → σ (1.96 * 2≈3.92)
        return mu, sigma
    except Exception as e:
        print(f"Parameter calculation error: {lc50_str} - {str(e)}")
        return np.log(0.1), 0.2  # Return safe default values


def extract_slope(slope_str):
    """Robustly extract slope value from string with possible errors

    Args:
        slope_str: String containing slope value with possible error notation

    Returns:
        float: Extracted slope value or default if parsing fails
    """
    try:
        # Handle various string formats: "1.5±0.2" or "1.5(0.2)"
        return safe_float_convert(str(slope_str).split('±')[0].split('(')[0])
    except:
        return 2.0  # Return biologically plausible default


# -------------------- Optimization Objective -------------------- #
def stable_norm_cdf(x):
    """Numerically stable normal CDF with clipping

    Args:
        x: Input values

    Returns:
        array: CDF values clipped to avoid numerical instability
    """
    return norm.cdf(np.clip(x, -10, 10))  # Clip extreme values


def objective_function(params, gen, df_b_true):
    """Optimization objective function combining MSE and KS statistic

    Args:
        params: [mu, sigma, slope] parameters to optimize
        gen: Current generation being processed
        df_b_true: Ground truth mortality data

    Returns:
        float: Combined loss value (MSE + KS statistic)
    """
    mu, sigma, slope = params

    # Parameter validity checks
    if sigma < 1e-6 or slope < 0.1 or slope > 10:
        return 1e6  # Large penalty for invalid parameters

    try:
        # Generate log-normal LC50 samples
        lc50_samples = np.exp(np.random.normal(mu, sigma, 100))  # 100 samples
        lc50_samples = np.clip(lc50_samples, 1e-6, 1e6)  # Prevent extreme values

        # Get experimental data for this generation
        b_data = df_b_true[df_b_true['generations'] == gen]
        conc = b_data['concentration(mg/l)'].values
        insects = b_data['insects'].values

        # Calculate predicted mortality
        log_conc = np.log(np.clip(conc, 1e-6, None))  # Log-transform concentrations
        log_lc50 = np.log(lc50_samples.mean())  # Use mean for stability

        # Probit model: P(death) = Φ(slope*(log_conc - log_lc50))
        prob = stable_norm_cdf(slope * (log_conc - log_lc50))
        synthetic_dead = np.round(insects * prob).astype(int)

        # Validity check
        if len(synthetic_dead) == 0 or np.isnan(synthetic_dead).any():
            return 1e6

        # Calculate evaluation metrics
        real_dead = b_data['dead number'].values
        mse = mean_squared_error(real_dead, synthetic_dead)  # Mean squared error
        ks_stat = ks_2samp(real_dead, synthetic_dead).statistic  # Distribution similarity
        return mse + ks_stat  # Combined loss function

    except Exception as e:
        print(f"Objective function error: {str(e)}")
        return 1e6  # Return large penalty on errors


# -------------------- Main Processing Pipeline -------------------- #
def main_process():
    """Main analysis pipeline with data processing and optimization"""

    # Data preparation
    df_a, df_b = load_and_clean_data()
    os.makedirs('output', exist_ok=True)  # Create output directory

    # Initialize results storage
    results = {
        'params': [],  # Optimized parameters
        'ks_test': [],  # Goodness-of-fit tests
        'data': []  # Synthetic dataset
    }

    # Process each generation separately
    for gen in df_a['Generations'].unique():
        print(f"\nProcessing Generation {gen}...")

        # Get initial parameters from experimental data
        lc50_row = df_a[df_a['Generations'] == gen].iloc[0]
        mu_init, sigma_init = calculate_params(lc50_row)
        slope_init = extract_slope(lc50_row['Slope±SE'])

        # Parameter optimization using L-BFGS-B algorithm
        try:
            res = minimize(
                objective_function,
                x0=[mu_init, sigma_init, slope_init],  # Initial guess
                args=(gen, df_b),  # Additional arguments
                method='L-BFGS-B',  # Bounded optimization
                bounds=[(None, None), (1e-6, 1), (0.1, 10)],  # Parameter bounds
                options={'maxiter': 50, 'ftol': 1e-4}  # Optimization controls
            )
            mu_opt, sigma_opt, slope_opt = res.x  # Extract optimized parameters
            print(f"Optimization result: {res.message}")
        except Exception as e:
            print(f"Optimization failed: {str(e)}, using initial parameters")
            mu_opt, sigma_opt, slope_opt = mu_init, sigma_init, slope_init

        # Generate final synthetic dataset
        lc50_samples = np.exp(np.random.normal(mu_opt, sigma_opt, 1000))  # 1000 samples
        b_data = df_b[df_b['generations'] == gen]

        synthetic_dead = []
        for _, row in b_data.iterrows():
            conc = max(row['concentration(mg/l)'], 1e-6)  # Ensure positive
            insects = row['insects']
            lc50 = np.random.choice(lc50_samples)  # Random LC50 sample

            # Calculate mortality probability
            prob = stable_norm_cdf(
                slope_opt * (np.log(conc) - np.log(lc50))
            )
            dead = int(np.clip(np.round(insects * prob), 0, insects))  # Ensure valid count
            synthetic_dead.append(dead)

        # Store results
        results['params'].append({
            'Generation': gen,
            'mu': mu_opt,  # Log-normal mean
            'sigma': sigma_opt,  # Log-normal std
            'slope': slope_opt  # Dose-response slope
        })

        # Goodness-of-fit test
        real_dead = b_data['dead number'].values
        results['ks_test'].append(ks_2samp(real_dead, synthetic_dead).pvalue)

        # Store synthetic data
        results['data'].extend(zip(
            [gen] * len(b_data),
            b_data['concentration(mg/l)'],
            b_data['insects'],
            synthetic_dead
        ))

    # Save outputs
    pd.DataFrame(results['params']).to_csv(
        'output/optimized_parameters.csv', index=False
    )

    pd.DataFrame(results['data'], columns=[
        'generations', 'concentration(mg/l)', 'insects', 'dead number'
    ]).to_csv('output/synthetic_dataset.csv', index=False)

    print("\nProcessing complete! Results saved to output directory")


if __name__ == "__main__":
    main_process()