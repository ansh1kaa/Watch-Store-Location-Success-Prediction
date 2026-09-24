"""
Watch Store Location Success Prediction - Dataset Generator
Generates a realistic 5,000 record dataset for academic practice.

Disclosure:
Synthetic dataset created for academic machine-learning practice.
It does not represent actual watch-store businesses or real-world locations.
"""

import os
import numpy as np
import pandas as pd

def generate_watch_store_dataset(num_samples=5000, random_seed=42):
    np.random.seed(random_seed)

    # 1. Generate core location & demographic features
    population = np.random.randint(10000, 500000, size=num_samples)
    average_monthly_income = np.round(np.random.uniform(1500, 12000, size=num_samples), 2)
    daily_foot_traffic = np.random.randint(500, 25000, size=num_samples)
    nearby_competitors = np.random.randint(0, 16, size=num_samples)
    monthly_rent = np.round(np.random.uniform(1000, 20000, size=num_samples), 2)
    distance_to_mall_km = np.round(np.random.uniform(0.1, 15.0, size=num_samples), 2)
    nearby_retail_stores = np.random.randint(5, 151, size=num_samples)
    
    # Foot traffic & income influence customer count and purchase value realistically
    customer_conversion_rate = np.random.uniform(0.02, 0.12, size=num_samples)
    estimated_monthly_customers = np.clip(
        np.round((daily_foot_traffic * 30 * customer_conversion_rate) / 10).astype(int),
        100, 3500
    )
    
    base_purchase = average_monthly_income * 0.04
    average_purchase_value = np.round(
        np.clip(base_purchase + np.random.normal(0, 50, size=num_samples), 40, 850),
        2
    )
    
    monthly_operating_cost = np.round(
        np.clip(monthly_rent * 0.8 + np.random.uniform(2000, 15000, size=num_samples), 2000, 25000),
        2
    )
    
    local_demand_score = np.round(np.random.uniform(1.0, 10.0, size=num_samples), 1)
    target_age_group_score = np.round(np.random.uniform(1.0, 10.0, size=num_samples), 1)

    # Modest, realistic business multipliers for secondary features
    pop_factor = 0.85 + 0.15 * (population / 250000.0)
    mall_factor = 1.05 - 0.02 * np.clip(distance_to_mall_km / 5.0, 0.0, 3.0)
    retail_factor = 0.90 + 0.10 * (nearby_retail_stores / 75.0)

    # 2. Target Generation (Latent economics with Gaussian noise to avoid 100% determinism)
    # NOTE: Latent variables are temporary and NEVER stored in the final dataset/features to avoid data leakage.
    latent_revenue = (
        estimated_monthly_customers * average_purchase_value * 0.10 *
        (0.5 + 0.08 * local_demand_score) *
        (0.5 + 0.08 * target_age_group_score) *
        (1.0 - 0.02 * nearby_competitors) *
        pop_factor * mall_factor * retail_factor
    )
    
    latent_total_cost = monthly_rent + monthly_operating_cost
    latent_net_profit = latent_revenue - latent_total_cost

    # Gaussian noise is added to create a less deterministic decision boundary.
    noise = np.random.normal(0, 3500, size=num_samples)
    adjusted_profit = latent_net_profit + noise

    # Binary target: 1 = Profitable, 0 = Not Profitable
    profitable = (adjusted_profit > 0).astype(int)

    # 3. Assemble Clean Pandas DataFrame
    df = pd.DataFrame({
        'population': population,
        'average_monthly_income': average_monthly_income,
        'daily_foot_traffic': daily_foot_traffic,
        'nearby_competitors': nearby_competitors,
        'monthly_rent': monthly_rent,
        'distance_to_mall_km': distance_to_mall_km,
        'nearby_retail_stores': nearby_retail_stores,
        'estimated_monthly_customers': estimated_monthly_customers,
        'average_purchase_value': average_purchase_value,
        'monthly_operating_cost': monthly_operating_cost,
        'local_demand_score': local_demand_score,
        'target_age_group_score': target_age_group_score,
        'profitable': profitable
    })

    return df

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "watch_store_locations.csv")
    
    df = generate_watch_store_dataset(num_samples=5000, random_seed=42)
    # Save clean CSV without inline header comments to prevent parsing issues
    df.to_csv(output_path, index=False)
    
    print(f"Successfully generated {len(df)} records.")
    print(f"Saved dataset to: {output_path}")
    print("\nTarget Class Distribution:")
    print(df['profitable'].value_counts(normalize=True))
