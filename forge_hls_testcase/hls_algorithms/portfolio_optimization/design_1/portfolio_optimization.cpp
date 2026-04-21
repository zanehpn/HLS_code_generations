#include <cmath>

#define N 1024

void portfolio_optimization(double returns[N], double cov_matrix[N][N], double weights[N], double risk_free_rate, double target_return, double optimized_weights[N]) {
#pragma HLS ARRAY_PARTITION variable=optimized_weights type=cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=weights type=cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=cov_matrix type=cyclic dim=2 factor=8
#pragma HLS ARRAY_PARTITION variable=cov_matrix type=cyclic dim=1 factor=4
#pragma HLS ARRAY_PARTITION variable=returns type=cyclic dim=1 factor=8
    double sum_weights = 0.0;
    double portfolio_return = 0.0;
    double portfolio_variance = 0.0;

    // Initialize weights
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
        weights[i] = 1.0 / N;
    }

    // Calculate initial portfolio return and variance
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=1
        portfolio_return += weights[i] * returns[i];
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=8
            portfolio_variance += weights[i] * weights[j] * cov_matrix[i][j];
        }
    }

    // Optimization loop (simplified for demonstration purposes)
    for (int iter = 0; iter < 1000; iter++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=4
        for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
            double gradient = returns[i] - risk_free_rate;
            for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=4
                gradient -= cov_matrix[i][j] * weights[j];
            }
            weights[i] += 0.01 * gradient;
        }

        // Normalize weights to sum to 1
        sum_weights = 0.0;
        for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
            sum_weights += weights[i];
        }
        for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
            weights[i] /= sum_weights;
        }

        // Calculate portfolio return and variance
        portfolio_return = 0.0;
        portfolio_variance = 0.0;
        for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
            portfolio_return += weights[i] * returns[i];
            for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=8
                portfolio_variance += weights[i] * weights[j] * cov_matrix[i][j];
            }
        }

        // Check if target return is met
        if (portfolio_return >= target_return) {
            break;
        }
    }

    // Copy optimized weights to output
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=16
        optimized_weights[i] = weights[i];
    }
}

// Top function name: portfolio_optimization
