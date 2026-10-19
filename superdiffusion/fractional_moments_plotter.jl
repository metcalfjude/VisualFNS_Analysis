using DrWatson
@quickactivate :FNS_Model_Analysis

using DataFrames
using CSV
using Plots
using Statistics
using LinearAlgebra

# =====================================================================
# 1. LOAD SAVED DATA
# =====================================================================
println("Loading processed data from CSV files...")

spectrum_path = datadir("processed", "time_averaged_spectrum.csv")
lambda_path   = datadir("processed", "time_averaged_lambdas.csv")
raw_moment_path = datadir("processed", "moment_spectrum.csv")

spectrum_df = CSV.read(spectrum_path, DataFrame)
lambda_df   = CSV.read(lambda_path, DataFrame)
raw_df      = CSV.read(raw_moment_path, DataFrame)

lag_axis_ms = spectrum_df.lag_ms
qs = lambda_df.q
num_qs = length(qs)

# Correctly extract fit boundaries from lambda_df
fit_start_ms = first(lag_axis_ms)
fit_end_ms   = first(lambda_df.fit_end_ms)

# =====================================================================
# 2. COLORMAP SELECTION
# =====================================================================
# Map 30 distinct steps uniformly across the :inferno colormap
my_colors = [cgrad(:inferno)[x] for x in range(0, 1, length=num_qs)]

# =====================================================================
# 3. FIGURE 1: RESCALED MOMENT SPECTRUM (with q=2 Best Fit Line)
# =====================================================================
println("Generating rescaled moment spectrum plot...")

moment_plot = plot(
    xlabel="Time Lag Δt (ms)", 
    ylabel="Rescaled Moment",
    title="Fractional Absolute Displacements",
    xscale=:log10,
    yscale=:log10,
    grid=true,
    legend=:outertopright,
    dpi=300
)

for i in 1:num_qs
    q = qs[i]
    is_labeled = (i % 5 == 1) || (i == num_qs)  # Label every 5th curve and the last one
    q_str = replace(string(round(q, digits=1)), "." => "_")
    
    col_mean = Symbol("q_", q_str, "_rescaled_mean")
    col_lower = Symbol("q_", q_str, "_rescaled_ci_lower")
    col_upper = Symbol("q_", q_str, "_rescaled_ci_upper")
    
    normalized_moment = spectrum_df[!, col_mean]
    lower_bound = spectrum_df[!, col_lower]
    upper_bound = spectrum_df[!, col_upper]
    
    line_color = my_colors[i]

    if is_labeled
        ribbon_tuple = (normalized_moment .- lower_bound, upper_bound .- normalized_moment)
        plot!(moment_plot, lag_axis_ms, normalized_moment, 
              label="q = $(round(q, digits=1))", 
              ribbon=ribbon_tuple, fillalpha=0.20,
              linewidth=2.0, alpha=1.0, color=line_color)
    else
        plot!(moment_plot, lag_axis_ms, normalized_moment, 
              label="", 
              linewidth=1.0, alpha=0.3, color=line_color)
    end
end

vspan!(moment_plot, [fit_start_ms, fit_end_ms], 
       color=:gray, alpha=0.2, label="Fitting Regime")

# Add line of best fit for q = 2 over the fitting regime
q2_idx = findfirst(q -> isapprox(q, 2.0), qs)
if !isnothing(q2_idx)
    q2_str = replace(string(round(qs[q2_idx], digits=1)), "." => "_")
    q2_rescaled = spectrum_df[!, Symbol("q_", q2_str, "_rescaled_mean")]
    
    fit_mask = (lag_axis_ms .>= fit_start_ms) .& (lag_axis_ms .<= fit_end_ms)
    valid_fit = fit_mask .& isfinite.(q2_rescaled) .& (q2_rescaled .> 0)
    
    if count(valid_fit) >= 2
        log_x = log10.(lag_axis_ms[valid_fit])
        log_y = log10.(q2_rescaled[valid_fit])
        X_fit = [ones(length(log_x)) log_x]
        beta = X_fit \ log_y
        
        # Correctly evaluate line in log-space over the fitting window
        fit_x = [fit_start_ms, 100]
        X_line = [ones(length(fit_x)) log10.(fit_x)]
        fit_y = 10 .^ (X_line * beta)
        
        plot!(moment_plot, fit_x, fit_y*1.5,  
              color=:black, linestyle=:dash, linewidth=2.5, 
              label="q = 2 Fit (slope = $(round(beta[2], digits=3)))")
    end
end

# =====================================================================
# 7. FIGURE 3: SCALING EXPONENT q*λ(q) VS MOMENT ORDER (q)
# =====================================================================
println("Generating publication-styled scaling exponent q*λ vs q scatter plot...")

# Check if lambda_sem exists in lambda_df to handle error bars cleanly
has_se = "lambda_sem" in names(lambda_df)
lambdas = lambda_df.lambda
valid_lambda = isfinite.(qs) .& isfinite.(lambdas)

# Transform lambda to q * lambda (which equals zeta)
q_lambdas = qs[valid_lambda] .* lambdas[valid_lambda]
q_lambdas_se = lambda_df.zeta_sem[valid_lambda]

lambda_plot = plot(
    title="Scaling Exponent vs. Moment Order", 
    xlabel="Fractional Moment Order (q)", 
    ylabel="Exponent qλ(q)",
    legend=:bottomright,
    dpi=300,
    framestyle=:box,          
    grid=:xy,                 
    gridalpha=0.3,            
    gridlinestyle=:dash,      
    titlefontsize=12,
    guidefontsize=11,
    tickfontsize=9,
    background_color=:white
)

# Styled scatter plot with error bars for q*lambda
scatter!(
    lambda_plot,
    qs[valid_lambda], 
    q_lambdas,
    yerror = q_lambdas_se,
    marker = :circle,
    markersize = 6,
    markercolor = RGB(0.12, 0.47, 0.71),      # Tableau Blue
    markerstrokecolor = :white,             
    markerstrokewidth = 1.0,
    markeralpha = 0.9,
    color = RGB(0.4, 0.4, 0.4),      # Neutral gray error bars
    alpha = 0.6,
    label = "Empirical qλ(q)"
)

# High-contrast linear fit line with R^2 and analytical slope standard error
if count(valid_lambda) >= 2
    y_data = q_lambdas
    x_data = qs[valid_lambda]
    Q_matrix = [ones(count(valid_lambda)) x_data]
    c = Q_matrix \ y_data
    
    # Calculate R^2
    y_pred = Q_matrix * c
    residuals = y_data .- y_pred
    ss_res = sum(residuals .^ 2)
    ss_tot = sum((y_data .- mean(y_data)) .^ 2)
    r2 = 1.0 - (ss_res / ss_tot)
    
    # Calculate OLS standard error for parameters [intercept, slope]
    n_pts = length(y_data)
    df_deg = n_pts - 2
    sigma2 = ss_res / df_deg
    inv_XtX = inv(Q_matrix' * Q_matrix)
    param_se = sqrt.(diag(sigma2 * inv_XtX))
    slope_se = param_se[2]
    
    m = round(c[2], digits=3)
    m_se = round(slope_se, digits=3)
    r2_str = round(r2, digits=3)

    fit_label = "Linear fit (slope = $m ± $m_se, R² = $r2_str)"

    plot!(
        lambda_plot,
        x_data,
        y_pred;
        linewidth = 2.5,
        color = RGB(0.89, 0.10, 0.11),        # Tableau Red
        label = fit_label,
        linestyle = :dash
    )
end

savefig(moment_plot, plotsdir("rescaled_moment_spectrum.png"))
savefig(lambda_plot, plotsdir("lambda_vs_q_fit.png"))
println("Saved styled q*lambda plot with slope SE and R²!")

