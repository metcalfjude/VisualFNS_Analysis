using DrWatson
@quickactivate("FNS_Model_Analysis")

using Base.Threads
using Statistics
using Random
using DataFrames
using LinearAlgebra
using Plots
using CSV
using DSP

# =====================================================================
# FUNCTION DEFINITIONS
# =====================================================================
"""
Zero-phase low-pass filtering of finite, contiguous segments.
Segments that are too short are left unchanged.
"""
function experimental_filter(
    trace::AbstractVector,
    fs=10000.0;
    cutoff=200,
    order=3,
    min_segment=16
)
    smoothed = Float64.(trace)
    valid_mask = isfinite.(trace)

    count(valid_mask) < min_segment && return smoothed

    my_filter = digitalfilter(
        Lowpass(cutoff),
        Butterworth(order);
        fs=fs
    )

    n = length(trace)
    i = 1

    while i <= n
        if !valid_mask[i]
            i += 1
            continue
        end

        first_idx = i

        while i <= n && valid_mask[i]
            i += 1
        end

        last_idx = i - 1

        if last_idx - first_idx + 1 >= min_segment
            smoothed[first_idx:last_idx] .=
                filtfilt(my_filter, Float64.(trace[first_idx:last_idx]))
        end
    end

    return smoothed
end

"""
Local slope of log(y) versus log(x), using a sliding window.
Invalid or nonpositive values produce NaN slopes.
"""
function local_log_derivative(x, y, window_size=3)
    length(x) == length(y) ||
        throw(DimensionMismatch("x and y must have equal lengths"))

    window_size >= 2 ||
        throw(ArgumentError("window_size must be at least 2"))

    n = length(x)
    slopes = fill(NaN, n)

    n < window_size && return slopes

    for i in 1:(n - window_size + 1)
        idx = i:(i + window_size - 1)

        if !all(isfinite, x[idx]) ||
           !all(isfinite, y[idx]) ||
           !all(>(0), x[idx]) ||
           !all(>(0), y[idx])
            continue
        end

        X = [ones(window_size) log.(x[idx])]
        slopes[i + div(window_size, 2)] =
            (X \ log.(y[idx]))[2]
    end

    return slopes
end

# =====================================================================
# MAIN EXECUTION: LOAD CSV TRAJECTORIES FROM /data/STD_Params/
# =====================================================================
csv_dir = datadir("STD-Params")
isdir(csv_dir) || error("Directory not found: $csv_dir")

csv_files = filter(f -> endswith(f, ".csv"), readdir(csv_dir, join=true))
isempty(csv_files) && error("No CSV files found in $csv_dir")

println("Found $(length(csv_files)) trajectory CSV files in $csv_dir.")

# Load all valid trajectories into a vector of matrices [x_trace y_trace]
raw_trajectories = Vector{Matrix{Float64}}()
for file in csv_files
    try
        data = CSV.read(file, DataFrame)
        if all(hasproperty.(Ref(data), [:t, :x, :y]))
            # Sort by time just in case
            sort!(data, :t)
            push!(raw_trajectories, hcat(Float64.(data.x), Float64.(data.y)))
        else
            @warn "Skipping $file: missing :t, :x, or :y columns."
        end
    catch e
        @warn "Could not read $file: $e"
    end
end

num_trials = length(raw_trajectories)
println("Successfully loaded $num_trials valid trajectory files.")

TEST_LIMIT = nothing # Set to an integer like 1000 if you want a test cap
if !isnothing(TEST_LIMIT) && TEST_LIMIT < num_trials
    raw_trajectories = first(raw_trajectories, TEST_LIMIT)
    num_trials = length(raw_trajectories)
    println("⚠️ TESTING MODE: Only using the first $num_trials trials.")
end

# =====================================================================
# 1. DEFINE PARAMETERS
# =====================================================================

# 1. Define sampling parameters
fs = 10000.0                       # 10 kHz
max_lag_ms = 200.0                 # 200 ms max lag
tau_max = Int(fs * max_lag_ms / 1000) # 2000 samples

# 2. Build lag index vector: dense at fine scales, decimated at large lags
# e.g., evaluate every sample up to lag 1000 (100 ms), then geometrically space up to lag 2000
fine_lags = 1:1000
coarse_lags = Int.(round.(exp.(range(log(51), log(tau_max), length=1000))))
lags = unique(vcat(fine_lags, coarse_lags)) 

num_lags = length(lags)
lag_axis_ms = lags ./ fs .* 1000

qs = collect(0.1:0.1:3.0) 
num_qs = length(qs)

fit_start, fit_end = 5, 30
fit_indices = fit_start:fit_end
trunc_lag = lag_axis_ms[fit_indices]

# =====================================================================
# 2. PRE-COMPUTE AND FILTER 2D TRAJECTORIES
# =====================================================================
println("Pre-computing and filtering trajectory data (200Hz cutoff)...")
filtered_trajectories = map(raw_trajectories) do traj
    smooth_x = experimental_filter(traj[:, 1], fs, cutoff=200)
    smooth_y = experimental_filter(traj[:, 2], fs, cutoff=200)
    hcat(smooth_x, smooth_y)
end

# =====================================================================
# 3. CALCULATE MOMENTS PER TRIAL
# =====================================================================
println("Calculating moments per trial using $(Threads.nthreads()) threads...")

trial_sums = zeros(Float64, num_trials, num_lags, num_qs)
trial_counts = zeros(Int, num_trials, num_lags)
progress_counter = Threads.Atomic{Int}(0)

Threads.@threads for trial_idx in 1:num_trials
    traj = filtered_trajectories[trial_idx]
    trial_len = size(traj, 1)
    
    for (l_idx, tau) in enumerate(lags)
        tau >= trial_len && break
        
        count = 0
        @inbounds for t in 1:(trial_len - tau)
            x1, y1 = traj[t, 1], traj[t, 2]
            x2, y2 = traj[t + tau, 1], traj[t + tau, 2]
            
            if isfinite(x1) && isfinite(y1) && isfinite(x2) && isfinite(y2)
                d = (x2 - x1)^2 + (y2 - y1)^2
                for i in 1:num_qs
                    trial_sums[trial_idx, l_idx, i] += d^(qs[i] / 2)
                end
                count += 1
            end
        end
        trial_counts[trial_idx, l_idx] = count
    end
    
    Threads.atomic_add!(progress_counter, 1)
    current_progress = progress_counter[]
    (current_progress % max(1, div(num_trials, 10)) == 0) && println("Processed $current_progress / $num_trials trials")
end

moments_matrix = fill(NaN, max_lag, num_qs)
global_sums = dropdims(sum(trial_sums, dims=1), dims=1)
global_counts = dropdims(sum(trial_counts, dims=1), dims=1)

for tau in 1:max_lag
    if global_counts[tau] > 0
        moments_matrix[tau, :] .= global_sums[tau, :] ./ global_counts[tau]
    end
end

# =====================================================================
# 3.5 TRIAL-LEVEL BOOTSTRAPPING (95% CI & SEM for Moments & Exponents)
# =====================================================================
B = 200
Random.seed!(12345) 

println("Running $B trial-level bootstrap iterations...")

boot_raw = fill(NaN, B, max_lag, num_qs)
boot_rescaled = fill(NaN, B, max_lag, num_qs)
boot_zeta = fill(NaN, B, num_qs)

Threads.@threads for b in 1:B
    sampled_indices = rand(1:num_trials, num_trials)

    counts_b = zeros(Int, max_lag)
    sums_b = zeros(Float64, max_lag, num_qs)

    for idx in sampled_indices
        counts_b .+= trial_counts[idx, :]
        sums_b .+= reshape(trial_sums[idx, :, :], max_lag, num_qs)
    end

    counts_b[1] == 0 && continue
    m1 = sums_b[1, :] ./ counts_b[1]

    moments_b = fill(NaN, max_lag, num_qs)
    for tau in 1:max_lag
        if counts_b[tau] > 0
            moments_b[tau, :] .= sums_b[tau, :] ./ counts_b[tau]
        end
    end

    for tau in 1:max_lag
        counts_b[tau] == 0 && continue
        mtau = moments_b[tau, :]

        for i in 1:num_qs
            boot_raw[b, tau, i] = mtau[i]
            if isfinite(m1[i]) && m1[i] > 0 && isfinite(mtau[i]) && mtau[i] >= 0
                boot_rescaled[b, tau, i] = (mtau[i] / m1[i])^(1 / qs[i])
            end
        end
    end

    for i in 1:num_qs
        y = moments_b[fit_indices, i]
        valid = isfinite.(trunc_lag) .& isfinite.(y) .& (trunc_lag .> 0) .& (y .> 0)
        if count(valid) >= 3
            X = [ones(count(valid)) log.(trunc_lag[valid])]
            boot_zeta[b, i] = (X \ log.(y[valid]))[2]
        end
    end
end

ci_lower_rescaled = fill(NaN, max_lag, num_qs)
ci_upper_rescaled = fill(NaN, max_lag, num_qs)
sems_rescaled = fill(NaN, max_lag, num_qs)
sems_raw = fill(NaN, max_lag, num_qs)

for tau in 1:max_lag, i in 1:num_qs
    raw_vals = filter(isfinite, boot_raw[:, tau, i])
    if length(raw_vals) >= 2
        sems_raw[tau, i] = std(raw_vals)
    end

    rescaled_vals = filter(isfinite, boot_rescaled[:, tau, i])
    if !isempty(rescaled_vals)
        ci_lower_rescaled[tau, i] = quantile(rescaled_vals, 0.025)
        ci_upper_rescaled[tau, i] = quantile(rescaled_vals, 0.975)
        if length(rescaled_vals) >= 2
            sems_rescaled[tau, i] = std(rescaled_vals)
        end
    end
end

zeta_sems = fill(NaN, num_qs)
lambda_sems = fill(NaN, num_qs)
zeta_ci_lower = fill(NaN, num_qs)
zeta_ci_upper = fill(NaN, num_qs)
lambda_ci_lower = fill(NaN, num_qs)
lambda_ci_upper = fill(NaN, num_qs)

for i in 1:num_qs
    z_vals = filter(isfinite, boot_zeta[:, i])
    if length(z_vals) >= 2
        zeta_sems[i] = std(z_vals)
        zeta_ci_lower[i] = quantile(z_vals, 0.025)
        zeta_ci_upper[i] = quantile(z_vals, 0.975)
    end
    
    l_vals = z_vals ./ qs[i]
    if length(l_vals) >= 2
        lambda_sems[i] = std(l_vals)
        lambda_ci_lower[i] = quantile(l_vals, 0.025)
        lambda_ci_upper[i] = quantile(l_vals, 0.975)
    end
end

# =====================================================================
# 4. SAVE RAW MOMENTS TO CSV
# =====================================================================
println("Formatting and saving raw spectrum to CSV...")
results_df = DataFrame(lag = 1:max_lag)

for i in 1:num_qs
    q_str = replace(string(round(qs[i], digits=1)), "." => "_")
    results_df[!, Symbol("q_", q_str)] = moments_matrix[:, i]
    results_df[!, Symbol("q_", q_str, "_sem")] = sems_raw[:, i]
end

mkdir(datadir("processed"))
CSV.write(datadir("processed", "moment_spectrum.csv"), results_df)

# ============================================================
# 5. SCALING REGIME & RESCALED SPECTRUM
# ============================================================
window_size = 3

q1_idx = findfirst(q -> isapprox(q, 1.0), qs)
isnothing(q1_idx) && error("Moment order q=1.0 is missing.")

local_slopes_q1 = local_log_derivative(
    lag_axis_ms,
    moments_matrix[:, q1_idx],
    window_size
)

zeta = fill(NaN, num_qs)
for i in 1:num_qs
    y = moments_matrix[fit_indices, i]
    valid = isfinite.(trunc_lag) .& isfinite.(y) .& (trunc_lag .> 0) .& (y .> 0)
    if count(valid) >= 3
        X = [ones(count(valid)) log.(trunc_lag[valid])]
        zeta[i] = (X \ log.(y[valid]))[2]
    end
end

lambdas = zeta ./ qs

spectrum_df = DataFrame(lag = 1:max_lag, lag_ms = lag_axis_ms)
for i in 1:num_qs
    q = qs[i]
    q_str = replace(string(round(q, digits=1)), "." => "_")
    mu_0 = moments_matrix[1, i]

    spectrum_df[!, Symbol("q_", q_str, "_raw_mean")] = moments_matrix[:, i]

    normalized = fill(NaN, max_lag)
    if isfinite(mu_0) && mu_0 > 0
        valid = isfinite.(moments_matrix[:, i]) .& (moments_matrix[:, i] .>= 0)
        normalized[valid] .= (moments_matrix[valid, i] ./ mu_0).^(1 / q)
    end

    spectrum_df[!, Symbol("q_", q_str, "_rescaled_mean")] = normalized
    spectrum_df[!, Symbol("q_", q_str, "_rescaled_sem")] = sems_rescaled[:, i]
    spectrum_df[!, Symbol("q_", q_str, "_rescaled_ci_lower")] = ci_lower_rescaled[:, i]
    spectrum_df[!, Symbol("q_", q_str, "_rescaled_ci_upper")] = ci_upper_rescaled[:, i]
end

CSV.write(datadir("processed", "time_averaged_spectrum.csv"), spectrum_df)

lambda_df = DataFrame(
    q = qs,
    zeta = zeta,
    zeta_sem = zeta_sems,
    zeta_ci_lower = zeta_ci_lower,
    zeta_ci_upper = zeta_ci_upper,
    lambda = lambdas,
    lambda_sem = lambda_sems,
    lambda_ci_lower = lambda_ci_lower,
    lambda_ci_upper = lambda_ci_upper,
    fit_start_ms = fill(lag_axis_ms[fit_start], num_qs),
    fit_end_ms = fill(lag_axis_ms[fit_end], num_qs)
)

CSV.write(datadir("processed", "time_averaged_lambdas.csv"), lambda_df)

# =====================================================================
# 6. VISUAL PRESENTATION
# =====================================================================
my_colors = [cgrad(:inferno)[x] for x in range(0, 1, length=num_qs)]

moment_plot = plot(
    xlabel="Time Lag Δt (ms)", 
    ylabel="Rescaled Moment ⟨|Δx|^q⟩^{1/q}",
    title="Spectrum of Generalized Length Scales",
    xscale=:log10,
    yscale=:log10,
    grid=true,
    legend=:outertopright,
    dpi=300
)

for i in 1:num_qs
    q = qs[i]
    is_labeled = (i % 5 == 1) || (i == num_qs)
    q_str = replace(string(round(q, digits=1)), "." => "_")
    
    normalized_moment = spectrum_df[!, Symbol("q_", q_str, "_rescaled_mean")]
    lower_bound = spectrum_df[!, Symbol("q_", q_str, "_rescaled_ci_lower")]
    upper_bound = spectrum_df[!, Symbol("q_", q_str, "_rescaled_ci_upper")]
    
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

vspan!(moment_plot, [lag_axis_ms[fit_start], lag_axis_ms[fit_end]], 
       color=:gray, alpha=0.2, label="Fitting Regime")

# Add q = 2 fit line
q2_idx = findfirst(q -> isapprox(q, 2.0), qs)
if !isnothing(q2_idx)
    q2_str = replace(string(round(qs[q2_idx], digits=1)), "." => "_")
    q2_rescaled = spectrum_df[!, Symbol("q_", q2_str, "_rescaled_mean")]
    fit_mask = (lag_axis_ms .>= lag_axis_ms[fit_start]) .& (lag_axis_ms .<= lag_axis_ms[fit_end])
    valid_fit = fit_mask .& isfinite.(q2_rescaled) .& (q2_rescaled .> 0)
    
    if count(valid_fit) >= 2
        log_x = log10.(lag_axis_ms[valid_fit])
        log_y = log10.(q2_rescaled[valid_fit])
        X_fit = [ones(length(log_x)) log_x]
        beta = X_fit \ log_y
        
        fit_x = [lag_axis_ms[fit_start], lag_axis_ms[fit_end]]
        X_line = [ones(length(fit_x)) log10.(fit_x)]
        fit_y = 10 .^ (X_line * beta)
        
        plot!(moment_plot, fit_x, fit_y, 
              color=:black, linestyle=:dash, linewidth=2.5, 
              label="q = 2 Fit (slope = $(round(beta[2], digits=3)))")
    end
end

deriv_plot = plot(
    lag_axis_ms, local_slopes_q1,
    xlabel="Time Lag Δt (ms)",
    ylabel="Local Exponent γ",
    title="Local Logarithmic Derivative (q=1.0)",
    xscale=:log10,
    linewidth=2,
    color=:black,
    xlims=(1, 100),
    ylims=(0, 1.5),
    legend=false,
    dpi=300
)
vspan!(deriv_plot, [lag_axis_ms[fit_start], lag_axis_ms[fit_end]], color=:gray, alpha=0.2)
hline!(deriv_plot, [lambdas[q1_idx]], color=:red, linestyle=:dash)

# =====================================================================
# 7. FIGURE 3: SCALING EXPONENT q*λ(q) VS MOMENT ORDER (q) WITH ERROR BARS
# =====================================================================
println("Generating publication-styled scaling exponent q*λ vs q scatter plot...")

valid_lambda = isfinite.(qs) .& isfinite.(lambdas)
q_lambdas = qs[valid_lambda] .* lambdas[valid_lambda]
q_lambdas_se = qs[valid_lambda] .* lambda_sems[valid_lambda]

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
    yerror_color = RGB(0.4, 0.4, 0.4),      # Neutral gray error bars
    yerror_alpha = 0.6,
    label = "Empirical qλ(q)"
)

if count(valid_lambda) >= 2
    y_data = q_lambdas
    x_data = qs[valid_lambda]
    Q_matrix = [ones(count(valid_lambda)) x_data]
    c = Q_matrix \ y_data
    
    y_pred = Q_matrix * c
    residuals = y_data .- y_pred
    ss_res = sum(residuals .^ 2)
    ss_tot = sum((y_data .- mean(y_data)) .^ 2)
    r2 = 1.0 - (ss_res / ss_tot)
    
    n_pts = length(y_data)
    df_deg = n_pts - 2
    sigma2 = ss_res / df_deg
    inv_XtX = inv(Q_matrix' * Q_matrix)
    param_se = sqrt.(diag(sigma2 * inv_XtX))
    slope_se = param_se[2]
    
    m = round(c[2], digits=3)
    m_se = round(slope_se, digits=3)
    r2_str = round(r2, digits=3)

    fit_label = "Linear fit (slope = $m ± $m_se, R^2 = $r2_str)"

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
savefig(deriv_plot, plotsdir("local_derivative_plateau.png"))
savefig(lambda_plot, plotsdir("lambda_vs_q_fit.png"))
println("All plots successfully generated and saved to the plots directory!")
