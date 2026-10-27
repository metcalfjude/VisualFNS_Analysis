using DrWatson
@quickactivate "VisualFNS_Analysis"

using Statistics
using Random
using DataFrames
using Plots
using CSV
using DSP

# =====================================================================
# HELPERS FUNCTIONS
# =====================================================================

"Zero-phase Butterworth low-pass on each finite, contiguous run of at least
`min_segment` samples. Shorter runs and non-finite samples are left unchanged."
function lowpass_segments(trace, fs; cutoff=200, order=3, min_segment=16)
    out = Float64.(trace)
    ok = isfinite.(out)
    design = digitalfilter(Lowpass(cutoff), Butterworth(order), fs=fs)
    i, n = 1, length(out)
    while i <= n
        if !ok[i]
            i += 1
            continue
        end
        j = i
        while j < n && ok[j + 1]
            j += 1
        end
        j - i + 1 >= min_segment && (out[i:j] = filtfilt(design, out[i:j]))
        i = j + 1
    end
    return out
end

"Closed-form OLS fit y ≈ b0 + b1*x; returns (b0, b1)."
function ols(x, y)
    b1 = cov(x, y) / var(x)
    return mean(y) - b1 * mean(x), b1
end

"Slope of log(y) vs log(x); NaN if fewer than 3 finite, positive points."
function loglog_slope(x, y)
    ok = @. isfinite(x) & isfinite(y) & (x > 0) & (y > 0)
    return count(ok) < 3 ? NaN : ols(log.(x[ok]), log.(y[ok]))[2]
end

"(M(τ)/M(τ₁))^(1/q) column-wise; anything non-finite becomes NaN."
function rescale_moments(M, qs)
    r = (M ./ M[1:1, :]) .^ (1 ./ qs')
    r[.!isfinite.(r)] .= NaN
    return r
end

"Apply `f` along the bootstrap dimension (1), ignoring non-finite replicates."
function bootstat(f, A)
    out = mapslices(A; dims=1) do v
        v = filter(isfinite, v)
        length(v) < 2 ? NaN : f(v)
    end
    return dropdims(out; dims=1)
end
q_lo(v) = quantile(v, 0.025)
q_hi(v) = quantile(v, 0.975)

qkey(q) = replace(string(round(q, digits=1)), "." => "_")


# =====================================================================
# 2. PARAMETERS
# =====================================================================

fs = 10_000.0
tau_max = round(Int, fs * 200.0 / 1000)          # 200 ms maximum lag

# Dense sampling at short lags, logarithmic sampling at long lags.
lags = sort(unique(vcat(
    1:1000,
    round.(Int, exp.(range(log(51.0), log(tau_max); length=1000)))
)))
num_lags = length(lags)
lag_ms = lags ./ fs .* 1000

qs = collect(0.1:0.1:3.0)
num_qs = length(qs)

fit_idx = 5:30                                   # scaling-fit window (lag indices)
trunc_lag = lag_ms[fit_idx]
fit_zeta(M) = [loglog_slope(trunc_lag, M[fit_idx, qi]) for qi in 1:size(M, 2)]

B, seed = 200, 12345                             # bootstrap replicates, RNG seed
TEST_LIMIT = Inf                                 # e.g. 1000 for development runs

println("Evaluating $num_lags lags and $num_qs moment orders.")
println("Fitting range: $(first(trunc_lag)) to $(last(trunc_lag)) ms.")


# =====================================================================
# LOAD AND FILTER TRAJECTORIES
# =====================================================================

csv_dir = datadir("STD-Params")
files = sort(filter(f -> endswith(lowercase(f), ".csv"),
                    readdir(csv_dir; join=true)))
isempty(files) && error("No CSV files found in $csv_dir")
println("Found $(length(files)) CSV files in $csv_dir. Loading and filtering...")

trajs = Matrix{Float64}[] # each N × 2: [x y], low-passed

for file in files
    length(trajs) >= TEST_LIMIT && break
    try
        df = CSV.read(file, DataFrame)
        if !all(in(propertynames(df)), (:t, :x, :y))
            @warn "Skipping $file: missing t, x, or y columns."
            continue
        end
        sort!(df, :t)
        cols = [lowpass_segments(Float64.(coalesce.(df[!, c], NaN)), fs)
                for c in (:x, :y)]
        push!(trajs, hcat(cols...))
    catch e
        @warn "Could not read $file" exception=(e, catch_backtrace())
    end
end

num_trials = length(trajs)
num_trials > 0 || error("No valid trajectories could be loaded from $csv_dir")
println("Loaded $num_trials trajectories.")


# =====================================================================
# DISPLACEMENT MOMENTS PER TRIAL
# =====================================================================

println("Calculating moments using $(Threads.nthreads()) Julia threads...")

trial_sums = zeros(num_trials, num_lags, num_qs)     # Σ |Δr|^q
trial_counts = zeros(Int, num_trials, num_lags)      # valid displacement pairs
progress = Threads.Atomic{Int}(0)

Threads.@threads for k in 1:num_trials
    traj = trajs[k]
    n = size(traj, 1)

    for (li, tau) in enumerate(lags)
        tau >= n && break
        nv = 0

        @inbounds for t in 1:(n - tau)
            d = (traj[t + tau, 1] - traj[t, 1])^2 +
                (traj[t + tau, 2] - traj[t, 2])^2
            isfinite(d) || continue          # NaN/Inf in any coordinate
            nv += 1
            for qi in 1:num_qs
                trial_sums[k, li, qi] += d^(qs[qi] / 2)
            end
        end

        trial_counts[k, li] = nv
    end

    done = Threads.atomic_add!(progress, 1) + 1
    (done % max(1, num_trials ÷ 10) == 0 || done == num_trials) &&
        println("Processed $done / $num_trials trials.")
end

# Pooled E[|Δr|^q] at each lag (NaN where no valid pairs exist).
moments = dropdims(sum(trial_sums; dims=1); dims=1) ./
          vec(sum(trial_counts; dims=1))


# =====================================================================
# TRIAL-LEVEL BOOTSTRAP
# =====================================================================

println("Running $B trial-level bootstrap iterations...")

boot_raw = fill(NaN, B, num_lags, num_qs)
boot_rescaled = fill(NaN, B, num_lags, num_qs)
boot_zeta = fill(NaN, B, num_qs)

Threads.@threads for b in 1:B
    rng = MersenneTwister(seed + b)              # independent stream per replicate
    cnt = zeros(Int, num_lags)
    sums = zeros(num_lags, num_qs)

    for k in rand(rng, 1:num_trials, num_trials)
        cnt .+= @view trial_counts[k, :]
        sums .+= @view trial_sums[k, :, :]
    end

    m = sums ./ cnt
    boot_raw[b, :, :] = m
    boot_rescaled[b, :, :] = rescale_moments(m, qs)
    boot_zeta[b, :] = fit_zeta(m)
end

sem_raw = bootstat(std, boot_raw)
sem_resc = bootstat(std, boot_rescaled)
ci_lo = bootstat(q_lo, boot_rescaled)
ci_hi = bootstat(q_hi, boot_rescaled)

z_sem = bootstat(std, boot_zeta)
z_lo = bootstat(q_lo, boot_zeta)
z_hi = bootstat(q_hi, boot_zeta)


# =====================================================================
# SCALING EXPONENTS AND CSV OUTPUT
# =====================================================================

resc = rescale_moments(moments, qs)
zeta = fit_zeta(moments)
lambdas = zeta ./ qs                  # λ = ζ/q, so its SE and CI scale by 1/q

processed_dir = datadir("processed")
mkpath(processed_dir)

raw_df = DataFrame(lag=lags, lag_ms=lag_ms)
spec_df = DataFrame(lag=lags, lag_ms=lag_ms)

for (qi, q) in enumerate(qs)
    k = "q_" * qkey(q)
    raw_df[!, k] = moments[:, qi]
    raw_df[!, k * "_sem"] = sem_raw[:, qi]

    spec_df[!, k * "_raw_mean"] = moments[:, qi]
    spec_df[!, k * "_rescaled_mean"] = resc[:, qi]
    spec_df[!, k * "_rescaled_sem"] = sem_resc[:, qi]
    spec_df[!, k * "_rescaled_ci_lower"] = ci_lo[:, qi]
    spec_df[!, k * "_rescaled_ci_upper"] = ci_hi[:, qi]
end

lambda_df = DataFrame(
    q=qs,
    zeta=zeta, zeta_sem=z_sem, zeta_ci_lower=z_lo, zeta_ci_upper=z_hi,
    lambda=lambdas, lambda_sem=z_sem ./ qs,
    lambda_ci_lower=z_lo ./ qs, lambda_ci_upper=z_hi ./ qs,
    fit_start_ms=first(trunc_lag), fit_end_ms=last(trunc_lag)
)

CSV.write(joinpath(processed_dir, "moment_spectrum.csv"), raw_df)
CSV.write(joinpath(processed_dir, "time_averaged_spectrum.csv"), spec_df)
CSV.write(joinpath(processed_dir, "time_averaged_lambdas.csv"), lambda_df)


# =====================================================================
# PLOT: RESCALED MOMENT SPECTRUM
# =====================================================================

cg = cgrad(:inferno)

moment_plot = plot(;
    xlabel="Time Lag Δt (ms)",
    ylabel="Rescaled Moment ⟨|Δx|^q⟩^(1/q)",
    title="Spectrum of Generalized Length Scales",
    xscale=:log10, yscale=:log10, legend=:outertopright, dpi=300
)

for (qi, q) in enumerate(qs)
    lab = qi % 5 == 1 || qi == num_qs
    yv = ifelse.(resc[:, qi] .> 0, resc[:, qi], NaN)   # log axis needs y > 0

    plot!(moment_plot, lag_ms, yv;
        label = lab ? "q = $(round(q, digits=1))" : "",
        ribbon = lab ? (yv .- ci_lo[:, qi], ci_hi[:, qi] .- yv) : nothing,
        fillalpha=0.2,
        linewidth = lab ? 2.0 : 1.0,
        alpha = lab ? 1.0 : 0.3,
        color=cg[(qi - 1) / (num_qs - 1)]
    )
end

vspan!(moment_plot, [first(trunc_lag), last(trunc_lag)];
       color=:gray, alpha=0.2, label="Fitting Regime")

# q = 2 power-law fit on the rescaled moment.
i2 = findfirst(q -> isapprox(q, 2.0), qs)
if !isnothing(i2)
    r2v = resc[fit_idx, i2]
    good = isfinite.(r2v) .& (r2v .> 0)
    if count(good) >= 3
        b0, b1 = ols(log10.(trunc_lag[good]), log10.(r2v[good]))
        fx = [first(trunc_lag), last(trunc_lag)]
        plot!(moment_plot, fx, 10 .^ (b0 .+ b1 .* log10.(fx));
              color=:black, linestyle=:dash, linewidth=2.5,
              label="q = 2 Fit (slope = $(round(b1, digits=3)))")
    end
end


# =====================================================================
# PLOT: LOCAL LOGARITHMIC DERIVATIVE (q = 1)
# =====================================================================

q1 = findfirst(q -> isapprox(q, 1.0), qs)

# 3-point sliding slope, assigned to the window centre.
slopes_q1 = [NaN;
             [loglog_slope(lag_ms[i:i+2], moments[i:i+2, q1]) for i in 1:num_lags-2];
             NaN]

deriv_plot = plot(lag_ms, slopes_q1;
    xlabel="Time Lag Δt (ms)", ylabel="Local Exponent γ",
    title="Local Logarithmic Derivative (q=1.0)",
    xscale=:log10, linewidth=2, color=:black,
    xlims=(1, 100), ylims=(0, 1.5), legend=false, dpi=300
)
vspan!(deriv_plot, [first(trunc_lag), last(trunc_lag)]; color=:gray, alpha=0.2)
isfinite(lambdas[q1]) &&
    hline!(deriv_plot, [lambdas[q1]]; color=:red, linestyle=:dash)


# =====================================================================
# PLOT λ(q) against q
# =====================================================================

ok = isfinite.(zeta)
qx, qy = qs[ok], zeta[ok]

lambda_plot = plot(;
    title="Scaling Exponent vs. Moment Order",
    xlabel="Fractional Moment Order (q)", ylabel="Exponent qλ(q)",
    legend=:bottomright, dpi=300, framestyle=:box,
    grid=:xy, gridalpha=0.3, gridlinestyle=:dash,
    titlefontsize=12, guidefontsize=11, tickfontsize=9, background_color=:white
)

scatter!(lambda_plot, qx, qy;
    yerror=z_sem[ok],
    marker=:circle, markersize=6, markercolor=:darkorange,
    markerstrokecolor=:white, markerstrokewidth=1.0, markeralpha=0.9,
    color=RGB(0.4, 0.4, 0.4), alpha=0.6,
    label="Empirical qλ(q)"
)

if length(qx) >= 3
    b0, b1 = ols(qx, qy)
    yhat = b0 .+ b1 .* qx
    ss_res = sum(abs2, qy .- yhat)
    ss_tot = sum(abs2, qy .- mean(qy))

    if ss_tot > eps(Float64)
        r2 = 1 - ss_res / ss_tot
        slope_se = sqrt(ss_res / (length(qx) - 2) / sum(abs2, qx .- mean(qx)))

        plot!(lambda_plot, qx, yhat;
            linewidth=2.5, color=RGB(0.89, 0.10, 0.11), linestyle=:dash,
            label="Linear fit (slope = $(round(b1, digits=3)) ± " *
                  "$(round(slope_se, digits=3)), R² = $(round(r2, digits=3)))")
    else
        @warn "Skipping regression uncertainty: degenerate data."
    end
end


# =====================================================================
# 10. SAVE FIGURES
# =====================================================================

mkpath(plotsdir())

for (name, p) in pairs((
        rescaled_moment_spectrum = moment_plot,
        local_derivative_plateau = deriv_plot,
        lambda_vs_q_fit = lambda_plot))
    savefig(p, plotsdir("$name.png"))
end

println("All plots and CSV outputs successfully generated.")
println("Processed data: $processed_dir")
println("Figures: $(plotsdir())")
