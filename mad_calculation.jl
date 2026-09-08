using Pkg
Pkg.activate(".")

using Base.Threads
using CSV
using Statistics
using DataFrames
using LinearAlgebra
using Plots
using Random

default(
    fontfamily = "Computer Modern",
    titlefontsize = 24,
    guidefontsize = 18,          
    tickfontsize = 20,
    legendfontsize = 16,
    grid = false,
    framestyle = :box,
    dpi = 300,
    margin = 8Plots.mm,
    legend = :topright,
)

const dt = 0.0001

function mad_calculate(input_dir::String, output_dir::String; max_lag::Int=10_000, num_lags::Int=1000, number_samples::Int=10_000, seed=27)
    isdir(output_dir) || mkdir(output_dir)

    csv_files = sort(filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true)))
    tot = length(csv_files)
    
    # Generate logarithmically spaced lags
    log_lags = unique(round.(Int, 10 .^ range(0, log10(max_lag), length=num_lags)))
    
    # Output array matches the size of log_lags
    time_avg_mean_ad_curve = zeros(Float64, length(log_lags))

    progress_counter = Threads.Atomic{Int}(0)

    println("Calculating Time-Averaged Mean AD Curve using $(Threads.nthreads()) threads...")

    Threads.@threads for i in eachindex(log_lags)
        tau = log_lags[i]
        
        # Create an independent, deterministic RNG for this specific lag
        rng = Xoshiro(seed + i)
        
        sum_dist = 0.0
        count_dist = 0
        
        for file in csv_files
            data = CSV.read(file, DataFrame, ntasks=1)
            x = coalesce.(data.value1, NaN)
            y = coalesce.(data.value2, NaN)
            
            trial_len = size(data, 1)
            max_t = trial_len - tau
            
            if max_t <= 0
                continue
            end
            
            for _ in 1:number_samples
                # Pass the local RNG as the first argument
                t = rand(rng, 1:max_t) 
                
                x1, y1 = x[t], y[t]
                x2, y2 = x[t + tau], y[t + tau]

                if !isnan(x1) && !isnan(y1) && !isnan(x2) && !isnan(y2)
                    d = sqrt((x2 - x1)^2 + (y2 - y1)^2)
                    sum_dist += d
                    count_dist += 1
                end
            end
        end
        
        # Index using `i`, NOT `tau`
        if count_dist == 0
            time_avg_mean_ad_curve[i] = NaN
        else
            time_avg_mean_ad_curve[i] = sum_dist / count_dist
        end
        
        Threads.atomic_add!(progress_counter, 1)
        current_progress = progress_counter[]
        if current_progress % 10 == 0
            println("Completed $current_progress / $(length(log_lags)) lags")
        end
    end

    # =====================================================================
    # PLOTTING
    # =====================================================================
    fs = 10_000
    
    # X-axis exactly matches the computed lags
    lag_axis_ms = log_lags ./ fs .* 1000

    # Safely truncate for the line of best fit at 50ms
    fit_cuton = 1
    fit_cutoff = findfirst(x -> x > 5, lag_axis_ms)
    x_trunc = lag_axis_ms[fit_cuton:fit_cutoff]
    y_trunc = time_avg_mean_ad_curve[fit_cuton:fit_cutoff]

    log_X = [ones(length(x_trunc)) log10.(x_trunc)]
    log_Y = log10.(y_trunc)

    c = log_X \ log_Y
    m = round(c[2], digits=2)
    α = round(1/c[2], digits=2)

    # Plot directly without `[2:end]` since log_lags does not contain 0
    mad_plot = plot(lag_axis_ms, time_avg_mean_ad_curve, 
        xlabel="Time Lag Δt (ms)", 
        ylabel="Mean Absolute Displacement (MAD)",
        title="Time-Averaged MAD Curve",
        label="Population Mean",
        xscale=:log10,
        yscale=:log10,
        linewidth=3,
        color=:blue,
        legend=:bottomright,
        grid=true)

    plot!(x_trunc, (x_trunc .^ c[2]) .* 10^(c[1]),
        label = "Best Fit (m=$m, α=$α)",
        linewidth=2,
        linestyle=:dash,
        color=:black
    )

    savefig(mad_plot, joinpath(output_dir, "time_avg_mean_ad_curve.png"))
    println("Done! Plot saved.")
end

mad_calculate("data", "results/MAD")
