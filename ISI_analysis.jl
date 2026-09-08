using Pkg
Pkg.activate(".")

using SavitzkyGolay
using StatsBase
using CSV
using DataFrames
using Statistics
using Plots
using Distributions

# ==========================================
# Configuration & Plotting Defaults
# ==========================================

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

# ==========================================
# Core Functions
# ==========================================

function get_sg_velocity(x::AbstractVector, y::AbstractVector, window_size::Int, dt::Float64)
    # Ensure window is odd
    window_size = iseven(window_size) ? window_size - 1 : window_size
    poly_order = 3
    
    sg_x = savitzky_golay(x, window_size, poly_order, deriv=1, rate=1/dt)
    sg_y = savitzky_golay(y, window_size, poly_order, deriv=1, rate=1/dt)
    
    vx = sg_x.y
    vy = sg_y.y
    
    return sqrt.(vx.^2 .+ vy.^2)
end

function find_med(x::AbstractVector, y::AbstractVector, timeout_smoother::Float64, dt::Float64)
    timeout_smoother_steps = Int(ceil(timeout_smoother / dt))
    v = get_sg_velocity(x, y, timeout_smoother_steps, dt)
    # Threshold: scaled median absolute deviation
    return median(abs.(v .- median(v))) / 0.6745
end

function calculate_isi(x::AbstractVector, y::AbstractVector, msd::Float64; 
                       timeout_smoother=0.007, timeout=0.150, λ=6.0, 
                       ratio=0.1, dt=0.0001, min_dur_steps=5)
                       
    timeout_steps = Int(ceil(timeout / dt))
    timeout_smoother_steps = Int(ceil(timeout_smoother / dt))
    v = get_sg_velocity(x, y, timeout_smoother_steps, dt)

    threshold_on = λ * msd
    threshold_off = (ratio * λ) * msd

    starts_timed = Int[]
    last_start = -timeout_steps 
    
    i = 1
    n = length(v)
    
    while i <= n
        if v[i] > threshold_on
            j = i + 1
            while j <= n && v[j] >= threshold_off
                j += 1
            end
            
            event_duration = j - i
            
            if event_duration >= min_dur_steps && (i - last_start) > timeout_steps
                push!(starts_timed, i)
                last_start = i
            end
            i = j
        else
            i += 1
        end
    end

    isi = diff(starts_timed) .* (dt * 1000) # Convert to ms
    return starts_timed, isi
end

# ==========================================
# Main Execution Block
# ==========================================

function analyse_saccades(input_dir::String, output_dir::String; plot_poincare::Bool=false)
    isdir(output_dir) || mkdir(output_dir)
    csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))
    
    if isempty(csv_files)
        error("No CSV files found in $input_dir")
    end

    dt = 0.0001
    N = length(csv_files)
    
    # Read files and calculate global noise (μ)
    println("Pass 1: Calculating global noise baseline on $(Threads.nthreads()) threads...")
    
    datasets = Vector{DataFrame}(undef, N)
    mus = Vector{Float64}(undef, N)
    
    Threads.@threads for i in 1:N
        df = CSV.read(csv_files[i], DataFrame)
        datasets[i] = df
        mus[i] = find_med(df.value1, df.value2, 2.0, dt)
    end
    
    μ_global = sum(mus) / N
    
    # Extract ISIs and generate trace plots
    println("Pass 2: Extracting ISIs...")
    
    isi_collections = Vector{Vector{Float64}}(undef, N)
    
    # Create a lock strictly for Plots.jl operations
    plot_lock = ReentrantLock()
    
    Threads.@threads for i in 1:N
        file = csv_files[i]
        df = datasets[i]
        x, y = df.value1, df.value2

        sac_starts, isi = calculate_isi(x, y, μ_global; 
            timeout_smoother=0.2, 
            timeout=0.05,
            ratio=0.1,
            λ=1.2,
            dt=dt, 
            min_dur_steps=1000
        )

        # Safely store data in this thread's designated index
        isi_collections[i] = isi

        if length(isi) > 5
            time_axis = 0.1 .* (1:length(x))
            
            # LOCK: Only one thread can interact with Plots.jl at a time
            # Convert steps to seconds to match frequency analysis scales
            time_axis_sec = (1:length(x)) .* dt

            lock(plot_lock) do
                # Match the darkblue/darkorange aesthetic from freq_analyse.jl
                p_trace = plot(time_axis_sec, x, label="X Position", color=:darkblue, alpha=0.8, linewidth=1.5)
                plot!(p_trace, time_axis_sec, y, label="Y Position", color=:darkorange, alpha=0.8, linewidth=1.5)
                
                # Add dashed vertical lines for saccades
                if !isempty(sac_starts)
                    # Plot the first line with a label for the legend
                    vline!(p_trace, [sac_starts[1] * dt], color=:black, linestyle=:dash, linewidth=1.5, alpha=0.6, label="Saccade Onset")
                    # Plot the rest without adding duplicate legend entries
                    if length(sac_starts) > 1
                        vline!(p_trace, sac_starts[2:end] .* dt, color=:black, linestyle=:dash, linewidth=1.5, alpha=0.6, label=false)
                    end
                end
                
                # Define sensible window limits in seconds rather than raw indices
                plot!(p_trace, 
                    xlims=(0.1, 0.4), # Example: View from 0.1s to 0.4s
                    xlabel="Time (s)", 
                    ylabel="Position (arb. units)",
                    title="Saccade Detection Trace"
                )
                
                outname = joinpath(output_dir, splitext(basename(file))[1] * "-SACCADES.pdf")
                savefig(p_trace, outname)
            end
        end
        println("Processed file $i/$N: $(basename(file))")
    end

    # Sequentially flatten the arrays back into a single vector
    isi_vec = reduce(vcat, isi_collections)

    if isempty(isi_vec)
        error("No ISIs detected across all files.")
    end

    # Statistics
    println("Pass 3: Statistical Modeling...")
    timelength = 1000
    plotting_x = range(0, stop=timelength, length=500)

    # Log-Normal Fit
    lISI = log.(isi_vec)
    log_σ = std(lISI)
    log_peak = median(lISI)
    RI = 1 / sqrt(exp(log_σ^2) - 1)
    fitted_ln = LogNormal(log_peak, log_σ)

    # Gamma Fits
    order = 10.0
    scale_param = mean(isi_vec) / order
    Gamma_fit_fixed = Gamma(order, scale_param)
    general_Gamma_fit = fit(Gamma, isi_vec) 

    # Stability Metrics (CV2 & Poincaré)
    CV_2 = Float64[]
    for i in 1:(length(isi_vec)-1)
        denom = isi_vec[i+1] + isi_vec[i]
        if denom > 0
            push!(CV_2, 2 * abs(isi_vec[i+1] - isi_vec[i]) / denom)
        end
    end
    RCRI = 1 / (1 + mean(CV_2))

    xs = isi_vec[1:end-1]
    ys = isi_vec[2:end]
    sd1 = std(ys .- xs) / sqrt(2)
    sd2 = std(ys .+ xs) / sqrt(2)

    println("\n--- Rhythm Stability Metrics ---")
    println("SD1 (Short-term jitter): ", round(sd1, digits=2))
    println("SD2 (Long-term drift):   ", round(sd2, digits=2))
    println("Ratio (SD2/SD1):         ", round(sd2/sd1, digits=2))
    println("Log SD:                  ", round(log_σ, digits=3))
    println("RCRI:                    ", round(RCRI, digits=3))

    # 4. Generate Aggregate Plots (Keep sequential)
    p_gamma = histogram(isi_vec, bins=range(0, timelength, length=40), normalize=:pdf,
        xlabel="Inter-Saccadic Interval (ms)", ylabel="Probability Density",
        xlims=(0, timelength), ylims=(0, maximum(pdf.(Gamma_fit_fixed, plotting_x))*2), label="Simulated ISI Data",
        color=:steelblue, linecolor=:white, linewidth=0.5, fillalpha=0.7)
    plot!(p_gamma, plotting_x, pdf.(Gamma_fit_fixed, plotting_x), 
        label="Gamma-10 (RCRI=$(round(RCRI, digits=3)))", color=:darkred, linewidth=2.5)
    savefig(p_gamma, joinpath(output_dir, "Total_ISIs_RCRI.pdf"))
 
    p_lognorm = histogram(isi_vec, bins=range(0, timelength, length=40), normalize=:pdf,
        xlabel="Inter-Saccadic Interval (ms)", ylabel="Probability Density",
        xlims=(0, timelength), ylims=(0, maximum(pdf.(fitted_ln, plotting_x))*2), label="Simulated ISI Data", 
        color=:steelblue, linecolor=:white, linewidth=0.5, fillalpha=0.7)
    plot!(p_lognorm, plotting_x, pdf.(fitted_ln, plotting_x), 
        label="Log-Normal (RI = $(round(RI, digits=3)))", color=:darkorange, linewidth=2.5)
    savefig(p_lognorm, joinpath(output_dir, "Total_ISIs.pdf"))

    p_ts = plot(isi_vec, xlabel="Index", ylabel="ISI (ms)", title="ISI Timeseries",
        label=false, color=:gray30, linewidth=1.0, linealpha=0.7)
    savefig(p_ts, joinpath(output_dir, "ISI_TIMESERIES.pdf"))

    if plot_poincare
        ax_min = min(minimum(xs), minimum(ys)) * 0.95
        ax_max = max(maximum(xs), maximum(ys)) * 1.05

        p_poincare = scatter(xs, ys, label="Intervals", alpha=0.5, markersize=2.5,
            markerstrokewidth=0, color=:steelblue, aspect_ratio=:equal, 
            xlims=(ax_min, ax_max), ylims=(ax_min, ax_max),
            title="Poincaré Plot (Return Map)", xlabel="Interval t (ms)",
            ylabel="Interval t+1 (ms)", legend=:topleft)
        
        plot!(p_poincare, [ax_min, ax_max], [ax_min, ax_max], 
            label="Identity (y=x)", color=:firebrick, linestyle=:dash, linewidth=1.5)
            
        savefig(p_poincare, joinpath(output_dir, "POINCARE.pdf"))
    end

    # 5. Export Data
    CSV.write(joinpath(output_dir, "ISI_Timeseries_Data.csv"), DataFrame(ISI = isi_vec))
    
    df_metrics = DataFrame(
        TimeLength = timelength,
        LogPeak = log_peak,
        LogSigma = log_σ,
        RI = RI,
        SD1 = sd1,
        SD2 = sd2,
        RCRI = RCRI
    )
    CSV.write(joinpath(output_dir, "ISI_Metrics.csv"), df_metrics)

    println("Saved ISI data and metrics successfully!")
end

# ==========================================
# Script Execution
# ==========================================

# Replace these with relative paths for publication
IN_DIR  = "./data" 
OUT_DIR = "./results/ISI_Parallel"

# Run the analysis
analyse_saccades(IN_DIR, OUT_DIR, plot_poincare=false)
