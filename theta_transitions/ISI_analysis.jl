using DrWatson
@quickactivate :VisualFNS_Analysis

using SavitzkyGolay
using StatsBase
using CSV
using DataFrames
using Statistics
using Plots
using Distributions

# ==========================================
# Core Functions
# ==========================================

function get_sg_velocity(x::AbstractVector, y::AbstractVector, window_size::Int, dt::Float64)
    length(x) == length(y) || throw(ArgumentError("x and y must have the same length"))
    length(x) > 1 || throw(ArgumentError("x and y must contain more than one sample"))

    # Ensure window is odd
    window_size = iseven(window_size) ? window_size - 1 : window_size
    poly_order = 3
    max_window = isodd(length(x)) ? length(x) : length(x) - 1
    window_size = min(window_size, max_window)
    window_size > poly_order || throw(ArgumentError(
        "smoothing window must contain more than $poly_order samples"
    ))
    
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
    return median(abs.(v .- median(v)))
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
    stops_timed = Int[]
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
                push!(stops_timed, min(j, n))
                last_start = i
            end
            i = j
        else
            i += 1
        end
    end

    isi = diff(starts_timed) .* (dt * 1000) # Convert to ms
    return starts_timed, stops_timed, isi
end

# ==========================================
# Main Execution Block
# ==========================================

function analyse_saccades(input_dir::String, save_folder::String, file_end::String; plot_poincare::Bool=false, timeout_smoother=0.2, timeout=0.05, ratio=0.1, λ=1.2, dt=dt, min_dur_steps=1000, gamma_shape=10.0, fit_gamma::Bool=true)
    gamma_shape > 0 || throw(ArgumentError("gamma_shape must be positive"))
    
    output_dir = joinpath(save_folder, file_end)
    individual_dir = joinpath(output_dir, "individual")
    isdir(save_folder) || mkdir(save_folder)
    isdir(output_dir) || mkdir(output_dir)
    isdir(individual_dir) || mkdir(individual_dir)

    csv_files = filter(f -> endswith(f, file_end), readdir(input_dir; join=true))
    
    if isempty(csv_files)
        error("No CSV files found in $input_dir")
    end

    N = length(csv_files)
    min_samples = Int(ceil(2.0 / dt))
    
    # Read files and calculate global noise (μ)
    println("Pass 1: Calculating global noise baseline on $(Threads.nthreads()) threads...")
    
    datasets = Vector{Union{Nothing, DataFrame}}(undef, N)
    mus = Vector{Float64}(undef, N)
    valid = falses(N)
    
    Threads.@threads for i in 1:N
        df_total = CSV.read(csv_files[i], DataFrame)
        if nrow(df_total) > min_samples
            datasets[i] = df_total
            mus[i] = find_med(df_total.x, df_total.y, 1.0, dt)
            valid[i] = true
        end
    end

    valid_indices = findall(valid)
    isempty(valid_indices) && error("No CSV files contain more than 2 seconds of data.")
    csv_files = csv_files[valid_indices]
    datasets = [datasets[i]::DataFrame for i in valid_indices]
    mus = mus[valid_indices]
    N = length(csv_files)
    
    μ_global = sum(mus) / N
    
    # Extract ISIs and generate trace plots
    println("Pass 2: Extracting ISIs...")
    
    isi_collections = Vector{Vector{Float64}}(undef, N)
    
    # Create a lock strictly for Plots.jl operations
    plot_lock = ReentrantLock()
    
    Threads.@threads for i in 1:N
        file = csv_files[i]
        df = datasets[i]
        x, y = df.x, df.y

        # Unpack the new stops_timed variable
        sac_starts, sac_stops, isi = calculate_isi(x, y, μ_global; 
            timeout_smoother=timeout_smoother, timeout=timeout, ratio=ratio, λ=λ, dt=dt, min_dur_steps=min_dur_steps
        )

        isi_collections[i] = isi

        # --- NEW EXPORT SECTION ---
        # Create a boolean mask for saccades
        is_saccade = falses(length(x))
        for (start_idx, stop_idx) in zip(sac_starts, sac_stops)
            is_saccade[start_idx:stop_idx] .= true
        end
        
        # Append to a copy of the dataframe and save
        df_out = copy(df)
        df_out[!, :is_saccade] = is_saccade
        
        annotated_dir = joinpath(output_dir, "Annotated_Data")
        isdir(annotated_dir) || mkpath(annotated_dir)
        
        out_csv = joinpath(annotated_dir, "annotated_" * basename(file))
        CSV.write(out_csv, df_out)
        # --------------------------

        if length(isi) > 5
            time_axis = 0.1 .* (1:length(x))
            
            # LOCK: Only one thread can interacoutput_dir * "/individual"t with Plots.jl at a time
            # Convert steps to seconds to match frequency analysis scales
            time_axis_sec = (1:length(x)) .* dt

            lock(plot_lock) do
                # Use a colour-blind-friendly blue/vermillion pair for the two eyes.
                p_trace = plot(time_axis_sec, x, label="Left eye", color="#0072B2", alpha=0.9, linewidth=4.0,
                    size=(1000, 650), titlefontsize=17, guidefontsize=18, tickfontsize=13,
                    legendfontsize=12, framestyle=:box, legend=:topright, margin=14Plots.mm)
                plot!(p_trace, time_axis_sec, y, label="Right eye", color="#D55E00", alpha=0.9, linewidth=4.0)
                
                # Add dashed vertical lines for saccades
                if !isempty(sac_starts)
                    # Plot the first line with a label for the legend
                    vline!(p_trace, [sac_starts[1] * dt], color=:black, linestyle=:dash, linewidth=3.0, alpha=0.6, label="Saccade Onset")
                    # Plot the rest without adding duplicate legend entries
                    if length(sac_starts) > 1
                        vline!(p_trace, sac_starts[2:end] .* dt, color=:black, linestyle=:dash, linewidth=3.0, alpha=0.6, label=false)
                    end
                end
                
                # Define sensible window limits in seconds rather than raw indices
                plot!(p_trace, 
                    xlims=(1.0, 3.1), # Example: View from 1.0s to 3.1s
                    xlabel="Time (s)", 
                    ylabel="Position (px)",
                    title="Saccade detection"
                )
                
                outname = joinpath(individual_dir, splitext(basename(file))[1] * "-SACCADES.pdf")
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
    timelength = 600  # ms
    plotting_x = range(0, stop=timelength, length=500)

    # Log-Normal Fit
    lISI = log.(isi_vec)
    log_σ = std(lISI)
    log_peak = median(lISI)
    RI = 1 / sqrt(exp(log_σ^2) - 1)
    fitted_ln = LogNormal(log_peak, log_σ)

    # Gamma fit. By default, both shape and scale are estimated by maximum likelihood.
    scale_param = mean(isi_vec) / gamma_shape
    gamma_fit = fit_gamma ? fit(Gamma, isi_vec) : Gamma(gamma_shape, scale_param)
    fitted_gamma_shape = shape(gamma_fit)
    fitted_gamma_scale = scale(gamma_fit)

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
    println("Gamma shape:             ", round(fitted_gamma_shape, digits=3))
    println("Gamma scale (ms):        ", round(fitted_gamma_scale, digits=3))
    println("RCRI:                    ", round(RCRI, digits=3))

    # 4. Generate Aggregate Plots (Keep sequential)
    reference_isi_ms = 1000 * 0.0002 * (80 / 3)^2
    reference_label = "Model (tθ = $(round(reference_isi_ms, digits=1)) ms)"
    gamma_peak_ms = fitted_gamma_shape >= 1 ?
        (fitted_gamma_shape - 1) * fitted_gamma_scale : 0.0
    gamma_peak_label = "Gamma fit peak ($(round(gamma_peak_ms, digits=1)) ms)"

    p_gamma = histogram(isi_vec, bins=range(0, timelength, length=40), normalize=:pdf,
        xlabel="Inter-saccadic interval (ms)", ylabel="Probability density",
        xlims=(0, timelength), ylims=(0, maximum(pdf.(gamma_fit, plotting_x))*2), label="Simulated ISI Data",
        color=:steelblue, linecolor=:white, linewidth=1.0, fillalpha=0.7,
        size=(900, 650), title="Inter-saccadic interval distribution",
        titlefontsize=16, guidefontsize=16, tickfontsize=11, legendfontsize=11,
        framestyle=:box, legend=:topright, margin=14Plots.mm)
    plot!(p_gamma, plotting_x, pdf.(gamma_fit, plotting_x),
        label="Gamma (shape = $(round(fitted_gamma_shape, digits=3)), scale = $(round(fitted_gamma_scale, digits=2)) ms)",
        color=:darkred, linewidth=5.0)
    vline!(p_gamma, [reference_isi_ms], label=reference_label, color=:gray50,
        linestyle=:dash, linewidth=6.0)
    vline!(p_gamma, [gamma_peak_ms], label=gamma_peak_label, color=:black,
        linestyle=:dot, linewidth=4.4)
    savefig(p_gamma, joinpath(output_dir, "Total_ISIs_RCRI.pdf"))
 
    p_lognorm = histogram(isi_vec, bins=range(0, timelength, length=40), normalize=:pdf,
        xlabel="Inter-saccadic interval (ms)", ylabel="Probability density",
        xlims=(0, timelength), ylims=(0, maximum(pdf.(fitted_ln, plotting_x))*2), label="Simulated ISI Data", 
        color=:steelblue, linecolor=:white, linewidth=1.0, fillalpha=0.7,
        size=(900, 650), title="Inter-saccadic interval distribution",
        titlefontsize=16, guidefontsize=16, tickfontsize=11, legendfontsize=11,
        framestyle=:box, legend=:topright, margin=14Plots.mm)
    plot!(p_lognorm, plotting_x, pdf.(fitted_ln, plotting_x), 
        label="Log-normal (RI = $(round(RI, digits=3)))", color=:darkorange, linewidth=5.0)
    vline!(p_lognorm, [reference_isi_ms], label=reference_label, color=:gray50,
        linestyle=:dash, linewidth=6.0)
    vline!(p_lognorm, [gamma_peak_ms], label=gamma_peak_label, color=:black,
        linestyle=:dot, linewidth=4.4)
    savefig(p_lognorm, joinpath(output_dir, "Total_ISIs.pdf"))

    p_ts = plot(isi_vec, xlabel="Saccade index", ylabel="Inter-saccadic interval (ms)",
        title="Inter-saccadic interval time series", label=false, color=:gray30,
        linewidth=2.0, linealpha=0.7, size=(900, 600), titlefontsize=16,
        guidefontsize=16, tickfontsize=11, legendfontsize=11, framestyle=:box,
        margin=14Plots.mm)
    savefig(p_ts, joinpath(output_dir, "ISI_TIMESERIES.pdf"))

    if plot_poincare
        ax_min = min(minimum(xs), minimum(ys)) * 0.95
        ax_max = max(maximum(xs), maximum(ys)) * 1.05

        p_poincare = scatter(xs, ys, label="Intervals", alpha=0.5, markersize=2.5,
            markerstrokewidth=0, color=:steelblue, aspect_ratio=:equal, 
            xlims=(ax_min, ax_max), ylims=(ax_min, ax_max),
            title="Poincaré plot (return map)", xlabel="Inter-saccadic interval t (ms)",
            ylabel="Inter-saccadic interval t+1 (ms)", legend=:topleft,
            size=(700, 700), titlefontsize=16, guidefontsize=16, tickfontsize=11,
            legendfontsize=11, framestyle=:box, margin=14Plots.mm)
        
        plot!(p_poincare, [ax_min, ax_max], [ax_min, ax_max], 
            label="Identity (y=x)", color=:firebrick, linestyle=:dash, linewidth=3.0)
            
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
IN_DIR  = "/home/ewig/Desktop/Thesis_Sims/Thesis_Metrics/data/STD-Params"
SAVE_DIR = "./results/ISI_Modelling/"

# Run the analysis
dt = 0.0001

analyse_saccades(
    IN_DIR,
    SAVE_DIR,
    "τ=0.0003.csv";
    plot_poincare=false,
    timeout_smoother=0.06,  # 60 ms
    timeout=0.025,           # 25 ms
    ratio=0.004,
    λ=12.0,
    dt=dt,
    min_dur_steps=ceil(Int, 0.006 / dt),  # 6 ms
    fit_gamma=true
)
