
using DrWatson
@quickactivate :VisualFNS_Analysis

using CSV
using DataFrames
using Statistics
using Plots
using Distributions
using Random

# ==========================================
# Configuration
# ==========================================

FOLDER_1 = "./results/ISI_Modelling/τ=0.0002.csv"
FOLDER_2 = "./results/ISI_Modelling/τ=0.00025.csv"
FOLDER_3 = "./results/ISI_Modelling/τ=0.0003.csv"

REFERENCE_TIMES = [2, 2.5, 3.0] .* (1e-4 * (80/3)^2) * 1e3

OUTPUT_DIR = "./results/ISI_Comparison"
OUTPUT_FILE = joinpath(OUTPUT_DIR, "Combined_ISI_Histograms.pdf")
METRICS_FILE = joinpath(OUTPUT_DIR, "ISI_Bootstrap_Metrics.csv")

ISI_FILENAME = "ISI_Timeseries_Data.csv"

BIN_WIDTH = 15.0
X_LIMITS = (0, 600)
NORMALIZE = :pdf

LABELS = [
    "142.22 ms",
    "177.78 ms",
    "213.33 ms"
]

# ==========================================
# Load ISI data
# ==========================================

function load_isi(folder::String; filename=nothing)
    isdir(folder) || error("Folder does not exist: $folder")

    files = if isnothing(filename)
        filter(f -> endswith(lowercase(f), ".csv"),
               readdir(folder; join=true))
    else
        found = String[]
        for (root, _, filenames) in walkdir(folder)
            if filename in filenames
                push!(found, joinpath(root, filename))
            end
        end
        found
    end

    isempty(files) && error("No matching CSV files found in $folder")

    isi_values = Float64[]

    for file in files
        df = CSV.read(file, DataFrame)

        col = if "ISI" in names(df)
            "ISI"
        elseif "isi" in names(df)
            "isi"
        else
            error("No ISI column found in $file. Columns: $(names(df))")
        end

        values = Float64.(df[!, col])
        filter!(x -> isfinite(x) && x > 0, values)
        append!(isi_values, values)
    end

    isempty(isi_values) && error("No valid ISI values found in $folder")

    println("Loaded $(length(isi_values)) ISIs from $(length(files)) file(s) in $folder")
    return isi_values
end

# ==========================================
# Load all three conditions
# ==========================================

folders = [FOLDER_1, FOLDER_2, FOLDER_3]

isi_data = [
    load_isi(folder; filename=ISI_FILENAME)
    for folder in folders
]

upper_limit = isnothing(X_LIMITS) ?
    maximum(maximum.(isi_data)) : X_LIMITS[2]

bins = 0:BIN_WIDTH:(ceil(upper_limit / BIN_WIDTH) * BIN_WIDTH)
x_fit = range(0.01, stop=upper_limit, length=1200)

# ==========================================
# Gamma fits
# ==========================================

gamma_fits = [fit(Gamma, data) for data in isi_data]

# ==========================================
# Plot histograms and Gamma fits
# ==========================================

alphas = [0.8, 0.7, 0.5]

BAR_COLORS = [
    "#cc4c02",
    "#fc8d59",
    "#fec44f"
]

FIT_COLORS = [
    "#8c2d04",
    "#d94801",
    "#fd8d3c"
]

#f16913
#d94801
#8c2d04

p = plot(
    xlabel="Inter-saccadic interval (ms)",
    ylabel="Probability density",
    xlims=isnothing(X_LIMITS) ? (0, upper_limit) : X_LIMITS,
    size=(1050, 720),
    guidefontsize=30,
    tickfontsize=30,
    legendfontsize=22,
    framestyle=:box,
    legend=:topright,
    background_color=:white,
    foreground_color=:black,
    grid=true,
    gridalpha=0.15,
    margin=12Plots.mm,
    ylims=(0, 0.007)
)

for i in 1:3
    histogram!(
        p, isi_data[i];
        bins=bins,
        normalize=:pdf,
        label=LABELS[i],
        color=BAR_COLORS[i],
        linecolor=:white,
        linewidth=0.5,
        fillalpha=alphas[i],
        bar_position=:overlay
    )
end

for i in 1:3
    plot!(
        p, x_fit, pdf.(gamma_fits[i], x_fit);
        color=FIT_COLORS[i],
        linewidth=5.0,
        linestyle=:dash,
        alpha=1.0,
        label=false
    )

    # vline!(
    #     p, [REFERENCE_TIMES[i]];
    #     color=FIT_COLORS[i],
    #     linestyle=:dot,
    #     linewidth=4,
    #     alpha=0.95,
    #     label=false
    # )
end

mkpath(OUTPUT_DIR)
savefig(p, OUTPUT_FILE)
println("Saved plot to: $OUTPUT_FILE")

# ==========================================
# Calculate RI and RCRI
# ==========================================

function calculate_isi_metrics(isi::Vector{Float64})
    length(isi) >= 2 ||
        error("At least two ISIs are required.")

    log_sigma = std(log.(isi))
    geoCV = sqrt(expm1(log_sigma^2))
    RI = geoCV > 0 ? 1 / geoCV : Inf

    cv2_values = [
        2 * abs(isi[i+1] - isi[i]) / (isi[i+1] + isi[i])
        for i in 1:(length(isi)-1)
    ]

    CV_2 = mean(cv2_values)

    # Retaining your current definition:
    # RCRI = 1 / (1 + CV2)
    RCRI = 1 / (1 + CV_2)

    return (
        RI=RI,
        RCRI=RCRI,
        CV_2=CV_2
    )
end

# ==========================================
# Bootstrap with replacement
# ==========================================

function calculate_all_metrics(isi::Vector{Float64})
    gamma_fit = fit(Gamma, isi)
    rhythm = calculate_isi_metrics(isi)

    gamma_shape = shape(gamma_fit)
    gamma_scale = scale(gamma_fit)

    # Gamma mode is zero for shape <= 1
    gamma_mode = gamma_shape > 1 ?
        (gamma_shape - 1) * gamma_scale : 0.0

    return (
        Gamma_shape=gamma_shape,
        Gamma_scale=gamma_scale,
        Gamma_mode=gamma_mode,
        RI=rhythm.RI,
        RCRI=rhythm.RCRI
    )
end

function bootstrap_isi(
    isi::Vector{Float64};
    nboot::Int=2000,
    seed::Int=1234
)
    rng = MersenneTwister(seed)
    n = length(isi)

    observed = calculate_all_metrics(isi)
    metric_names = collect(keys(observed))

    boot = Dict(name => Float64[] for name in metric_names)

    for b in 1:nboot
        # Sample the original number of ISIs with replacement
        sample = isi[rand(rng, 1:n, n)]

        try
            result = calculate_all_metrics(sample)

            for name in metric_names
                value = getproperty(result, name)
                if isfinite(value)
                    push!(boot[name], value)
                end
            end
        catch
            # Skip failed bootstrap fits
        end
    end

    ci = Dict{Symbol, Tuple{Float64, Float64}}()

    for name in metric_names
        values = boot[name]

        ci[name] = length(values) >= 0.9 * nboot ?
            (quantile(values, 0.025), quantile(values, 0.975)) :
            (NaN, NaN)
    end

    return observed, ci
end

# ==========================================
# Print and save metrics with 95% CIs
# ==========================================

N_BOOT = 2000

all_results = DataFrame(
    Condition=String[],
    Metric=String[],
    Estimate=Float64[],
    CI_lower=Float64[],
    CI_upper=Float64[]
)

println("\n========== ISI FIT & BOOTSTRAP METRICS ==========")

for i in eachindex(isi_data)
    observed, ci = bootstrap_isi(
        isi_data[i];
        nboot=N_BOOT,
        seed=1234 + i
    )

    rhythm = calculate_isi_metrics(isi_data[i])
    fitted = gamma_fits[i]

    println("\n$(LABELS[i])")
    println("  Number of ISIs = $(length(isi_data[i]))")
    println("  Gamma shape    = $(round(shape(fitted), digits=4))")
    println("  Gamma scale    = $(round(scale(fitted), digits=4)) ms")

    mode_est = shape(fitted) > 1 ?
        (shape(fitted) - 1) * scale(fitted) : 0.0
    println("  Gamma mode     = $(round(mode_est, digits=4)) ms")

    println("  RI             = $(round(rhythm.RI, digits=5))")
    println("  RCRI           = $(round(rhythm.RCRI, digits=5))")

    for name in keys(observed)
        estimate = getproperty(observed, name)
        lo, hi = ci[name]

        println(
            "  $(name) 95% CI = [" *
            "$(round(lo, digits=5)), $(round(hi, digits=5))]"
        )

        push!(all_results, (
            LABELS[i],
            string(name),
            estimate,
            lo,
            hi
        ))
    end
end

CSV.write(METRICS_FILE, all_results)

println("\nMetrics saved to: $METRICS_FILE")
println("=================================================")
