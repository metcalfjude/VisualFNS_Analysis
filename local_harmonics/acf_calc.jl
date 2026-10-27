using DrWatson
@quickactivate :VisualFNS_Analysis

using CSV
using DataFrames
using DSP
using Plots
using Statistics
using LinearAlgebra
using FFTW

default(
    titlefontsize = 16,
    guidefontsize = 18,
    tickfontsize = 20,
    legendfontsize = 14,
    grid = false,
    framestyle = :box,
    dpi = 300,
    margin = 8Plots.mm,
    legend = :topright,
)

function normalized_autocorrelation(signal::AbstractVector, max_lag_samples::Int)
    centered = Float64.(signal) .- mean(signal)
    variance = sum(abs2, centered)
    variance > 0 || throw(ArgumentError("Cannot calculate autocorrelation for a constant signal."))

    n_samples = length(centered)
    autocorrelation = Vector{Float64}(undef, max_lag_samples + 1)
    for lag in 0:max_lag_samples
        n_pairs = n_samples - lag
        autocorrelation[lag + 1] =
            dot(centered[1:n_pairs], centered[(lag + 1):n_samples]) /
            (n_pairs * variance / n_samples)
    end
    return autocorrelation
end

function gradient(values::AbstractVector, spacing::Real)
    n_samples = length(values)
    n_samples >= 2 || throw(ArgumentError("At least two samples are required to calculate velocity"))
    result = Vector{Float64}(undef, n_samples)
    result[1] = (values[2] - values[1]) / spacing
    result[end] = (values[end] - values[end - 1]) / spacing
    for index in 2:(n_samples - 1)
        result[index] = (values[index + 1] - values[index - 1]) / (2 * spacing)
    end
    return result
end

function fourier_interpolate_acf(
    autocorrelation::AbstractVector,
    interpolation_factor::Int,
)
    interpolation_factor >= 1 ||
        throw(ArgumentError("interpolation_factor must be at least 1"))
    interpolation_factor == 1 && return Float64.(autocorrelation)

    # Mirror the non-negative lags so Fourier interpolation preserves the
    # autocorrelation's even symmetry.
    symmetric_acf = vcat(reverse(autocorrelation), autocorrelation)
    n_original = length(symmetric_acf)
    n_interpolated = interpolation_factor * n_original
    spectrum = fftshift(fft(symmetric_acf))
    padding = n_interpolated - n_original
    padded_spectrum = vcat(
        zeros(ComplexF64, padding ÷ 2),
        spectrum,
        zeros(ComplexF64, padding - padding ÷ 2),
    )
    interpolated = real.(ifft(ifftshift(padded_spectrum))) .* interpolation_factor

    zero_lag_index = length(autocorrelation) * interpolation_factor + 1
    interpolated_lag_count = (length(autocorrelation) - 1) * interpolation_factor
    return interpolated[zero_lag_index:(zero_lag_index + interpolated_lag_count)]
end

function acf_minimum_and_crossover(lags, acf)
    minimum_index = argmin(acf[2:end]) + 1
    minimum_lag = lags[minimum_index]
    minimum_value = acf[minimum_index]

    crossover_index = findfirst(
        i -> acf[i] < 0 <= acf[i + 1],
        minimum_index:(length(acf) - 1),
    )
    crossover_index === nothing &&
        return (minimum_lag, minimum_value, nothing)

    # Linear interpolation gives a sub-sample estimate of the zero crossing.
    i = crossover_index
    crossover_lag = lags[i] +
        (lags[i + 1] - lags[i]) * (-acf[i]) / (acf[i + 1] - acf[i])
    return (minimum_lag, minimum_value, crossover_lag)
end

function fixation_segments(data::DataFrame, file::AbstractString)
    required = (:t, :x, :y, :is_saccade)
    all(column -> column in propertynames(data), required) ||
        throw(ArgumentError(
            "$file must contain columns t, x, y, and is_saccade",
        ))

    all(isfinite, data.t) || throw(ArgumentError("$file contains non-finite t values"))
    all(isfinite, data.x) || throw(ArgumentError("$file contains non-finite x values"))
    all(isfinite, data.y) || throw(ArgumentError("$file contains non-finite y values"))
    issorted(data.t) || throw(ArgumentError("$file has unsorted t values"))
    all(diff(data.t) .> 0) || throw(ArgumentError("$file must have strictly increasing t values"))

    is_saccade = if eltype(data.is_saccade) <: Bool
        data.is_saccade
    else
        lowercase_string = lowercase.(strip.(string.(data.is_saccade)))
        all(value -> value == "true" || value == "false", lowercase_string) ||
            throw(ArgumentError("$file contains non-boolean is_saccade values"))
        lowercase_string .== "true"
    end
    is_fixation = .!is_saccade
    transitions = diff(vcat(false, is_fixation, false))
    starts = findall(==(1), transitions)
    stops = findall(==(-1), transitions) .- 1
    return [(Float64.(data.t[start:stop]),
             Float64.(data.x[start:stop]),
             Float64.(data.y[start:stop]))
            for (start, stop) in zip(starts, stops)]
end

function read_fixation_segments(input_dir::AbstractString)
    isdir(input_dir) || throw(ArgumentError("Input directory does not exist: $input_dir"))
    csv_files = sort(filter(file -> endswith(lowercase(file), ".csv"),
                            readdir(input_dir; join=true)))
    isempty(csv_files) && throw(ArgumentError("No CSV files found in $input_dir"))

    segments = Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}[]
    for file in csv_files
        data = CSV.read(file, DataFrame)
        append!(segments, fixation_segments(data, file))
    end
    isempty(segments) && throw(ArgumentError("No fixation samples found in $input_dir"))
    return segments
end

function sampling_rate(segments)
    sample_intervals = reduce(vcat, [diff(segment[1]) for segment in segments])
    isempty(sample_intervals) && throw(ArgumentError("At least two samples are required"))
    median_interval = median(sample_intervals)
    all(isapprox.(sample_intervals, median_interval; rtol=1e-3, atol=eps(Float64))) ||
        throw(ArgumentError("Input t values must have a consistent sampling interval"))
    return 1 / median_interval
end

function plot_fixation_acf(
    segments;
    fs=nothing,
    filter_freq=250,
    max_lag=0.5,
    interpolation_factor=1,
)
    max_lag > 0 || throw(ArgumentError("max_lag must be positive"))
    interpolation_factor >= 1 ||
        throw(ArgumentError("interpolation_factor must be at least 1"))

    fs = isnothing(fs) ? sampling_rate(segments) : Float64(fs)
    fs > 0 || throw(ArgumentError("fs must be positive"))
    0 < filter_freq < fs / 2 ||
        throw(ArgumentError("filter_freq must be between 0 and the Nyquist frequency"))
    max_lag_samples = round(Int, max_lag * fs)
    max_lag_samples > 0 || throw(ArgumentError("max_lag must contain at least one sample"))

    lpf = digitalfilter(Lowpass(filter_freq), Butterworth(4), fs=fs)
    acf_x_list = Vector{Float64}[]
    acf_y_list = Vector{Float64}[]
    skipped_short = 0
    skipped_constant = 0

    for (_, x, y) in segments
        n_samples = length(x)
        if n_samples <= max_lag_samples
            skipped_short += 1
            continue
        end

        x_pos_filtered = filtfilt(lpf, x)
        y_pos_filtered = filtfilt(lpf, y)
        x_velocity = gradient(x_pos_filtered, 1 / fs)
        y_velocity = gradient(y_pos_filtered, 1 / fs)

        try
            push!(acf_x_list, normalized_autocorrelation(x_velocity, max_lag_samples))
            push!(acf_y_list, normalized_autocorrelation(y_velocity, max_lag_samples))
        catch error
            if error isa ArgumentError
                skipped_constant += 1
            else
                rethrow()
            end
        end
    end

    isempty(acf_x_list) &&
        error("No fixation is longer than max_lag=$(max_lag) seconds.")
    skipped_short > 0 &&
        println("Skipped $skipped_short valid fixations shorter than max_lag=$(max_lag) seconds.")
    skipped_constant > 0 &&
        println("Skipped $skipped_constant fixations with constant velocity.")

    acf_x_mat = reduce(hcat, acf_x_list)
    acf_y_mat = reduce(hcat, acf_y_list)
    n_fixations = size(acf_x_mat, 2)

    mean_acf_x = fourier_interpolate_acf(
        vec(mean(acf_x_mat, dims=2)),
        interpolation_factor,
    )
    mean_acf_y = fourier_interpolate_acf(
        vec(mean(acf_y_mat, dims=2)),
        interpolation_factor,
    )
    ci95_x = fourier_interpolate_acf(
        1.96 .* vec(std(acf_x_mat, dims=2)) ./ sqrt(n_fixations),
        interpolation_factor,
    )
    ci95_y = fourier_interpolate_acf(
        1.96 .* vec(std(acf_y_mat, dims=2)) ./ sqrt(n_fixations),
        interpolation_factor,
    )
    lags = collect(0:(max_lag_samples * interpolation_factor)) /
        (fs * interpolation_factor)
    lags = lags .* 1000  # Convert to milliseconds

    minimum_x = acf_minimum_and_crossover(lags, mean_acf_x)
    minimum_y = acf_minimum_and_crossover(lags, mean_acf_y)
    println(
        "X-velocity ACF minimum: $(round(minimum_x[2], digits=4)) at " *
        "$(round(minimum_x[1], digits=3)) ms; crossover: " *
        (minimum_x[3] === nothing ? "not found" : "$(round(minimum_x[3], digits=3)) ms"),
    )
    println(
        "Y-velocity ACF minimum: $(round(minimum_y[2], digits=4)) at " *
        "$(round(minimum_y[1], digits=3)) ms; crossover: " *
        (minimum_y[3] === nothing ? "not found" : "$(round(minimum_y[3], digits=3)) ms"),
    )

    p = plot(
        lags,
        mean_acf_x,
        ribbon=(ci95_x, ci95_x),
        fillalpha=0.3,
        label="X-velocity",
        color="#0072B2",
        linewidth=4,
        xlabel="Lag (ms)",
        ylabel="Normalized autocorrelation",
        title="Fixation Autocorrelation with 95% CI\n$n_fixations fixations",
        size=(1000, 650),
        titlefontsize=24,
        guidefontsize=18,
        tickfontsize=20,
        legendfontsize=20,
        framestyle=:box,
        legend=:topright,
        margin=14Plots.mm,
        grid=true,
        minorticks=true,
    )

    plot!(
        p,
        lags,
        mean_acf_y,
        ribbon=(ci95_y, ci95_y),
        fillalpha=0.3,
        label="Y-velocity",
        color="#D55E00",
        linewidth=4,
    )
    hline!(p, [0], color=:black, linestyle=:dash, linewidth=2, label=nothing)

    scatter!(
        p,
        [minimum_x[1]],
        [minimum_x[2]],
        color="#0072B2",
        markersize=7,
        label="X minimum",
    )
    scatter!(
        p,
        [minimum_y[1]],
        [minimum_y[2]],
        color="#D55E00",
        markersize=7,
        label="Y minimum",
    )
    if minimum_x[3] !== nothing
        scatter!(
            p,
            [minimum_x[3]],
            [0],
            color="#0072B2",
            marker=:xcross,
            markersize=8,
            label="X crossover",
        )
    end
    if minimum_y[3] !== nothing
        scatter!(
            p,
            [minimum_y[3]],
            [0],
            color="#D55E00",
            marker=:xcross,
            markersize=8,
            label="Y crossover",
        )
    end
    return p
end

IN_DIR  = "./results/ISI_Modelling/τ=0.0002.csv/Annotated_Data"
SAVE_DIR = "./results/ACF_Results"

fixation_data = read_fixation_segments(IN_DIR)
println("Found $(length(fixation_data)) fixation segments in $IN_DIR.")

acf_plot = plot_fixation_acf(
    fixation_data,
    max_lag=0.05,
    interpolation_factor=4,
)
display(acf_plot)

mkpath(SAVE_DIR)
savefig(acf_plot, joinpath(SAVE_DIR, "fixation_acf_plot.pdf"))
