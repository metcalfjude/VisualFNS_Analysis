using Pkg
Pkg.activate(".")

using DSP
using CSV
using DataFrames
using Plots
using Statistics
using Printf

const dt = 0.0001

function superscript_integer(value::Int)
    superscripts = Dict(
        '0' => '⁰', '1' => '¹', '2' => '²', '3' => '³', '4' => '⁴',
        '5' => '⁵', '6' => '⁶', '7' => '⁷', '8' => '⁸', '9' => '⁹',
        '-' => '⁻'
    )
    return join(superscripts[c] for c in string(value))
end

function gradient(v::AbstractVector, dx::Real=1.0)
    n = length(v)
    grad = similar(v)

    n < 2 && return grad

    grad[1] = (v[2] - v[1]) / dx
    grad[end] = (v[end] - v[end-1]) / dx

    # Using central difference
    @inbounds for i in 2:(n-1)
        grad[i] = (v[i+1] - v[i-1]) / (2 * dx)
    end

    return grad
end

function freq_analyse(input_dir::String, output_dir::String;
                      welch_window_length::Int=512,
                      welch_overlap_fraction::Float64=0.5,
                      nfft::Int=4096,
                      filter_freq::Real=150.0)
    welch_window_length > 1 || throw(ArgumentError("welch_window_length must be greater than 1"))
    0.0 <= welch_overlap_fraction < 1.0 || throw(ArgumentError(
        "welch_overlap_fraction must be in the range [0, 1)"
    ))
    nfft >= welch_window_length || throw(ArgumentError("nfft must be at least welch_window_length"))
    0 < filter_freq < 0.5 / dt || throw(ArgumentError(
        "filter_freq must be between 0 and the Nyquist frequency"
    ))
    isdir(output_dir) || mkdir(output_dir)

    csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))
    tot = length(csv_files)
    total_segments = 0
    files_with_segments = 0
    total_fixation_segments = 0

    fs = 1 / dt # Restored missing sampling frequency definition
    welch_size_overlap = floor(Int, welch_window_length * welch_overlap_fraction)
    frequency_resolution = fs / welch_window_length
    lpf = digitalfilter(Lowpass(filter_freq), Butterworth(4), fs=fs)
    psd_x_list = Vector{Float64}[]
    psd_y_list = Vector{Float64}[]
    freqs = Float64[]
    skipped_short = 0

    for (i, file) in enumerate(csv_files)
        println("Processing file $i of $tot: $file")
        data = CSV.read(file, DataFrame)

        # We only want to analyze data where is_saccade is false (i.e., fixations)
        is_fixation = .!data.is_saccade

        # Find the start and end indices of continuous fixation blocks
        diff_fix = diff(vcat(false, is_fixation, false))
        fix_starts = findall(==(1), diff_fix)
        fix_stops = findall(==(-1), diff_fix) .- 1
        file_segments = 0

        for (st, sp) in zip(fix_starts, fix_stops)
            segment_length = sp - st + 1
            total_fixation_segments += 1
            
            # Only process segments long enough for the Welch window
            if segment_length >= welch_window_length
                # Extract the raw position segment
                seg_x = data.x[st:sp]
                seg_y = data.y[st:sp]

                # Filter and differentiate within each fixation to avoid saccade artifacts.
                seg_x = filtfilt(lpf, seg_x)
                seg_y = filtfilt(lpf, seg_y)
                v_x = gradient(seg_x, dt)
                v_y = gradient(seg_y, dt)

                v_x .-= mean(v_x)
                v_y .-= mean(v_y)

                pgram_x = welch_pgram(v_x, welch_window_length, welch_size_overlap;
                    nfft=nfft, fs=fs, window=hanning)
                pgram_y = welch_pgram(v_y, welch_window_length, welch_size_overlap;
                    nfft=nfft, fs=fs, window=hanning)

                if isempty(freqs)
                    freqs = pgram_x.freq
                elseif pgram_x.freq != freqs
                    error("Welch frequency grid changed between fixations.")
                end

                push!(psd_x_list, pgram_x.power)
                push!(psd_y_list, pgram_y.power)
                total_segments += 1
                file_segments += 1
            else
                skipped_short += 1
            end
        end
        files_with_segments += ifelse(file_segments > 0, 1, 0)
    end

    if isempty(psd_x_list)
        error("No valid inter-saccadic segments found of length >= $(welch_window_length) across all files.")
    end

    psd_x_mat = reduce(hcat, psd_x_list)
    psd_y_mat = reduce(hcat, psd_y_list)
    PSD_x_avg = vec(mean(psd_x_mat, dims=2))
    PSD_y_avg = vec(mean(psd_y_mat, dims=2))
    z_score = 1.96
    ci95_x = z_score .* vec(std(psd_x_mat, dims=2)) ./ sqrt(total_segments)
    ci95_y = z_score .* vec(std(psd_y_mat, dims=2)) ./ sqrt(total_segments)
    skipped_short > 0 && println("Skipped $skipped_short fixation segments shorter than welch_window_length.")
    println("Files discovered: $tot")
    println("Files contributing PSDs: $files_with_segments")
    println("Files without qualifying fixations: $(tot - files_with_segments)")
    println("Total fixation segments found: $total_fixation_segments")
    println("Qualifying fixation segments used: $total_segments")
    println("Welch window: $(welch_window_length) samples ($(round(welch_window_length / fs, digits=3)) s)")
    println("Welch frequency resolution: $(round(frequency_resolution, digits=3)) Hz")

    # Save the averaged PSDs to CSV
    avg_psd_df = DataFrame(Frequency=freqs, PSD_X=PSD_x_avg, PSD_Y=PSD_y_avg,
        CI95_X=ci95_x, CI95_Y=ci95_y)
    csv_outpath = joinpath(output_dir, "Averaged_PSD_Results.csv")
    CSV.write(csv_outpath, avg_psd_df)
    println("Saved averaged PSD results from $total_segments segments to $csv_outpath")

    max_index_x = argmax(PSD_x_avg)
    max_index_y = argmax(PSD_y_avg)
    println("Max PSD for X at frequency: $(freqs[max_index_x]) Hz")
    println("Max PSD for Y at frequency: $(freqs[max_index_y]) Hz")

    avg_max_frequency = (freqs[max_index_x] + freqs[max_index_y]) / 2
    println("Average of max frequencies: $avg_max_frequency Hz")

    # Plot the averaged PSDs
    plot_max_frequency = 250.0
    plotted_indices = (freqs .> 0) .& (freqs .<= plot_max_frequency)
    f_plot = freqs[plotted_indices]
    px_plot = PSD_x_avg[plotted_indices]
    py_plot = PSD_y_avg[plotted_indices]
    ci_x_plot = ci95_x[plotted_indices]
    ci_y_plot = ci95_y[plotted_indices]
    px_ribbon = (ci_x_plot, ci_x_plot)
    py_ribbon = (ci_y_plot, ci_y_plot)
    plotted_peak = maximum(vcat(px_plot, py_plot))
    plot_max_psd = 1.2 * plotted_peak
    y_exponent = floor(Int, log10(plot_max_psd))
    y_scale = 10.0^y_exponent
    y_tick_formatter = y -> @sprintf("%.2g", y / y_scale)

    p_avg_psd = Plots.plot(f_plot, px_plot, ribbon=px_ribbon, fillalpha=0.3,
        label="X-velocity", color="#0072B2", linewidth=4,
        size=(1000, 650), titlefontsize=24, guidefontsize=18, tickfontsize=20,
        legendfontsize=20, framestyle=:box, legend=:topright, margin=14Plots.mm,
        grid=true, minorticks=true, yformatter=y_tick_formatter)
    Plots.plot!(f_plot, py_plot, ribbon=py_ribbon, fillalpha=0.3,
        label="Y-velocity", color="#D55E00", linewidth=4)
    Plots.vline!([avg_max_frequency], color=:black, linestyle=:dash, linewidth=3, label="Peak Frequency: $(round(avg_max_frequency, digits=2)) Hz")

    Plots.xlims!(25, 250)
    Plots.xticks!(0:50:250)
    Plots.ylims!(1e4, plot_max_psd)
    Plots.xlabel!("Frequency (Hz)")
    Plots.ylabel!("Velocity PSD (×10$(superscript_integer(y_exponent)) (px/s)²/Hz)")
    Plots.title!("Power spectrum with 95% CI\n$(total_segments) segments from $(files_with_segments) files")
    Plots.savefig(Plots.joinpath(output_dir, "Averaged_PSD_Plot.pdf"))

    println("Saved averaged PSD plot to Averaged_PSD_Plot.pdf") 
end

freq_analyse("results/ISI_Modelling/Annotated_Data", joinpath("results/", "PSD_Results"))
