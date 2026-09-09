using Pkg
Pkg.activate(".")

using DSP
using CSV
using DataFrames
using Plots
using Statistics

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

function freq_analyse(input_dir::String, output_dir::String)
    isdir(output_dir) || mkdir(output_dir)

    csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))
    tot = length(csv_files)
    total_segments = 0

    welch_noverlap = 8
    fs = 1 / dt # Restored missing sampling frequency definition

    # Initialise Nothing PSD for x and y
    sum_PSD_x = Nothing
    sum_PSD_y = Nothing
    freqs_x = Nothing
    freqs_y = Nothing

    for (i, file) in enumerate(csv_files)
        println("Processing file $i of $tot: $file")
        data = CSV.read(file, DataFrame)

        # We only want to analyze data where is_saccade is false (i.e., fixations)
        is_fixation = .!data.is_saccade

        # Find the start and end indices of continuous fixation blocks
        diff_fix = diff(vcat(false, is_fixation, false))
        fix_starts = findall(==(1), diff_fix)
        fix_stops = findall(==(-1), diff_fix) .- 1

        fixed_window_length = 2048 
        welch_size_overlap = div(fixed_window_length, welch_noverlap)

        for (st, sp) in zip(fix_starts, fix_stops)
            segment_length = sp - st + 1
            
            # Only process segments long enough for the Welch window
            if segment_length >= fixed_window_length
                # Extract the raw position segment
                seg_x = data.value1[st:sp]
                seg_y = data.value2[st:sp]

                # Calculate velocity strictly within the fixation to avoid jump artifacts
                v_x = gradient(seg_x, dt)
                v_y = gradient(seg_y, dt)

                # Detrend
                v_x .-= mean(v_x)
                v_y .-= mean(v_y)

                # Calculate PSD for this segment
                pgram_x = welch_pgram(v_x, n=fixed_window_length, noverlap=welch_size_overlap, fs=fs)
                pgram_y = welch_pgram(v_y, n=fixed_window_length, noverlap=welch_size_overlap, fs=fs)

                # Accumulate
                if sum_PSD_x === Nothing
                    sum_PSD_x = pgram_x.power
                    freqs_x = pgram_x.freq
                    sum_PSD_y = pgram_y.power
                    freqs_y = pgram_y.freq
                else
                    sum_PSD_x .+= pgram_x.power
                    sum_PSD_y .+= pgram_y.power
                end
                
                total_segments += 1 
            end
        end
    end

    if total_segments == 0
        error("No valid inter-saccadic segments found of length >= 2048 across all files.")
    end

    PSD_x_avg = sum_PSD_x ./ total_segments
    PSD_y_avg = sum_PSD_y ./ total_segments

    # Save the averaged PSDs to CSV
    avg_psd_df = DataFrame(Frequency=freqs_x, PSD_X=PSD_x_avg, PSD_Y=PSD_y_avg)
    csv_outpath = joinpath(output_dir, "Averaged_PSD_Results.csv")
    CSV.write(csv_outpath, avg_psd_df)
    println("Saved averaged PSD results from $total_segments segments to $csv_outpath")

    max_index_x = argmax(PSD_x_avg)
    max_index_y = argmax(PSD_y_avg)
    println("Max PSD for X at frequency: $(freqs_x[max_index_x]) Hz")
    println("Max PSD for Y at frequency: $(freqs_y[max_index_y]) Hz")

    avg_max_frequency = (freqs_x[max_index_x] + freqs_y[max_index_y]) / 2
    println("Average of max frequencies: $avg_max_frequency Hz")

    # Plot the averaged PSDs
    p_avg_psd = Plots.plot(freqs_x[2:end], PSD_x_avg[2:end], label="PSD X", color=:darkblue, linewidth=2)
    Plots.plot!(freqs_y[2:end], PSD_y_avg[2:end], label="PSD Y", color=:darkorange, linewidth=2, xscale=:log10, yscale=:log10, legendfontsize=12, legend=:topleft)
    Plots.vline!([avg_max_frequency], color=:black, linestyle=:dash, linewidth=1.5, label=false) 
    Plots.annotate!([(avg_max_frequency*0.8, 4e4, text("Peak Freq: $(round(avg_max_frequency, digits=2)) Hz", "Computer Modern", :black, 16, :right))]) 
    Plots.xlims!(1, 1000.0)
    Plots.ylims!(1e4, 1e7)
    Plots.xlabel!("Frequency (Hz)")
    Plots.ylabel!("PSD (arb. units)")
    Plots.title!("Inter-Saccadic Velocity Welch Periodogram")
    Plots.savefig(Plots.joinpath(output_dir, "Averaged_PSD_Plot.pdf"))

    println("Saved averaged PSD plot to Averaged_PSD_Plot.pdf") 
end

freq_analyse("results/ISI_Parallel/Annotated_Data", joinpath("results/", "PSD_Results"))
