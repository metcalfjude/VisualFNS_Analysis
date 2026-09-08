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

    welch_n = 16
    welch_noverlap = 8

    # Initialise Nothing PSD for x and y
    sum_PSD_x = Nothing
    sum_PSD_y = Nothing

    freqs_x = Nothing
    freqs_y = Nothing

    for (i, file) in enumerate(csv_files)
          println("Processing file $i of $tot: $file")
          data = CSV.read(file, DataFrame)

          # Extract the relevant columns from the DataFrame
          time = data.timestamp
          x = data.value1
          y = data.value2

          # Standard finite difference for velocity (avoids unknown smoothing)
          v_x = gradient(x, dt)
          v_y = gradient(y, dt)

          # Detrend/remove mean
          v_x .-= mean(v_x)
          v_y .-= mean(v_y)
          fs = 1 / dt

          welch_window_length = div(length(v_x), welch_n)  # Use 1/32 of the data length for the Welch window
          welch_size_overlap = div(welch_window_length, welch_noverlap)  # 1/8 overlap

          if sum_PSD_x === Nothing && sum_PSD_y === Nothing
              pgram_x = welch_pgram(v_x, n=welch_window_length, noverlap=welch_size_overlap, fs=fs)
              pgram_y = welch_pgram(v_y, n=welch_window_length, noverlap=welch_size_overlap, fs=fs)

              window_length = div(length(v_x), 64) 
    
              spec_x = spectrogram(v_x, window_length, div(window_length, 8), fs=fs)
              power_x_db = 10 .* log10.(spec_x.power .+ eps())
              p_spec = Plots.heatmap(
                   spec_x.time, 
                   spec_x.freq, 
                   power_x_db,
                   xlabel="Time (s)", 
                   ylabel="Frequency (Hz)",
                   title="Spectrogram of Position X",
                   color=:viridis,        # viridis or :magma are excellent for spectrograms
                   cbar=true,             # Show the colorbar
                   ylims=(0, 200.0)        # CRITICAL: Zoom in on the low frequencies! 
              )
    
              # 4. Save the plot
              spec_pdfpath_x = replace(file, ".csv" => "_spectrogram_x.pdf")
              savefig(p_spec, spec_pdfpath_x)
              println("Saved spectrogram to: $spec_pdfpath_x")
    
              spec_y = spectrogram(v_y, window_length, div(window_length, 8), fs=fs)
              power_y_db = 10 .* log10.(spec_y.power .+ eps())
              p_spec_y = Plots.heatmap(
                   spec_y.time, 
                   spec_y.freq, 
                   power_y_db,
                   xlabel="Time (s)", 
                   ylabel="Frequency (Hz)",
                   title="Spectrogram of Position Y",
                   color=:viridis,        # viridis or :magma are excellent for spectrograms
                   cbar=true,             # Show the colorbar
                   ylims=(0, 200.0)        # CRITICAL: Zoom in on the low frequencies! 
              )
    
               # 4. Save the plot
               spec_pdfpath_y = replace(file, ".csv" => "_spectrogram_y.pdf")
               savefig(p_spec_y, spec_pdfpath_y)

               sum_PSD_x = pgram_x.power
               freqs_x = pgram_x.freq
     
               sum_PSD_y = pgram_y.power
               freqs_y = pgram_y.freq
     
               # Save the results
               pdfpath_x = replace(file, ".csv" => "_pxx_x.pdf")
               pdfpath_y = replace(file, ".csv" => "_pxx_y.pdf")
               println("Saving periodogram for x to: $pdfpath_x")
               println("Saving periodogram for y to: $pdfpath_y")
     
               # ADD yscale=:log10 to reveal hidden peaks
               Plots.plot(freqs_x, sum_PSD_x, xlabel="Frequency (Hz)", ylabel="Power Spectral Density", 
                    title="Example Welch Periodogram for x", legend=false, yscale=:log10, xlims=(0, 200.0), ylims=(1e5, 1e7))
               savefig(pdfpath_x)
     
               Plots.plot(freqs_y, sum_PSD_y, xlabel="Frequency (Hz)", ylabel="Power Spectral Density", 
                    title="Example Welch Periodogram for y", legend=false, yscale=:log10, xlims=(0, 200.0), ylims=(1e5, 1e7))
               savefig(pdfpath_y)

          elseif sum_PSD_x !== Nothing && sum_PSD_y !== Nothing
               sum_PSD_x += welch_pgram(v_x, n=welch_window_length, noverlap=welch_size_overlap, fs=fs).power
               sum_PSD_y += welch_pgram(v_y, n=welch_window_length, noverlap=welch_size_overlap, fs=fs).power
          else
              error("Unexpected state: One of sum_PSD_x or sum_PSD_y is Nothing while the other is not.")
          end
     end

     PSD_x_avg = sum_PSD_x ./ tot
     PSD_y_avg = sum_PSD_y ./ tot

     # Save the averaged PSDs to CSV
     avg_psd_df = DataFrame(Frequency=freqs_x, PSD_X=PSD_x_avg, PSD_Y=PSD_y_avg)
     csv_outpath = joinpath(output_dir, "Averaged_PSD_Results.csv")
     CSV.write(csv_outpath, avg_psd_df)
     println("Saved averaged PSD results to $csv_outpath")

     max_index_x = argmax(PSD_x_avg)
     max_index_y = argmax(PSD_y_avg)
     println("Max PSD for X at frequency: $(freqs_x[max_index_x]) Hz with value: $(PSD_x_avg[max_index_x])")
     println("Max PSD for Y at frequency: $(freqs_y[max_index_y]) Hz with value: $(PSD_y_avg[max_index_y])")

     avg_max_frequency = (freqs_x[max_index_x] + freqs_y[max_index_y]) / 2
     println("Average of max frequencies: $avg_max_frequency Hz")

     # Plot the averaged PSDs
     p_avg_psd = Plots.plot(freqs_x[2:end], PSD_x_avg[2:end], label="PSD X", color=:darkblue, linewidth=2)
     Plots.plot!(freqs_y[2:end], PSD_y_avg[2:end], label="PSD Y", color=:darkorange, linewidth=2, xscale=:log10, yscale=:log10, legendfontsize=12, legend=:topleft)
     Plots.vline!([avg_max_frequency], color=:black, linestyle=:dash, linewidth=1.5, label=false) #, label="Peak Freq: $(round(avg_max_frequency, digits=2)) Hz")
     Plots.annotate!([(avg_max_frequency*0.8, 4e4, text("Peak Freq: $(round(avg_max_frequency, digits=2)) Hz", "Computer Modern", :black, 16, :right))]) 
     Plots.xlims!(1, 1000.0)
     Plots.ylims!(1e4, 1e7)
     Plots.xlabel!("Frequency (Hz)")
     Plots.ylabel!("PSD (arb. units)")
     Plots.title!("Velocity Welch Periodogram")
     Plots.savefig(Plots.joinpath(output_dir, "Averaged_PSD_Plot.pdf"))

     println("Saved averaged PSD plot to Averaged_PSD_Plot.pdf") 
end

freq_analyse("data/", joinpath("results/", "PSD_Results"))
