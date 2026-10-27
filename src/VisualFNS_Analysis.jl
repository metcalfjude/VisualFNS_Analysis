module VisualFNS_Analysis

using DrWatson
@quickactivate "VisualFNS_Analysis"

using Plots
using CairoMakie
using StatsBase

function weighted_corspearman(x, y, w_values)
    # Transform to ranks
    rx = tiedrank(x)
    ry = tiedrank(y)
   
    # Find Pearson's weighted correlation
    return cor(rx, ry, analyticweights(w_values))
end

include(srcdir("plotting_defaults.jl"))

end
