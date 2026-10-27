
# Set global publication defaults for Plots.jl
default(
    # Typography
    fontfamily = "Computer Modern",
    titlefontsize = 11,
    guidefontsize = 10,    # Axis label size
    tickfontsize = 9,     # Tick label size
    legendfontsize = 9,

    # Dimensions & Layout (e.g., standard single-column width)
    size = (450, 320),
    margin = 5Plots.mm,

    # Styling & Aesthetics
    framestyle = :box,    # Encloses the plot in a clean border
    grid = false,         # Turn off distracting background grids (or use true/alpha if needed)
    linewidth = 1.5,      # Thicker lines look sharper in print
    markersize = 4,
    palette = :tab10      # Professional, colorblind-friendly categorical palette
)

# Build a comprehensive publication theme
pub_theme = merge(
    theme_latexfonts(), # Sets Computer Modern font globally
    Theme(
        fontsize = 10,  # Base font size matching 10pt LaTeX documents

        # Axis-specific tweaks
        Axis = (
            xlabelsize = 11,
            ylabelsize = 11,
            xticklabelsize = 10,
            yticklabelsize = 10,
            linewidth = 1.2,            # Axis spine and tick line width
            xminorticksvisible = true,  # Professional minor ticks
            yminorticksvisible = true,
            xminorticks = IntervalsBetween(5),
            yminorticks = IntervalsBetween(5),
            xgridvisible = false,       # Turn off grids by default
            ygridvisible = false,
        ),

        # Plot element defaults
        Lines = (linewidth = 1.5,),
        Scatter = (markersize = 6,),
        Legend = (
            framevisible = true,
            framewidth = 1.0,
            backgroundcolor = :white,
        )
    )
)

# Activate the theme globally
set_theme!(pub_theme)