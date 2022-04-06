using Documenter, JunctionTrees

makedocs(
  modules = [JunctionTrees],
  clean = true,
  format = Documenter.HTML(
    assets = [joinpath("assets", "favicon.ico")],
    prettyurls = get(ENV, "CI", nothing) == "true"
  ),
  sitename = "JunctionTrees.jl",
  authors = "Martin Roa Villescas and Patrick Wijnings",
  pages = [
    "Home" => "index.md",
    "Background" => "background.md",
    "Usage" => "usage.md",
    "Vizualization" => Any[
      "Junction trees" => "visualization/junction_trees.md",
      "Markov random fields" => "visualization/markov_random_fields.md"
    ],
    "File formats" => Any[
      "UAI" => "file_formats/uai.md",
      "PACE" => "file_formats/pace.md"
    ],
    "Library" => Any[
      "Public" => "library/public.md",
      "Internals" => "library/internals.md"
    ],
  ],
)

if get(ENV, "CI", nothing) == "true"
  deploydocs(
    repo = "github.com/mroavi/JunctionTrees.jl.git"
  )
end
