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
    "Usage" => "usage.md",
    "Examples" => "examples.md",
    "UAI file formats" => "uai_file_formats.md",
    "PACE file formats" => "pace_file_formats.md",
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
