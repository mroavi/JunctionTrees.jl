module TestDocs

using Test, Documenter, JunctionTrees

  DocMeta.setdocmeta!(JunctionTrees, :DocTestSetup, :(using JunctionTrees, Artifacts;); recursive=true)
  Documenter.doctest(JunctionTrees; manual=false)

end # module
