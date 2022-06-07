using Test

for file in filter(f -> endswith(f, ".jl"), readdir(@__DIR__))

  if file in ["runtests.jl",]
    continue
  end

  @testset "$(file)" begin
    t = time()
    include(file)
    println("$(file) took $(round(time() - t; digits = 1)) seconds.")
  end

end
