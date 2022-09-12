using Test

for file in filter(f -> endswith(f, ".jl"), readdir(@__DIR__))

  if file in ["runtests.jl",]
    continue
  end

  @testset "$(file)" begin
    include(file)
  end

end
