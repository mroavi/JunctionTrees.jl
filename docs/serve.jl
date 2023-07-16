function serve(; host::String="0.0.0.0", port::Int=8000)
  # setup environment
  docs_dir = @__DIR__
  julia_cmd = "using Pkg; Pkg.instantiate()"
  run(`$(Base.julia_exename()) --project=$docs_dir -e $julia_cmd`)

  serve_cmd = """
  using LiveServer;
  LiveServer.servedocs(;
      doc_env=false,
      skip_dirs=[
          joinpath("docs", "src", "generated"),
          joinpath("docs", "build"),
      ],
      skip_files=[
          joinpath("docs", "Manifest.toml"),
      ],
      #literate="examples",
      host=\"$host\",
      port=$port,
  )
  """
  try
    run(`$(Base.julia_exename()) --project=$docs_dir -e $serve_cmd`)
  catch e
    if e isa InterruptException
      return
    else
      rethrow(e)
    end
  end
  return
end

serve()
