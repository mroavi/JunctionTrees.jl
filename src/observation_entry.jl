"""
    inject_redus_in_msgs(g, before_pass_msgs, obsvars, obsvals)

Inject a reduction statement for each observed variable. Each reduction takes
the observed variable and it's corresponding observed value. The reduction
statements are introduced as late as possible, i.e. just before the observed
variable is marginalized.

"""
function inject_redus_in_msgs(g, before_pass_msgs, obsvars, obsvals)

  after_pass_msgs = quote end |> rmlines

  # Get the messages where one or more observed vars are marginalized
  mar_obs_msgs = filter_edges(g, :mar_obs_vars) |> # get a list of edges over which msgs that marginalize obs var pass
    mar_obs_edges -> map(x -> get_prop(g, x, :mar_obs_vars), mar_obs_edges) |> # get the property value
    mar_obs_msgs -> map(x -> Edge(x.src, x.dst), mar_obs_msgs) # create and store edges with msg direction info

  # # DEBUG: display the `:mar_obs_var` property for those edges that have it
  # filter_edges(g, :mar_obs_vars) |> mar_obs_edges -> map(x -> get_prop(g, x, :mar_obs_vars), mar_obs_edges) |> display

  for before_pass_msg in before_pass_msgs.args
    if @capture(before_pass_msg, var_ = f_(fargs__)) # parse the current msg (Note: this filters the line number nodes)
      src, dst = split(string(var), "_") |> x -> x[2:3] |> x -> parse.(Int, x) # get msg src and dst
      # Is the current msg in the set of msgs that marginalize one or more observed vars?
      # AND is the sepset of the edge through which this msg passes not empty?
      msg = Edge(src, dst) # create a directed edge that represents the current message
      if (msg in mar_obs_msgs) && (get_prop(g, msg, :sepset) |> !isempty)
        # Yes, then add a redu statement right before the marginalization
        mar_obs_vars = get_prop(g, msg, :mar_obs_vars) |> x -> x.vars # get the marginalized observed vars
        indx_mar_obs_vars = indexin(mar_obs_vars, obsvars) # find index of each elem in mar_obs_vars in the obsvars array
        mar_obs_vals = map(i -> :(obsvals[$i]), indx_mar_obs_vars) |> Tuple
        redu_expr = :(redu($(fargs[1]), $(mar_obs_vars), ($(mar_obs_vals...),))) # wrap the evaled prod in the redu expr
        after_pass_msg = :($var = $f($redu_expr, $(fargs[2:end]...))) # wrap the redu expr in the msg
      else
        # No, then do not add a redu statement
        after_pass_msg = before_pass_msg
      end
    else
      # The current msg is an evaled factor expr. Add it unmodified to the resulting expr arr
      after_pass_msg = before_pass_msg
    end
    push!(after_pass_msgs.args, after_pass_msg)
  end

  return after_pass_msgs

end

"""
    inject_redus_in_pots(g, before_pass_pots, obsvars, obsvals)

Inject a reduction statement for observed variables contained inside isolated
bags. An isolated bag is a leaf bag connected to the rest of the tree via one
empty sepset. Each reduction takes the observed variable and it's corresponding
observed value. 

"""
function inject_redus_in_pots(g, before_pass_pots, obsvars, obsvals)
  after_pass_pots = quote end |> rmlines
  for before_pass_pot in before_pass_pots.args
    if @capture(before_pass_pot, var_ = factor_)
      bag = split(string(var), "_") |> x -> x[2] |> x -> parse(Int, x) # get the bag id from the expr
      # Does the current bag contain an observed variable?
      # AND is the current bag isolated? (i.e. a leaf node connected to the rest of the tree via one empty sepset)
      bag_neighbors = neighbors(g, bag)
      if has_prop(g, bag, :obsvars) && length(bag_neighbors) == 1 && (get_prop(g, bag, bag_neighbors[1], :sepset) |> isempty)
        # Yes, then allow marginals to be extracted from this isolated bag
        set_prop!(g, bag, :isconsistent, true)
        # Add a redu statement to the current expression
        bag_obsvars = get_prop(g, bag, :obsvars) |> Tuple
        indx_bag_obsvars = indexin(bag_obsvars, obsvars) # find index of each elem in mar_obsvars in the obsvars array
        bag_obsvals = map(i -> :(obsvals[$i]), indx_bag_obsvars) |> Tuple
        redu_expr = :(redu($(factor), $(bag_obsvars), ($(bag_obsvals...),))) # wrap the evaled prod in the redu expr
        after_pass_pot = :($var = $redu_expr) # wrap the redu expr in the msg
      else
        # No, then do not add a redu statement
        after_pass_pot = before_pass_pot
      end
      push!(after_pass_pots.args, after_pass_pot)
    end
  end
  return after_pass_pots
end

"""
    inject_redus(g, before_pass_pots, obsvars, obsvals)

Inject a reduction expression to potentials that contain observed variables.

"""
function inject_redus(g, before_pass_pots, obsvars, obsvals)
  after_pass_pots = quote end |> rmlines
  for before_pass_pot in before_pass_pots.args
    if @capture(before_pass_pot, var_ = factor_)
      bag = split(string(var), "_") |> x -> x[2] |> x -> parse(Int, x) # get the bag id from the expr
      # Does the current bag contain an observed variable?
      if has_prop(g, bag, :obsvars)
        # Yes, then reduce the potential based on the observed vars and values
        bag_obsvars = get_prop(g, bag, :obsvars) |> Tuple
        indx_bag_obsvars = indexin(bag_obsvars, obsvars) # find index of each elem in mar_obsvars in the obsvars array
        bag_obsvals = map(i -> :(obsvals[$i]), indx_bag_obsvars) |> Tuple
        redu_expr = :(redu($(factor), $(bag_obsvars), ($(bag_obsvals...),))) # wrap the evaled prod in the redu expr
        after_pass_pot = :($var = $redu_expr) # wrap the redu expr in the msg
      else
        # No, then do not add a redu statement
        after_pass_pot = before_pass_pot
      end
      push!(after_pass_pots.args, after_pass_pot)
    end
  end
  return after_pass_pots
end

