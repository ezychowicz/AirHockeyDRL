module AirHockey
println(pwd())
using LinearAlgebra
using Distributions
using Flux
# Wczytaj wewnętrzne pliki
include("CoreTypes.jl")
include("Utils.jl")
include("Collisions.jl")
include("Environment.jl")
include("Actions.jl")
include("Actor.jl")
include("Critic.jl")

end
       