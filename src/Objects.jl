module Objects
abstract type Object end

mutable struct Puck <: Object
    v = Vector{Float32}()
    pos = Vector{Float32}()
end

mutable struct Mallet <: Object
    v = Vector{Float32}()
    pos = Vector{Float32}()
end

end

