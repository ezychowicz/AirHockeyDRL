export Object, Puck, Mallet, EnvParams, AirHockeyEnv, State, Action, CollisionType, Collision, COLLISIONS_MAPPING, 
    PuckWallCollision_X, PuckWallCollision_Y, MalletWallCollision_X, MalletWallCollision_Y, PuckMalletCollision, Policy,
    RandomPolicy, ActorPolicy, NN, Experience, LearntPolicy

abstract type Object end
mutable struct Puck <: Object
    v::Vector{Real}
    pos::Vector{Real}
end

mutable struct Mallet <: Object
    v::Vector{Real}
    pos::Vector{Real}
end

Base.@kwdef struct EnvParams{T <: Real}
    x_len::T
    y_len::T
    goal_width::T
    puck_radius::T
    mallet_radius::T
    agent1_initial_pos::Vector{T}
    agent2_initial_pos::Vector{T}
    dt::T
    # max_dv::T
    band_e_loss::T
    restitution::T
    puck_mass::T
    mallet_mass::T
    max_dvx::T
    max_dvy::T
    max_vxy::T
end

Base.@kwdef mutable struct State # kwdef helps with constructor
    agent1::Union{Mallet, Nothing}
    agent2::Union{Mallet, Nothing}
    puck::Puck
end

Base.@kwdef struct Action
    dvx::Float32  # składowa x wektora zmiany prędkości
    dvy::Float32  # składowa y wektora zmiany prędkości
end

mutable struct AirHockeyEnv{T <: Real} 
    params::EnvParams{T}
    state::State
    done::Bool
    step_acc_reward1::Float32 # zakumulowana nagroda za przebieg jednego dt (np. za odbicia)
    step_acc_reward2::Float32
end

abstract type CollisionType end
struct PuckWallCollision_X <: CollisionType end
struct PuckWallCollision_Y <: CollisionType end
struct MalletWallCollision_X <: CollisionType end
struct MalletWallCollision_Y <: CollisionType end
struct PuckMalletCollision <: CollisionType end    
const COLLISIONS_MAPPING = Dict(
    1 => PuckWallCollision_X(),
    2 => PuckWallCollision_Y(),
    3 => PuckMalletCollision(),
    4 => PuckMalletCollision(),
    5 => MalletWallCollision_X(),
    6 => MalletWallCollision_X(),
    7 => MalletWallCollision_Y(),
    8 => MalletWallCollision_Y()
)
mutable struct Collision{T <: CollisionType, V <: Real} # teraz mogę robić multiple-dispatch ze względu na CollisionType
    params::EnvParams{V}
    mid_state::State
    type::T
    is_goal::Bool
    Collision(params::EnvParams{V}, mid_state::State, type::T)  where {T <: CollisionType, V <: Real} =
        new{T, V}(params, mid_state, type, false)
end

abstract type Policy end
struct RandomPolicy{V1,V2 <: Distribution} <: Policy
    distribution_vx::V1
    distribution_vy::V2
end
  
mutable struct NN
    model::Chain
    optimizer::Adam
    target::Chain
    opt_state::NamedTuple
end


struct ActorPolicy{V <: Distribution} <: Policy
    actor::NN
    noise::V # UWAGA: noise to rozkład odnoszący się do KARTEZJAŃSKICH współrzędnych wektora dv. I to zdenormalizowanych
end


struct LearntPolicy <: Policy
    model::Chain
end

struct Experience{T,V <: Real}
    s::Vector{T}
    a::Vector{V}
    r::Float32
    s_next::Vector{T}
    d::Int64
end

