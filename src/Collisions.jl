module Collisions

abstract type CollisionType end
struct PuckWallCollision_X <: CollisionType end
struct PuckWallCollision_Y <: CollisionType end
struct MalletWallCollision_X <: CollisionType end
struct MalletWallCollision_Y <: CollisionType end
struct PuckMalletCollision <: CollisionType end    
const COLLISIONS_MAPPING = Dict(
    1 => PuckWallCollision_X,
    2 => PuckWallCollision_Y,
    3 => PuckMalletCollision,
    4 => PuckMalletCollision,
    5 => MalletWallCollision_X,
    6 => MalletWallCollision_Y,
    7 => MallerWallCollision_X,
    8 => MallerWallCollision_Y
)
struct Collision{T <: CollisionType} # teraz mogę robić multiple-dispatch ze względu na CollisionType
    params::EnvParams
    mid_state::StateVector
    type::T
end

map_int_to_type(type::Int)::CollisionType = COLLISIONS_MAPPING[type]


function handle_collision(collision::Collision{PuckWallCollision_X})
    
end

function handle_collision(collision::Collision{PuckWallCollision_X})

end

function handle_collision(collision::Collision{PuckWallCollision_X})

end

function handle_collision(collision::Collision{PuckWallCollision_X})

end

end