using Base.Math: atan
function solve_quadratic(a, b, c)
    if isapprox(a, 0; atol=0, rtol=sqrt(eps(Float64)))
        # Równanie liniowe: bx + c = 0
        return isapprox(b, 0) ? [] : [-c / b]
    end

    D = b^2 - 4a*c
    if D < 0
        return []
    elseif isapprox(D, 0)
        return [-b / (2a)]
    else
        sqrtD = sqrt(D)
        # wersja z unikaną utraty precyzji:
        q = -0.5 * (b + sign(b) * sqrtD)
        x1 = q / a
        x2 = c / q
        return sort([x1, x2])
    end
end

convert_from_polar_to_cartesian(r, θ) = Vector{Float32}([r * cos(θ), r * sin(θ)])
convert_from_cartesian_to_polar(x, y) = Vector{Float32}([atan(y, x), sqrt(x^2 + y^2)])
function normalize_puck_position(params::EnvParams, pos::Vector{T}) where {T <: Real}
    """
    Map position from [r_puck, x_len - r_puck] x [r_puck, y_len - r_puck] to [-1,1]² 
    """
    x_len, y_len = params.x_len, params.y_len
    r = params.puck_radius

    x_scaled = 2 * (pos[1] - r) / (x_len - 2r) - 1
    y_scaled = 2 * (pos[2] - r) / (y_len - 2r) - 1

    return clamp.(Float32[x_scaled, y_scaled], -1, 1)
end

function normalize_mallet_position(params::EnvParams, pos::Vector{T}) where {T <: Real}
    """
    Map mallet position from:
    [r_mallet, x_len/2 - r_mallet] x [r_mallet, y_len - r_mallet] or
    [x_len/2 + r_mallet, x_len - r_mallet] x [r_mallet, y_len - r_mallet] 
    to [-1,1]²
    """
    x_len, y_len = params.x_len, params.y_len
    r = params.mallet_radius

    if pos[1] < x_len/2
        # Lewa połowa: [r, x_len/2 - r]
        x_scaled = 2 * (pos[1] - r) / (x_len/2 - 2r) - 1
    else
        # Prawa połowa: [x_len/2 + r, x_len - r]
        x_scaled = 2 * (pos[1] - (x_len/2 + r)) / (x_len/2 - 2r) - 1
    end

    y_scaled = 2 * (pos[2] - r) / (y_len - 2r) - 1

    return clamp.(Float32[x_scaled, y_scaled], -1, 1)
end

function normalize_action(params::EnvParams, action::Action)
    """
    Map action from [-max_dvx, max_dvx] x [-max_dvy, max_dvy] to [-1,1]²
    """
    max_dvx, max_dvy = params.max_dvx, params.max_dvy
    return clamp.(Float32[action.dvx/max_dvx, action.dvy/max_dvy], -1, 1)
end

function denormalize_action(env::AirHockeyEnv, output::Vector{V}) where {V <: Real}
    """
    Map NN output from [-1,1]² to Vector [-max_dvx, max_dvx] x [-max_dvy, max_dvy]
    """
    max_dvx, max_dvy = env.params.max_dvx, env.params.max_dvy
    return Float32[output[1]*max_dvx, output[2]*max_dvy]
end

function normalize_velocity(params::EnvParams, vec::Vector{T}) where {T <: Real}
    """
    Map velocity from [-max_vxy, max_vxy]² to [-1,1]². 
    Used to map object velocity to state NN input structure.
    """
    max_vxy = params.max_vxy
    return clamp.(Float32[vec[1]/max_vxy, vec[2]/max_vxy], -1, 1)
end
