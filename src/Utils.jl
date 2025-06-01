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

function normalize_position(params::EnvParams, pos::Vector{T}) where {T <: Real}
    """
    Map position from [0, x_len] x [0, y_len] to [-1,1]²
    """
    x_len, y_len = params.x_len, params.y_len
    return Vector{Float32}([clamp(2*pos[1]/x_len - 1, -1, 1), clamp(2*pos[2]/y_len - 1, -1, 1)])
end

function normalize_action(params::EnvParams, action::Action)
    """
    Map action from [-π, π] x [0, max_dv] to [-1,1]²
    """
    max_dv = params.max_dv
    return Vector{Float32}([clamp(action.dv_angle/π, -1, 1), clamp(-1 + 2*action.dv_len/max_dv, -1, 1)])
end

