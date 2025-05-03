module Utils
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

end