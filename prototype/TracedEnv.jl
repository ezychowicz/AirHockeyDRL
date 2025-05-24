# stworzone aby nie ingerować w kod RLowy ale żeby zapisywać stany potrzebne do animacji

module TracedEnv
using AirHockey # Puck itd. są eksportowane wiec nie musze przedrostków dodawać 
mutable struct Trace{T<:Real}
    puck_trace::Vector{Puck}
    mallet1_trace::Vector{Mallet}
    mallet2_trace::Vector{Mallet}
    times::Vector{T} # różnice czasów między kolejnymi stanami Tracea
end
function update_positions!(
    puck::Puck,
    mallet1::Union{Mallet, Nothing},
    mallet2::Union{Mallet, Nothing},
    dt::T,
) where {T<:Real}
    
    puck.pos += puck.v * dt
    if mallet1 !== nothing
        mallet1.pos += mallet1.v * dt
    end
    if mallet2 !== nothing
        mallet2.pos += mallet2.v * dt
    end


end

function simulate_dt!(env::AirHockeyEnv)
    mallet1, mallet2, puck = env.state.agent1, env.state.agent2, env.state.puck
    t_remain = env.params.dt
    trace = Trace{Float32}(Vector{Puck}(), Vector{Mallet}(),Vector{Mallet}(),Vector{Float32}())
    while t_remain > 0
        tx, ty = AirHockey.time_to_wall(env, puck)
        tm1 = AirHockey.time_to_mallet(env, puck, mallet1) 
        tm2 = AirHockey.time_to_mallet(env, puck, mallet2)
        tm1_to_wallx, tm1_to_wally = AirHockey.time_to_wall(env, mallet1)
        tm2_to_wallx, tm2_to_wally = AirHockey.time_to_wall(env, mallet2)
        # Znajdź najbliższe zdarzenie - UWAGA: tak naprawdę tylko ten pierwszy czas jest zawsze prawdziwy.
        # Obliczone czasy nie uwzględniają  bowiem zderzenia, które się wydarzy w minimum z tych czasów.
        times = [tx, ty, tm1, tm2, tm1_to_wallx, tm2_to_wallx, tm1_to_wally, tm2_to_wally]
        t_next, idx = findmin(times)

        update_positions!(puck, mallet1, mallet2, t_next) # przewiń symulacje do momentu kolizji

        if t_next != Inf # doszło do kolizji
            AirHockey.execute_collision!(env, idx)
        end
        push!(trace.puck_trace, deepcopy(puck))
        push!(trace.mallet1_trace, mallet1 !== nothing ? deepcopy(mallet1) : Mallet(Vector{V}(),Vector{V}()))
        push!(trace.mallet2_trace, mallet2 !== nothing ? deepcopy(mallet2) : Mallet(Vector{V}(),Vector{V}()))
        push!(trace.times, t_next)
        t_remain -= t_next # zmniejsz pozostały czas o czas już zhandlowany
    end
    return trace
end

end