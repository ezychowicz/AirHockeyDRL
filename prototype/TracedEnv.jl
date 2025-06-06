# stworzone aby nie ingerować w kod RLowy ale żeby zapisywać stany potrzebne do animacji

module TracedEnv
using AirHockey # Puck itd. są eksportowane wiec nie musze przedrostków dodawać 
using StaticArrays
mutable struct Trace{T<:Real}
    puck_trace::Vector{Puck}
    mallet1_trace::Vector{Mallet}
    mallet2_trace::Vector{Mallet}
    times::Vector{T} # różnice czasów między kolejnymi stanami Tracea
    result::Vector{Union{Nothing, Bool}} #True=lewy agent strzelił, False=prawy. nothing - dt upłynęło bez gola
    rewards::Vector{Vector{Float32}} # nagrody przyznawane po każdym kroku obu agentom. Vector z dwuelementowymi wektorami
end

function update_positions!(
    puck::Puck,
    mallet1::Union{Mallet, Nothing},
    mallet2::Union{Mallet, Nothing},
    dt::T
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
    trace = Trace{Float32}(Vector{Puck}(), Vector{Mallet}(),Vector{Mallet}(),Vector{Float32}(), Vector{Union{Nothing, Bool}}(), Vector{Vector{Float64}}())
    
    while t_remain > 0 && !env.done
        tx, ty = AirHockey.time_to_wall(env, puck)
        tm1 = AirHockey.time_to_mallet(env, puck, mallet1) 
        tm2 = AirHockey.time_to_mallet(env, puck, mallet2)
        tm1_to_wallx, tm1_to_wally = AirHockey.time_to_wall(env, mallet1)
        tm2_to_wallx, tm2_to_wally = AirHockey.time_to_wall(env, mallet2)
        # Znajdź najbliższe zdarzenie - UWAGA: tak naprawdę tylko ten pierwszy czas jest zawsze prawdziwy.
        # Obliczone czasy nie uwzględniają  bowiem zderzenia, które się wydarzy w minimum z tych czasów.
        times = [tx, ty, tm1, tm2, tm1_to_wallx, tm2_to_wallx, tm1_to_wally, tm2_to_wally]
        
        t_next, idx = findmin(times)
        # print("min_time=$(t_next) idx=$(idx) t_remain = $(t_remain)")
        

        if t_next != Inf # doszło do kolizji
            update_positions!(puck, mallet1, mallet2, t_next) # przewiń symulacje do momentu kolizji
            AirHockey.execute_collision!(env, idx)
        else
            t_next = t_remain
            update_positions!(puck, mallet1, mallet2, t_next) #przewiń do końca dt
        end
        push!(trace.puck_trace, deepcopy(puck))
        push!(trace.mallet1_trace, mallet1 !== nothing ? deepcopy(mallet1) : Mallet(Vector{V}(),Vector{V}()))
        push!(trace.mallet2_trace, mallet2 !== nothing ? deepcopy(mallet2) : Mallet(Vector{V}(),Vector{V}()))
        push!(trace.result, nothing)
        push!(trace.times, t_next)
        push!(trace.rewards, Float32[0,0])
        t_remain -= t_next # zmniejsz pozostały czas o czas już zhandlowany
    end
    return trace
end

function reset!(env::AirHockeyEnv, trace::Trace)
    # hm? czy to zawsze dziala? chyba tak bo nie ma prawa dojsc do innej kolizji miedzy kolizja-golem a reset!
    if env.state.puck.pos[1] < env.params.x_len/2 # zapisz kto strzelił
        trace.result[end] = true 
    else
        trace.result[end] = false 
    end
    AirHockey.reset!(env)
end

function step!(env::AirHockeyEnv, action1::Action, action2::Action)
    """
    step - przejście w czasie o dt.
    1. Agenci wykonuja action1 i action2 równolegle, nie znając swoich ruchów nawzajem
    2. Zakładam brak odrzutu rączki po odbiciu od krążka (gracz powstrzymuje cofanie).
    w obliczeniach zakładam jednak pewną masę rączki, która w rzeczywistości umożliwiłaby
    lekki odrzut. wyobrażam to sobie tak, że gracz reaguje na odrzut już po skończeniu zderzenia i powstrzymuje
    cofanie się rączki. 
    3. Zakładam brak tarcia, energia kinetyczna jest natomiast wytracana przy zderzeniach.
    4. Zmiany prędkości przez akcję agentów (dv) są błyskawiczne (agent steruje prędkością, nie 
    siłą (F=ma, dopiero a prowadzi do v), co jest lekkim uproszczeniem). 
    5. Czyli wszelkie ruchy (w okresach dt) są tak naprawdę jednostajne.

    Schemat:
    Stan 0 -> Agenci wybierają akcję ->
    step!: mija dt i środowisko zmienia się zgodnie z wybranymi akcjami ->
    Stan 1 -> ...
    """

    # predkości zaraz po wykonaniu akcji (dodaniu dv do poprzednich wektorów predkości). 
    # Zmieniam stan enva na after_state - stan po wykonaniu akcji, ale przed stanem następnym
    env.state.agent1.v .+= AirHockey.convert_from_polar_to_cartesian(action1.dv_len, action1.dv_angle)
    env.state.agent2.v .+= AirHockey.convert_from_polar_to_cartesian(action2.dv_len, action2.dv_angle)
    trace = simulate_dt!(env)
    r1, r2 = AirHockey.reward(env)
    trace.rewards[end] .= [r1, r2] # zamień ostatnią nagrodę z zer na rzeczywistą (ten same sposób co z result w reset!)
    if AirHockey.is_terminated(env); reset!(env, trace) end
    
    return trace
end

end