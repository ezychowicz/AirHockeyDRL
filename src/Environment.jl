module Environment


struct EnvParams{T <: Real}
    x_len::T
    y_len::T
    goal_width::T
    puck_radius::T
    agent_radius::T
    agent1_initial_pos::T
    agent2_initial_pos::T
    dt::T
    terminal_velo::T
    band_e_loss::T
end

Base.@kwdef struct StateVector # kwdef helps with constructor
    agent1_position::Vector{Float32}
    agent2_position::Vector{Float32}   
    agent1_velo::Vector{Float32}
    agent2_velo::Vector{Float32}  
    puck_position::Vector{Float32}     
    puck_velocity::Vector{Float32}     
end

mutable struct AirHockeyEnv{T <: Real} <: AbstractEnv
    params::EnvParams{T}
    state::StateVector
    done::Bool
end


function opponent_goal(env::AirHockeyEnv)::Bool 
    puck_pos = env.state.puck.position
    puck_pos[1] > env.params.x_len + 2puck.radius && (env.y_len/2 - env.goal_width/2 <= puck_pos[2] <= env.y_len/2 + env.goal_width/2)  
end

function reward(env::AirHockeyEnv)
    if done
        if opponent_goal(env::AirHockeyEnv)
            return -1.0
        else
            return 1.0
    end
    return 0.0
end

function reset!(env::AirHockeyEnv)
    env.state = StateVector(
        agent1_position = env.agent1_initial_pos,
        agent2_position = env.agent2_initial_pos,
        agent1_velo = zeros(Float32, 2),
        agent2_velo = zeros(Float32, 2),
        puck_position = [env.x_len/2, env.y_len/2],
        puck_velocity = zeros(Float32, 2)
    )
    env.done = false
    return env.state
end

convert_from_polar_to_cartesian(r, θ) = Vector{Float32}([r * cos(θ), r * sin(θ)])

function border_collision(env::AirHockeyEnv, puck::Puck)
    t_remain = env.params.dt
    r = env.params.puck_radius
    x_len = env.params.x_len
    y_len = env.params.y_len

    while t_remain > 0 # bo może być w teorii kilka odbić od ściany w trakcie jednego dt


        t_min = min(tx, ty)

        if t_min > t_remain # brak kolizji
            puck.pos[1] += puck.v[1] * t_remain
            puck.pos[2] += puck.v[2] * t_remain
            break
        else
            puck.pos[1] += puck.v[1] * t_min # przewiń do kolizji
            puck.pos[2] += puck.v[2] * t_min

            if abs(tx - ty) < 1e-6  # kolizja w rogu (jak DVD logo bouncing game)
                puck.v[1] *= -1
                puck.v[2] *= -1
            elseif tx < ty # kolizja ze ścianą zabramkową
                puck.v[1] *= -1
            else # kolizja ze ścianą boczną
                puck.v[2] *= -1
            end

            t_remain -= t_min
        end
    end
end

function time_to_wall(env::AirHockeyEnv, puck::Puck) 
    """
    Oblicza czas do kolizji krążka ze ścianami bocznymi i zabramkowymi przy aktualnej prędkości i pozycji.
    """
    r = env.params.puck_radius
    x_len = env.params.x_len
    y_len = env.params.y_len
    tx = ty = Inf

    if puck.v[1] > 0
        tx = (x_len - r - puck.pos[1]) / puck.v[1]
    elseif puck.v[1] < 0
        tx = (r - puck.pos[1]) / puck.v[1]
    end

    if puck.v[2] > 0
        ty = (y_len - r - puck.pos[2]) / puck.v[2]
    elseif puck.v[2] < 0
        ty = (r - puck.pos[2]) / puck.v[2]
    end
    return tx, ty
end

function time_to_wall(env::AirHockeyEnv, mallet::Mallet)
    """
    Oblicza czas do kolizji malleta ze ścianami bocznymi i zabramkowymi przy aktualnej prędkości i pozycji.
    DODATKOWO: uwzględnia kolizje ze ścianą środka boiska.
    """
function time_to_mallet(env::AirHockeyEnv, puck::Puck, mallet::Mallet)
    """
    Oblicz czas do kolizji krążka z malletem.
    """
end

function update_positions(puck::Puck, mallet1::Mallet, mallet2::Mallet, dt::Float32)
    """
    Zaktualizuj pozycje wszystkich obiektów w czasie dt. 
    Funkcja ta przyjmuje, że w tym czasie NIE MA kolizji.
    """

end


function simulate_dt(env::AirHockeyEnv, puck::Puck, mallet1::Mallet, mallet2::Mallet)
    t_remain = env.params.dt

    while t_remain > 0
        tx, ty = time_to_wall(env, puck)
        tm1 = time_to_mallet(env, puck, mallet1)
        tm2 = time_to_mallet(env, puck, mallet2)
        tm1_to_wall = time_to_wall(env, mallet1)
        tm2_to_wall = time_to_wall(env, mallet2)
        # Znajdź najbliższe zdarzenie
        times = [tx, ty, tm1, tm2, t_remain]
        t_next, idx = findmin(times)

        # Przesuń wszystko do momentu kolizji
        update_positions(puck, mallet1, mallet2, t_next)

        # Obsłuż kolizję
        if 
            resolve_puck_mallet_collision!(puck, mallet)
        elseif isapprox(t_next, tx) || t_next ≈ ty
            resolve_wall_collision!(puck, tx < ty)
        end

        t_remain -= t_next
    end
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

    # predkości zaraz po wykonaniu akcji (dodaniu dv do poprzednich wektorów predkości)
    a1_start_velo = env.state.agent1_velo + convert_from_polar_to_cartesian(action1.dv_len, action1.dv_angle)
    a2_start_velo = env.state.agent2_velo + convert_from_polar_to_cartesian(action2.dv_len, action2.dv_angle)

    #TODO rozwazyx  ruchy krazka i agentow i czy dochodzi do zderzenia (za pomocą (x(t), y(t)))
    env.state = StateVector(
        agent1_position = ,
        agent2_position = ,
        agent1_velo = ,
        agent2_velo = ,
        puck_position = ,
        puck_velocity = 
    )
end

is_terminated(env::AirHockeyEnv) = env.done

end
