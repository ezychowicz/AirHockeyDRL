

function reset!(env::AirHockeyEnv)
    env.state = State(
        agent1 = Mallet(zeros(Float32, 2), env.params.agent1_initial_pos),
        agent2 = Mallet(zeros(Float32, 2), env.params.agent2_initial_pos),
        puck = Puck(zeros(Float32, 2), [env.params.x_len/2, env.params.y_len/2])
    )
    env.done = false
    return env
end

# function reward(env::AirHockeyEnv)
#     """
#     Zwraca nagrodę obu agentom. 
#     Agent 1 dostaje 1.0, gdy padł gol i krążek jest na połowie przeciwnika.
#     Dodatkowe shaping rewards za zbliżenie się do krążka.
#     """
#     if env.done && env.state.puck.pos[1] > env.params.x_len / 2
#         return 1.0f0, -1.0f0
#     elseif env.done
#         return -1.0f0, 1.0f0
#     end

#     x_len = env.params.x_len
#     puck_r = env.params.puck_radius
#     agent_r = env.params.mallet_radius
#     puck_pos = env.state.puck.pos
#     mallet1_pos = env.state.agent1.pos
#     mallet2_pos = env.state.agent2.pos

#     function distance_reward(pos1, pos2)
#         dist = max(norm(pos1 .- pos2), 0.001f0)
#         base = dist < 0.35 * x_len ? Float32(0.05 * (puck_r + agent_r) / dist) : -0.0002f0
#         bonus = dist < 0.04 * x_len ? 0.1f0 * (1.0f0 - dist / (0.04f0 * x_len)) : 0.0f0
#         return Float32(base + bonus)
#     end

#     if puck_pos[1] < x_len / 2 - 0.01 
#         return distance_reward(puck_pos, mallet1_pos), 0.0002f0
#     elseif puck_pos[1] > x_len / 2 + 0.01
#         return 0.0002f0, distance_reward(puck_pos, mallet2_pos)
#     else
#         return (
#             distance_reward(puck_pos, mallet1_pos), 
#             distance_reward(puck_pos, mallet2_pos)
#         )
#     end
# end

function reward(env::AirHockeyEnv)
    """
    Zwraca nagrodę obu agentom. 
    Agent 1 dostaje 1.0, gdy padł gol i krążek jest na połowie przeciwnika.
    Dodatkowe shaping rewards za zbliżenie się do krążka oraz karę za dotykanie ściany.
    """
    if env.done && env.state.puck.pos[1] > env.params.x_len / 2
        return 1.0f0, -1.0f0
    elseif env.done
        return -1.0f0, 1.0f0
    end

    x_len = env.params.x_len
    y_len = env.params.y_len
    puck_r = env.params.puck_radius
    agent_r = env.params.mallet_radius
    puck_pos = env.state.puck.pos
    mallet1_pos = env.state.agent1.pos
    mallet2_pos = env.state.agent2.pos

    function distance_reward(pos1, pos2)
        dist = max(norm(pos1 .- pos2), 0.001f0)
        base = dist < 0.35 * x_len ? Float32(0.05 * (puck_r + agent_r) / dist) : -0.0002f0
        bonus = dist < 0.04 * x_len ? 0.1f0 * (1.0f0 - dist / (0.04f0 * x_len)) : 0.0f0
        return Float32(base + bonus)
    end

    r1 = if puck_pos[1] < x_len / 2 - 0.01
        distance_reward(puck_pos, mallet1_pos)
    elseif puck_pos[1] > x_len / 2 + 0.01
        0.0002f0
    else
        distance_reward(puck_pos, mallet1_pos)
    end

    r2 = if puck_pos[1] > x_len / 2 + 0.01
        distance_reward(puck_pos, mallet2_pos)
    elseif puck_pos[1] < x_len / 2 - 0.01
        0.0002f0
    else
        distance_reward(puck_pos, mallet2_pos)
    end

    # Dodanie kar za kolizje ze ścianą i nagordy za kolizje z krążkiem
    r1 += clamp(env.step_acc_reward1, -0.2f0,0.4f0)
    r2 += clamp(env.step_acc_reward2, -0.2f0,0.4f0)

    return r1, r2
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
    return (tx <= env.params.dt ? tx : Inf), (ty <= env.params.dt ? ty : Inf)
end

function time_to_wall(env::AirHockeyEnv, mallet::Mallet)
    """
    Oblicza czas do kolizji malleta ze ścianami bocznymi i zabramkowymi przy aktualnej prędkości i pozycji.
    DODATKOWO: uwzględnia kolizje ze ścianą środka boiska.
    """
    r = env.params.mallet_radius
    x_len = env.params.x_len
    y_len = env.params.y_len

    half = x_len / 2
    is_left = mallet.pos[1] <= half # czy mallet jest po lewej 

    if is_left
        x_min = r
        x_max = half - r
    else
        x_min = half + r
        x_max = x_len - r
    end

    tx = Inf # czas do kolizji w osi OX (środek i banda zabramkowa)
    if mallet.v[1] > 0
        tx = (x_max - mallet.pos[1]) / mallet.v[1]
    elseif mallet.v[1] < 0
        tx = (x_min - mallet.pos[1]) / mallet.v[1]
    end

    ty = Inf # czas do kolizji w osi OY (boczne bandy)
    if mallet.v[2] > 0
        ty = (y_len - r - mallet.pos[2]) / mallet.v[2]
    elseif mallet.v[2] < 0
        ty = (r - mallet.pos[2]) / mallet.v[2]
    end

    return (tx <= env.params.dt ? tx : Inf), (ty <= env.params.dt ? ty : Inf)
end



function time_to_mallet(env::AirHockeyEnv, puck::Puck, mallet::Mallet)
    """
    Oblicz czas do kolizji krążka z malletem.
    Kolizja zachodzi gdy d(S1,S2) <= r1 + r2 (S1, S2 - środki obiektów).
    Szukamy t takiego, że (x1(t) - x2(t))² + (y1(t) - y2(t))² = (r1 + r2)²
    """

    dpos = puck.pos .- mallet.pos
    dv = puck.v .- mallet.v

    # a*t² + b*t + c = 0
    a = dot(dv, dv)
    b = 2dot(dpos, dv)
    c = dot(dpos, dpos) - (env.params.puck_radius + env.params.mallet_radius)^2
    # println("$(a)t² + $(b)t + $c = 0")
    roots = solve_quadratic(a, b, c) # zwraca posortowane pierwiastki, jeśli istnieją
    # println(roots)
    for t in roots
        if 1e-6 < t < env.params.dt 
            return t
        end
    end

    return Inf
end


function update_positions!(puck::Puck, mallet1::Mallet, mallet2::Mallet, dt::V) where {V<:Real}
    """
    Zaktualizuj pozycje wszystkich obiektów w czasie dt. 
    Funkcja ta przyjmuje, że w tym czasie NIE MA kolizji.
    Służy ona do aktualizowania położeń między zdarzeniami.
    """
    puck.pos += puck.v * dt
    mallet1.pos += mallet1.v * dt
    mallet2.pos += mallet2.v * dt
end

function execute_collision!(env::AirHockeyEnv, idx::Int)
    mallet1, mallet2, puck = env.state.agent1, env.state.agent2, env.state.puck
    if idx >= 3  
        mallet1 = idx % 2 == 1 ? mallet1 : nothing # posłuży jako informacja który mallet sie zderza
        mallet2 = idx % 2 == 0 ? mallet2 : nothing
        env.step_acc_reward1 = idx == 3 ? env.step_acc_reward1 + 0.2f0 : env.step_acc_reward1 # uderzenie w krazek
        env.step_acc_reward2 = idx == 4 ? env.step_acc_reward2 + 0.2f0 : env.step_acc_reward2
        if idx > 4
            # println("KURWA SCIANA")
            env.step_acc_reward1 = idx % 2 == 1 ? env.step_acc_reward1 - 0.1f0 : env.step_acc_reward1 # uderzenie sobą o sciane
            env.step_acc_reward2 = idx % 2 == 0 ? env.step_acc_reward2 - 0.1f0 : env.step_acc_reward2
        end
    end
    
    mid_state = State( # przekazujemy mid_state - stan w momencie kolizji (zmienił się na pewno względem stanu wejściowego i afterstatea)
        agent1 = mallet1, #przekaz wskazania/nothingi
        agent2 = mallet2,
        puck = puck
    )

    # println(typeof(Collision(env.params, mid_state, map_int_to_type(idx))))
    coll = Collision(env.params, mid_state, map_int_to_type(idx))
    handle_collision!(coll) #modyfikuje pola mid_state, które są referencjami do tego samego co w envie
    if coll.is_goal
        env.done = true
    end
    # if idx <= 4 # przy zderzeniach z udziałem krążka
    #     ϵ = 1e-1
    #     puck.pos += puck.v * ϵ # przesuń trochę zgodnie z nową prędkością - żeby nie wykryło dugi raz tej samej kolizji
    # end
end

function simulate_dt!(env::AirHockeyEnv)
    mallet1, mallet2, puck = env.state.agent1, env.state.agent2, env.state.puck
    t_remain = env.params.dt

    while t_remain > 0 && !env.done
        tx, ty = time_to_wall(env, puck)
        tm1 = time_to_mallet(env, puck, mallet1) 
        tm2 = time_to_mallet(env, puck, mallet2)
        tm1_to_wallx, tm1_to_wally = time_to_wall(env, mallet1)
        tm2_to_wallx, tm2_to_wally = time_to_wall(env, mallet2)
        # Znajdź najbliższe zdarzenie - UWAGA: tak naprawdę tylko ten pierwszy czas jest zawsze prawdziwy.
        # Obliczone czasy nie uwzględniają  bowiem zderzenia, które się wydarzy w minimum z tych czasów.
        # println(times)
        times = [tx, ty, tm1, tm2, tm1_to_wallx, tm2_to_wallx, tm1_to_wally, tm2_to_wally]
        t_next, idx = findmin(times)

        
        if t_next != Inf # doszło do kolizji
            update_positions!(puck, mallet1, mallet2, t_next) # przewiń symulacje do momentu kolizji
            execute_collision!(env, idx)
        else
            t_next = t_remain
            update_positions!(puck, mallet1, mallet2, t_next) #przewiń do końca dt
        end
        t_remain -= t_next # zmniejsz pozostały czas o czas już zhandlowany
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

    # predkości zaraz po wykonaniu akcji (dodaniu dv do poprzednich wektorów predkości). 
    # Zmieniam stan enva na after_state - stan po wykonaniu akcji, ale przed stanem następnym
    env.state.agent1.v .+= [action1.dvx, action1.dvy]
    env.state.agent2.v .+= [action2.dvx, action2.dvy]
    env.step_acc_reward1, env.step_acc_reward2 = 0.0f0, 0.0f0

    simulate_dt!(env)
    r1, r2, s_next, d = reward(env)..., deepcopy(env.state), deepcopy(is_terminated(env))
    if is_terminated(env); reset!(env);println("GOL") end
    
    return r1, r2, s_next, d
end

is_terminated(env::AirHockeyEnv) = env.done

