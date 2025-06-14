
map_int_to_type(type::Int)::CollisionType = COLLISIONS_MAPPING[type]


function handle_collision!(collision::Collision{PuckWallCollision_X, V}) where {V<:Real}
    puck = collision.mid_state.puck
    e = collision.params.band_e_loss # energy loss coefficient
    puck.v[1] = -e*puck.v[1] 
    if (collision.params.y_len/2 + collision.params.goal_width/2 > puck.pos[2] > collision.params.y_len/2 - collision.params.goal_width/2)
        collision.is_goal = true
    end
end

function handle_collision!(collision::Collision{PuckWallCollision_Y, V}) where {V<:Real}
    puck = collision.mid_state.puck
    e = collision.params.band_e_loss # energy loss coefficient
    puck.v[2] = -e*puck.v[2] 
end

function handle_collision!(collision::Collision{MalletWallCollision_X, V}) where {V<:Real}
    mallet1, mallet2 = collision.mid_state.agent1, collision.mid_state.agent2
    mallet = !isnothing(mallet1) ? mallet1 : mallet2
    # mallet.v = zeros(Float32, 2) # po prostu następuje zatrzymanie malleta przy zderzeniu
    e = collision.params.band_e_loss
    mallet.v .= [0.0f0, mallet.v[2]]
end

function handle_collision!(collision::Collision{MalletWallCollision_Y, V}) where {V<:Real}
    mallet1, mallet2 = collision.mid_state.agent1, collision.mid_state.agent2
    mallet = !isnothing(mallet1) ? mallet1 : mallet2
    # mallet.v = zeros(Float32, 2) 
    e = collision.params.band_e_loss
    mallet.v .= [mallet.v[1], 0.0f0]
end

function handle_collision!(collision::Collision{PuckMalletCollision, V}) where {V<:Real}
    C = collision.params.restitution
    puck = collision.mid_state.puck
    mallet1, mallet2 = collision.mid_state.agent1, collision.mid_state.agent2
    mallet = !isnothing(mallet1) ? mallet1 : mallet2
    # zbudujmy bazę wektorów, gdzie jeden jest w kierunku środków obiektów, drugi prostopadły
    base1 = normalize(mallet.pos .- puck.pos)
    base2 = [-base1[2], base1[1]] # taki wektor jest prostopadly i znormalizowany. wychodzi to z układu rownań: prostopadłość ⩓ normalny
    A = hcat(base1, base2)
    v_mallet = A \ mallet.v # współczynniki predkości w nowej bazie dla malleta ([v_normalna, v_prostopadla])
    v_puck = A \ puck.v

    mallet_mass, puck_mass = collision.params.mallet_mass, collision.params.puck_mass
    # Uwzględniając 2 równania: 
    # 1. zasadę zachowania pędu w osi normalnej (tylko ją uwzględniamy)
    # pęd przed = pęd po (można na skalarach skoro już mamy nową bazę i działamy w jednej osi)
    # mallet_mass * v_mallet[1] + puck_mass * v_puck[1] = mallet_mass * v_mallet_after + puck_mass * v_puck_after
    # 2. równanie "zachowania" energii z pewnym współczynnikiem straty
    # Zakładam, że prędkość malleta po zderzeniu jest zerowa - zatrzymuje się (gracz po prostu 
    # kontroluje odrzut)

    mallet.v = zeros(Float32, 2)

    u1 = v_mallet[1]
    u2 = v_puck[1]
    m1 = mallet_mass
    m2 = puck_mass

    v_normal_puck = (m1*u1 + m2*u2 + m1*C*(u1 - u2)) / (m1 + m2)

    v_puck[1] = v_normal_puck # zaktualizowana składowa normalna 
    @. puck.v = base1*v_puck[1] + base2*v_puck[2] # kombinacja liniowa w 
    # println("nowe v:$(puck.v) $(puck.pos)")
end
