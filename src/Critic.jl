# krytyk - sieć aproksymująca Q*(s,a)

function state_action_to_input(env::AirHockeyEnv, state::State, action::Action)
    vcat(normalize_position(env.params, state.puck.pos),
        normalize_position(env.params, state.agent1.pos),
        normalize_position(env.params, state.agent2.pos),
        normalize_action(env.params, action))
end


critic = Chain(
    Dense(8, 64, relu),     # warstwa wejściowa: 8 -> 64
    Dense(64, 64, relu),    # ukryta warstwa: 64 -> 64
    Dense(64, 1)            # wyjście: jedna wartość Q(s, a)
)
optimizer_critic = ADAM(1e-3)


critic = init_critic()
target_critic = deepcopy(critic)



# Bufor doświadczeń (przykładowa implementacja)
replay_buffer = []

function update_critic!(batch, env::AirHockeyEnv; γ=0.99, ρ=0.995)
    s, a, r, s′, d = batch
    
    # Obliczanie targetów (bez propagacji gradientu przez targety!)
    target_actions = target_actor.(s′)
    target_qs = [target_critic(vcat(s′, a′))[1] for (s′, a′) in zip(s′, target_actions)]
    ys = r .+ γ .* (1 .- d) .* target_qs
    
    # Obliczanie aktualnych Q-wartości
    current_qs = [critic(state_action_to_input(env,s,a))[1] for (s, a) in zip(s, a)]
    
    # Strata MSE
    loss = Flux.Losses.mse(current_qs, ys)
    
    # Propagacja wsteczna i aktualizacja
    grads = gradient(() -> loss, params(critic))
    update!(optimizer_critic, params(critic), grads)
    
    # Aktualizacja targetu (polyak averaging)
    for (p_target, p) in zip(params(target_critic), params(critic))
        p_target .= ρ .* p_target .+ (1 - ρ) .* p
    end
end

