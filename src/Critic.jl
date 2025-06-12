# krytyk - sieć aproksymująca Q*(s,a)

function state_action_to_input(env::AirHockeyEnv, state::State, action::Action)
    vcat(state_to_input(env, state),
        normalize_action(env.params, action))
end

function init_critic()
    critic = Chain(
        Dense(14, 64, relu),     # wejście: Wektor vcat(stan, akcja)
        Dense(64, 64, relu),   
        Dense(64, 1)            # wyjście: jedna wartość Q(s, a)
    )
    optimizer_critic = ADAM(1e-3)
    critic_target = deepcopy(critic)

    opt_state = Flux.setup(optimizer_critic, critic)
    NN(critic, optimizer_critic, critic_target, opt_state)
end




