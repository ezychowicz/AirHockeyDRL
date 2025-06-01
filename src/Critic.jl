# krytyk - sieć aproksymująca Q*(s,a)

function state_action_to_input(env::AirHockeyEnv, state::State, action::Action)
    vcat(normalize_position(env.params, state.puck.pos),
        normalize_position(env.params, state.agent1.pos),
        normalize_position(env.params, state.agent2.pos),
        normalize_action(env.params, action))
end

function init_critic()
    critic = Chain(
        Dense(8, 64, relu),     # warstwa wejściowa: 8 -> 64
        Dense(64, 64, relu),    # ukryta warstwa: 64 -> 64
        Dense(64, 1)            # wyjście: jedna wartość Q(s, a)
    )
    optimizer_critic = ADAM(1e-3)
    critic_target = deepcopy(critic)
    NN(critic, optimizer_critic, critic_target)
end




