# krytyk - sieć aproksymująca Q*(s,a)

function state_to_input(env::AirHockeyEnv, state::State)
    vcat(normalize_position(env.params, state.puck.pos),
        normalize_position(env.params, state.agent1.pos),
        normalize_position(env.params, state.agent2.pos),
        )
end

function output_to_action(env::AirHockeyEnv, output::Vector{Float32})
    """
    Map NN output from [-1,1]² to Action [-π, π] x [0, max_dv] 
    """
    max_dv = env.params.max_dv
    Action(output[1]*π, (output[2] + 1)/2*max_dv)
end

function init_actor()
    actor = Chain(
        Dense(6, 64, relu), # Wejście: Wektor stanu
        Dense(64, 64, relu),
        Dense(64, 2, tanh)   # Wyjście: [-1,1]²
    )
    optimizer_actor = ADAM(1e-3)
    actor_target = deepcopy(actor)
    NN(actor, optimizer_actor, actor_target)
end









