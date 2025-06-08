# aktor - sieć aproksymująca π*(s)

function state_to_input(env::AirHockeyEnv, state::State)
    vcat(normalize_puck_position(env.params, state.puck.pos),
        normalize_mallet_position(env.params, state.agent1.pos),
        normalize_mallet_position(env.params, state.agent2.pos),
        normalize_velocity(env.params, state.agent1.v),
        normalize_velocity(env.params, state.agent2.v),
        normalize_velocity(env.params, state.puck.v)
        )
end

function output_to_action(env::AirHockeyEnv, output::Vector{V}) where {V <: Real}
    """
    Map NN output from [-1,1]² to Action [-max_dvx, max_dvx] x [-max_dvy, max_dvy]
    """
    max_dvx, max_dvy = env.params.max_dvx, env.params.max_dvy
    Action(output[1]*max_dvx, output[2]*max_dvy)
end

function make_actor_model()
    return Chain(
        Dense(12, 64, relu), # Wejście: Wektor stanu
        Dense(64, 64, relu),
        Dense(64, 2, tanh)   # Wyjście: [-1,1]²
    )
end
function init_actor()
    actor = make_actor_model()
    optimizer_actor = ADAM(1e-3)
    actor_target = deepcopy(actor)
    opt_state = Flux.setup(optimizer_actor, actor)
    NN(actor, optimizer_actor, actor_target, opt_state)
end









