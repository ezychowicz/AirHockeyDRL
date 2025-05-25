# krytyk - sieć aproksymująca Q*(s,a)

function state_to_input(env::AirHockeyEnv, state::State)
    vcat(normalize_position(env.params, state.puck.pos),
        normalize_position(env.params, state.agent1.pos),
        normalize_position(env.params, state.agent2.pos),
        )
end


actor = Chain(
    Dense(6, 64, relu), # Wejście: Wektor stanu
    Dense(64, 64, relu),
    Dense(64, 2, tanh)   # Wyjście: [-1,1]²
)

target_actor = deepcopy(actor)