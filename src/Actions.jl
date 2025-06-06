
function action(env::AirHockeyEnv, p::RandomPolicy)
    Action(
        rand(p.distribution_angle),
        rand(p.distribution_len))
end


function action(env::AirHockeyEnv, p::ActorPolicy)
    state = env.state
    # nie da sie sensownie dodawać wektorów polarnych więc i tak trzeba konwertować na kartezjanskie
    policy_polar_action = denormalize_action(env.params,p.actor.model(state_to_input(env, state)))
    # w teorii wypadałoby jakoś normalizować szum bo jest samplowany w kartezjańskich i nie wiemy jak one sie mają do zakresów [-1,1]² w polarnych
    action_cartesian = convert_from_polar_to_cartesian(policy_polar_action...) .+ rand(p.noise) #to rand przyjmuje sie ze jest zdenormalizowane
    normalized_polar_action = action_cartesian |>
        x -> convert_from_cartesian_to_polar(x...) |>
        x -> Action(x...) |>
        x -> normalize_action(env.params, x) |>
        x -> clamp.(x, -1, 1)

    return output_to_action(env, normalized_polar_action)
end


function action(env::AirHockeyEnv, p::LearntPolicy)
    state = env.state
    action = p.model(state_to_input(env, state)) 
    clamp.(action, -1, 1)
    return output_to_action(env, action)
end