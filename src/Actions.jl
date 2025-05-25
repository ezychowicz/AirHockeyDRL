
function action(env::AirHockeyEnv, p::RandomPolicy)
    Action(
        rand(p.distribution_angle),
        rand(p.distribution_len))
end


function action(env::AirHockeyEnv, p::ActorPolicy)
    state = env.state
    action = p.actor(state_to_input(env, state)) .+ rand(p.distribution)
    clamp.(action, -1, 1)
    return output_to_action(env, action)
end