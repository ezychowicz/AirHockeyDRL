
function action(env::AirHockeyEnv, p::RandomPolicy)
    Action(
        rand(p.distribution_vx),
        rand(p.distribution_vy))
end


function action(env::AirHockeyEnv, p::ActorPolicy)
    state = env.state
    policy_action = denormalize_action(env, p.actor.model(state_to_input(env, state)))
    noise = rand(p.noise)
    # println("policy: $(round.(policy_action, digits = 2))")
    action_cartesian = policy_action .+ noise #to rand przyjmuje sie ze jest zdenormalizowane
    return Action(action_cartesian...)
end


function action(env::AirHockeyEnv, p::LearntPolicy)
    state = env.state
    action = p.model(state_to_input(env, state)) 
    clamp.(action, -1, 1)
    return output_to_action(env, action)
end