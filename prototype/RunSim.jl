push!(LOAD_PATH, @__DIR__)   
using Visualize
using JLD2
using Flux
using AirHockey
using Distributions

include("TracedEnv.jl")
using .TracedEnv

function load_agent_models()
    JLD2.@load joinpath(@__DIR__, "..", "models", "actor1.jld2") actor1_state
    JLD2.@load joinpath(@__DIR__, "..", "models", "actor2.jld2") actor2_state
    model1, model2 = AirHockey.make_actor_model(), AirHockey.make_actor_model()
    Flux.loadmodel!(model1, actor1_state)
    Flux.loadmodel!(model2, actor2_state)
    return model1, model2
end

function simulate(env::AirHockeyEnv)
    policy_model1, policy_model2 = load_agent_models()
    policy1 = AirHockey.LearntPolicy(policy_model1)
    # policy2 = AirHockey.LearntPolicy(policy_model2)
    policy2 = RandomPolicy(Uniform(-env.params.max_dvx, env.params.max_dvx), Uniform(-env.params.max_dvy, env.params.max_dvy))
    puck_states = Vector{AirHockey.Puck}()
    mallet1_states = Vector{AirHockey.Mallet}()
    mallet2_states = Vector{AirHockey.Mallet}()
    time_diffs = Vector{Float32}()
    result_states = Vector{Union{Bool, Nothing}}()
    rewards_states = Vector{Vector{Float32}}()
    push!(puck_states, deepcopy(env.state.puck))
    push!(mallet1_states, deepcopy(env.state.agent1))
    push!(mallet2_states, deepcopy(env.state.agent2))
    push!(result_states, nothing)
    # push!(rewards_states, Float32[0,0])
    for _ ∈ 1:1000
        # Tutaj agent wykonuje akcje
        action1 = AirHockey.action(env, policy1)
        action2 = AirHockey.action(env, policy2)
        # println(action1, action2)  # Czy są różne?
        # action1 = AirHockey.Action(0.0f0,0.0f0)
        # action2 = AirHockey.Action(0.0f0,0.0f0)
        trace = TracedEnv.step!(env, action1, action2)  
        append!(time_diffs, trace.times)
        append!(puck_states, trace.puck_trace)
        append!(mallet1_states, trace.mallet1_trace)
        append!(mallet2_states, trace.mallet2_trace)
        append!(result_states, trace.result)
        append!(rewards_states, trace.rewards)
    end
    
    puck_states, mallet1_states, mallet2_states, time_diffs, result_states, rewards_states
end

# === INICJALIZACJA === #
params = AirHockey.EnvParams(  
    x_len = 100.0f0,
    y_len = 50.0f0,
    goal_width = 15.0f0,
    puck_radius = 1.5f0,
    mallet_radius = 2.0f0,
    agent1_initial_pos = [10.0f0, 25.0f0],
    agent2_initial_pos = [90.0f0, 25.0f0],
    dt = 0.02f0,
    band_e_loss = 0.95f0,
    restitution = 0.99f0,
    puck_mass = 1.5f0,
    mallet_mass = 2.0f0,
    max_dvx = 15.0f0,
    max_dvy = 15.0f0,
    max_vxy = 80.0f0
)


# Tworzymy instancję środowiska z AirHockeyEnv
agent1 = AirHockey.Mallet([0.0f0, 0.0f0], [5.0f0, 25.0f0])
agent2 = AirHockey.Mallet([0.0f0, 0.0f0], [95.0f0, 25.0f0])
env = AirHockey.AirHockeyEnv(params, AirHockey.State(agent1, agent2, AirHockey.Puck([0.0f0, 0.0f0], [50.0f0, 25.0f0])), false, 0.0f0, 0.0f0)
# w sumie to co wyzej bez znaczenia
AirHockey.reset!(env)



visualize(env.params, simulate(env)...)
