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
    policy2 = AirHockey.LearntPolicy(policy_model2)
    # policy1 = AirHockey.RandomPolicy(Uniform(-π, π), Uniform(0, env.params.max_dv))
    # policy2 = AirHockey.RandomPolicy(Uniform(-π, π), Uniform(0, env.params.max_dv))
    puck_states = Vector{AirHockey.Puck}()
    mallet1_states = Vector{AirHockey.Mallet}()
    mallet2_states = Vector{AirHockey.Mallet}()
    times_cumul = Vector{Float32}()
    push!(puck_states, deepcopy(env.state.puck))
    push!(mallet1_states, deepcopy(env.state.agent1))
    push!(mallet2_states, deepcopy(env.state.agent2))
    for _ ∈ 1:5000
        # Tutaj agent wykonuje akcje
        action1 = AirHockey.action(env, policy1)
        action2 = AirHockey.action(env, policy2)
        # action1 = AirHockey.Action(0.0f0,0.0f0)
        # action2 = AirHockey.Action(0.0f0,0.0f0)
        trace = TracedEnv.step!(env, action1, action2)  
        append!(times_cumul, trace.times)
        append!(puck_states, trace.puck_trace)
        append!(mallet1_states, trace.mallet1_trace)
        append!(mallet2_states, trace.mallet2_trace)
    end
    puck_states, mallet1_states, mallet2_states, times_cumul
end

# === INICJALIZACJA === #
params = AirHockey.EnvParams(  
    x_len = 100.0f0,
    y_len = 50.0f0,
    goal_width = 15.0f0,
    puck_radius = 1.0f0,
    mallet_radius = 2.0f0,
    agent1_initial_pos = [10.0f0, 25.0f0],
    agent2_initial_pos = [90.0f0, 25.0f0],
    dt = 0.02f0,
    max_dv = 10.0f0,
    band_e_loss = 0.95f0,
    restitution = 0.99f0,
    puck_mass = 1.0f0,
    mallet_mass = 2.0f0
)

# Tworzymy instancję środowiska z AirHockeyEnv
agent1 = AirHockey.Mallet([0.0f0, 0.0f0], [5.0f0, 25.0f0])
agent2 = AirHockey.Mallet([0.0f0, 0.0f0], [95.0f0, 25.0f0])
env = AirHockey.AirHockeyEnv(params, AirHockey.State(agent1, agent2, AirHockey.Puck([0.0f0, 0.0f0], [50.0f0, 25.0f0])), false)
# w sumie to co wyzej bez znaczenia
AirHockey.reset!(env)



visualize(env.params, simulate(env)...)
