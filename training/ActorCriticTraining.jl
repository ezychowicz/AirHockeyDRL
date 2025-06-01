push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using LinearAlgebra
using Distributions 
using DataStructures
using AirHockey 
using Flux
using JLD2
include("../prototype/TracedEnv.jl")
using .TracedEnv

using GLMakie
function parse_experience(s, a, r, s_next, d)
    s = AirHockey.state_to_input(env, s)
    a = AirHockey.normalize_action(env.params, a)
    s_next = AirHockey.state_to_input(env, s_next)
    d = d ? 1 : 0
    return Experience(s,a,r,s_next,d)
end

function train(num_episodes)
    σ² = 0.25 # 95% wyników bedzie sie mieścić w [-1,1]
    Σ = diagm([σ², 2]) # diagonalna Matrix{}
    policy1 = ActorPolicy(AirHockey.init_actor(), MvNormal(zeros(2), Σ))
    policy2 = ActorPolicy(AirHockey.init_actor(), MvNormal(zeros(2), Σ))
    critic1 = AirHockey.init_critic()
    critic2 = AirHockey.init_critic()
    replay_buff1 = CircularBuffer{Experience}(10_000)
    replay_buff2 = CircularBuffer{Experience}(10_000)
    update_freq, r = 50, 0 
    start_learning = false
    for ep ∈ 1:num_episodes
        a1 = AirHockey.action(env, policy1)
        a2 = AirHockey.action(env, policy2)
        s = deepcopy(env.state)
        r1, r2, s_next, d = AirHockey.step!(env, a1, a2)
        push!(replay_buff1, parse_experience(s,a1,r1,s_next,d))
        push!(replay_buff2, parse_experience(s,a2,r2,s_next,d))
        # Pierwsze 10000 kroków nie ucz się
        if !start_learning && isfull(replay_buff1)
            r = rem(ep, update_freq) # teraz co update_freq (czyli gdy ep % update_freq == r) będzie update
            start_learning = true
        end

        # Po 10000 kroków ucz się co update_freq 
        if start_learning && ep % update_freq == r
            update_NNs!(policy1.actor, critic1, replay_buff1; samples = 128)
            update_NNs!(policy2.actor, critic2, replay_buff2; samples = 128)
        end
    end
    save_models(policy1.actor.model, policy2.actor.model)
end

function update_NNs!(
    actor::NN, critic::NN, replay_buff::CircularBuffer{Experience};
    samples::Int64 = 10, γ = 0.99, ρ = 0.99)

    batch = rand(replay_buff, samples)


    s_a_inputs = hcat([vcat(b.s, b.a) for b in batch]...)  # kolumny to inputy
    ss = hcat([b.s for b in batch]...)
    next_ss = hcat([b.s_next for b in batch]...) # kolumny to stany
    rewards = Float32[b.r for b in batch] 
    dones = Float32[b.d for b in batch]  # będzie 0.0 albo 1.0

    next_actions = actor.target(next_ss)
    critic_input_next =  vcat(next_ss, next_actions)
    target_q = critic.target(critic_input_next)
    target_ys = transpose(rewards) .+ γ .* (1 .- transpose(dones)) .* target_q

    function loss(nn_out, expected_out)
        loss_val = Flux.Losses.mse(nn_out, expected_out)
        # println("Critic loss:$loss_val")
        loss_val
    end
    train_set = [(s_a_inputs, target_ys)]
    opt_state1 = Flux.setup(critic.optimizer, critic.model)
    Flux.train!((m, x, y) -> loss(m(x), y), critic.model, train_set, opt_state1) 

    # gradient ascent na wartości oczekiwanej Q stanów s z batcha
    function actor_loss(ins)         
        actor_actions = actor.model(ins)
        # dodaj do wektorów s akcje z polityki (stwórz wejścia do Q)
        critic_inputs = vcat(ins, actor_actions)
        loss_val = -mean(critic.model(critic_inputs))
        # println("Actor loss:$loss_val")
        return -mean(critic.model(critic_inputs))
    end
    train_set = [(ss, nothing)]
    opt_state2 = Flux.setup(actor.optimizer, actor.model)
    Flux.train!((_, x, _) -> actor_loss(x), actor.model, train_set, opt_state2)

    # polyak update of target networks
    soft_update!(critic.target, critic.model, ρ)
    soft_update!(actor.target, actor.model, ρ)
end

function soft_update!(target::Chain, source::Chain, ρ)
    @assert length(Flux.trainables(target)) == length(Flux.trainables(source))

    for (t, s) in zip(Flux.trainables(target), Flux.trainables(source))
        t .= ρ .* t .+ (1 - ρ) .* s
    end
end

function save_models(model1::Chain, model2::Chain)
    actor1_state = Flux.state(model1)
    actor2_state = Flux.state(model2)
    JLD2.@save joinpath(@__DIR__, "..", "models", "actor1.jld2") actor1_state 
    JLD2.@save joinpath(@__DIR__, "..", "models", "actor2.jld2") actor2_state
end


function train_with_visualization(num_episodes)
    σ² = 0.25 # 95% wyników bedzie sie mieścić w [-1,1]
    Σ = diagm([σ², 2]) # diagonalna Matrix{}
    policy1 = ActorPolicy(AirHockey.init_actor(), MvNormal(zeros(2), Σ))
    policy2 = ActorPolicy(AirHockey.init_actor(), MvNormal(zeros(2), Σ))
    critic1 = AirHockey.init_critic()
    critic2 = AirHockey.init_critic()
    replay_buff1 = CircularBuffer{Experience}(10_000)
    replay_buff2 = CircularBuffer{Experience}(10_000)
    update_freq, r = 50, 0 
    start_learning = false

    puck_states = Vector{AirHockey.Puck}()
    mallet1_states = Vector{AirHockey.Mallet}()
    mallet2_states = Vector{AirHockey.Mallet}()
    time_diffs = Vector{Float32}()
    result_states = Vector{Union{Bool, Nothing}}()
    rewards_states = Vector{Vector{Float64}}()
    push!(puck_states, deepcopy(env.state.puck))
    push!(mallet1_states, deepcopy(env.state.agent1))
    push!(mallet2_states, deepcopy(env.state.agent2))
    push!(result_states, nothing)
    push!(rewards_states, Float32[0,0])
    for ep ∈ 1:num_episodes
        a1 = AirHockey.action(env, policy1)
        a2 = AirHockey.action(env, policy2)
        s = deepcopy(env.state)
        env_freeze = deepcopy(env) # poniżej robimy krok w srodowisku uczącym, a potrzebujemy zrobić ten sam krok w środowisku rejestrującym potem
        r1, r2, s_next, d = AirHockey.step!(env, a1, a2)
        push!(replay_buff1, parse_experience(s,a1,r1,s_next,d))
        push!(replay_buff2, parse_experience(s,a2,r2,s_next,d))
        # Pierwsze 10000 kroków nie ucz się
        if !start_learning && isfull(replay_buff1)
            r = rem(ep, update_freq) # teraz co update_freq (czyli gdy ep % update_freq == r) będzie update
            start_learning = true
        end

        # Po 10000 kroków ucz się co update_freq 
        if start_learning && ep % update_freq == r
            update_NNs!(policy1.actor, critic1, replay_buff1; samples = 10)
            update_NNs!(policy2.actor, critic2, replay_buff2; samples = 10)
        end


        trace = TracedEnv.step!(env_freeze, a1, a2)  
        append!(time_diffs, trace.times)
        append!(puck_states, trace.puck_trace)
        append!(mallet1_states, trace.mallet1_trace)
        append!(mallet2_states, trace.mallet2_trace)
        append!(result_states, trace.result)
        append!(rewards_states, trace.rewards)
    end
    # println(result_states)
    puck_states, mallet1_states, mallet2_states, time_diffs, result_states, rewards_states
end

params = AirHockey.EnvParams(  
    x_len = 100.0f0,
    y_len = 50.0f0,
    goal_width = 15.0f0,
    puck_radius = 1.5f0,
    mallet_radius = 2.0f0,
    agent1_initial_pos = [10.0f0, 25.0f0],
    agent2_initial_pos = [90.0f0, 25.0f0],
    dt = 0.02f0,
    max_dv = 10.0f0,
    band_e_loss = 0.95f0,
    restitution = 0.99f0,
    puck_mass = 1.5f0,
    mallet_mass = 2.0f0
)



# Tworzymy instancję środowiska z AirHockeyEnv
agent1 = Mallet([0.0f0, 0.0f0], [5.0f0, 25.0f0])
agent2 = Mallet([0.0f0, 0.0f0], [95.0f0, 25.0f0])
env = AirHockeyEnv(params, State(agent1, agent2, Puck([0.0f0, 0.0f0], [50.0f0, 25.0f0])), false)
# w sumie to co wyzej bez znaczenia
AirHockey.reset!(env)



# train(100000)

push!(LOAD_PATH, joinpath(@__DIR__, "..", "prototype"))
using Visualize
visualize(env.params, train_with_visualization(11000)...)
