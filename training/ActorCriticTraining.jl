push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using LinearAlgebra
using Distributions 
using DataStructures
using AirHockey 
using Flux
using JLD2
using Plots

include("../prototype/TracedEnv.jl")
using .TracedEnv
using PrettyTables
using GLMakie
function parse_experience(s, a, r, s_next, d)
    s = AirHockey.state_to_input(env, s)
    a = AirHockey.normalize_action(env.params, a)
    s_next = AirHockey.state_to_input(env, s_next)
    d = d ? 1 : 0
    return Experience(s,a,r,s_next,d)
end

function train(num_episodes)
    σ² = 1
    Σ = diagm([σ², σ²]) # diagonalna Matrix{}
    policy1 = ActorPolicy(AirHockey.init_actor(), MvNormal(zeros(2), Σ))
    policy2 = ActorPolicy(AirHockey.init_actor(), MvNormal(zeros(2), Σ))
    critic1 = AirHockey.init_critic()
    critic2 = AirHockey.init_critic()
    replay_buff1 = CircularBuffer{Experience}(10_000)
    replay_buff2 = CircularBuffer{Experience}(10_000)
    update_freq, r = 50, 0 
    start_learning = false
    curr_return1, curr_return2 = 0, 0
    returns1 = Float32[]
    returns2 = Float32[]

    for ep ∈ 1:num_episodes
        a1 = AirHockey.action(env, policy1)
        a2 = AirHockey.action(env, policy2)
        s = deepcopy(env.state)
        r1, r2, s_next, d = AirHockey.step!(env, a1, a2)

        curr_return1 += r1
        curr_return2 += r2
        if d
            @info "return1: $curr_return1\n return2: $curr_return2"
            push!(returns1, curr_return1)
            push!(returns2, curr_return2)
            curr_return1 = 0
            curr_return2 = 0
        end

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
    gr()  # używa backendu GR (domyślnie)
    Plots.scatter(
        1:length(returns1), returns1,
        label = "Return Policy 1",
        xlabel = "Episode",
        ylabel = "Return",
        title = "Agent Returns",
        legend = :topright
    )
    Plots.scatter!(1:length(returns2), returns2, label = "Return Policy 2")
    display(current())         
    println("Return całej gry: $(sum(returns1)), $(sum(returns2))")
end

function update_NNs!(
    actor::NN, critic::NN, replay_buff::CircularBuffer{Experience};
    samples::Int64 = 10, γ = 0.99, ρ = 0.90)
    
    batch = rand(replay_buff, samples)


    s_a_inputs = hcat([vcat(b.s, b.a) for b in batch]...)  # kolumny to inputy
    # pretty_table(s_a_inputs)
    ss = hcat([b.s for b in batch]...)
    next_ss = hcat([b.s_next for b in batch]...) # kolumny to stany
    rewards = Float32[b.r for b in batch] 
    dones = Float32[b.d for b in batch]  # będzie 0.0 albo 1.0

    next_actions = actor.target(next_ss)
    critic_inputs_next =  vcat(next_ss, next_actions) # vcat na macierzach działa
    target_qs = critic.target(critic_inputs_next)
    target_ys = transpose(rewards) .+ γ .* (1 .- transpose(dones)) .* target_qs
    
    function loss(nn_out, expected_out)
        loss_val = Flux.Losses.mse(nn_out, expected_out)
        # println("Critic loss:$loss_val")
        loss_val
    end
    train_set = [(s_a_inputs, target_ys)]
    Flux.train!((m, x, y) -> loss(m(x), y), critic.model, train_set, critic.opt_state) 

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

    Flux.train!((_, x, _) -> actor_loss(x), actor.model, train_set, actor.opt_state)
    # @info "Critic loss: $(loss(critic.model(s_a_inputs), target_ys))"
    # @info "Actor loss: $(actor_loss(ss))"
    # mean_q = mean(critic.model(s_a_inputs))
    # @info "Mean Q-value: $mean_q"
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


function train_with_visualization(num_episodes; start_recording = 1)
    σ² = 1
    Σ = diagm([σ², σ²]) # diagonalna Matrix{}
    # policy1 = ActorPolicy(AirHockey.init_actor(), MvNormal(zeros(2), Σ))
    # policy2 = ActorPolicy(AirHockey.init_actor(), MvNormal(zeros(2), Σ))
    policy1 = RandomPolicy(Uniform(-env.params.max_dvx, env.params.max_dvx), Uniform(-env.params.max_dvy, env.params.max_dvy))
    # policy2 = RandomPolicy(Uniform(-env.params.max_dvx, env.params.max_dvx), Uniform(-env.params.max_dvy, env.params.max_dvy))
    policy2 = RandomPolicy(Uniform(-0.01, 0.01), Uniform(-0.01, 0.01))
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
    rewards_states = Vector{Vector{Float32}}()
    push!(puck_states, deepcopy(env.state.puck))
    push!(mallet1_states, deepcopy(env.state.agent1))
    push!(mallet2_states, deepcopy(env.state.agent2))
    push!(result_states, nothing)
    push!(rewards_states, Float32[0,0])
    curr_return1, curr_return2 = 0, 0
    for ep ∈ 1:num_episodes
        # println("nowy epizod")
        a1 = AirHockey.action(env, policy1)
        a2 = AirHockey.action(env, policy2)
        s = deepcopy(env.state)
        env_freeze = deepcopy(env) # poniżej robimy krok w srodowisku uczącym, a potrzebujemy zrobić ten sam krok w środowisku rejestrującym potem
        r1, r2, s_next, d = AirHockey.step!(env, a1, a2)
        # println("r1=$r1, r2=$r2")
        curr_return1 += r1
        curr_return2 += r2
        if d
            @info "return1: $curr_return1\n return2: $curr_return2"
            curr_return1 = 0
            curr_return2 = 0
        end
        
        push!(replay_buff1, parse_experience(s,a1,r1,s_next,d))
        push!(replay_buff2, parse_experience(s,a2,r2,s_next,d))
        # Pierwsze 10000 kroków nie ucz się
        if !start_learning && isfull(replay_buff1)
            r = rem(ep, update_freq) # teraz co update_freq (czyli gdy ep % update_freq == r) będzie update
            start_learning = true
            policy1 = ActorPolicy(AirHockey.init_actor(), MvNormal(zeros(2), Σ))
        end

        # Po 10000 kroków ucz się co update_freq 
        if start_learning && ep % update_freq == r
            update_NNs!(policy1.actor, critic1, replay_buff1; samples = 128)
            # update_NNs!(policy2.actor, critic2, replay_buff2; samples = 10)
        end


        trace = TracedEnv.step!(env_freeze, a1, a2)  
        if ep >= start_recording
            append!(time_diffs, trace.times)
            append!(puck_states, trace.puck_trace)
            append!(mallet1_states, trace.mallet1_trace)
            append!(mallet2_states, trace.mallet2_trace)
            append!(result_states, trace.result)
            append!(rewards_states, trace.rewards)
        end
    end


    return puck_states, mallet1_states, mallet2_states, time_diffs, result_states, rewards_states
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
    band_e_loss = 0.95f0,
    restitution = 0.99f0,
    puck_mass = 1.5f0,
    mallet_mass = 2.0f0,
    max_dvx = 15.0f0,
    max_dvy = 15.0f0,
    max_vxy = 80.0f0
)



# Tworzymy instancję środowiska z AirHockeyEnv
agent1 = Mallet([0.0f0, 0.0f0], [5.0f0, 25.0f0])
agent2 = Mallet([0.0f0, 0.0f0], [95.0f0, 25.0f0])
env = AirHockeyEnv(params, State(agent1, agent2, Puck([0.0f0, 0.0f0], [50.0f0, 25.0f0])), false, 0.0f0, 0.0f0)
# w sumie to co wyzej bez znaczenia
AirHockey.reset!(env)



# train(500000)

push!(LOAD_PATH, joinpath(@__DIR__, "..", "prototype"))
using Visualize
visualize(env.params, train_with_visualization(101000; start_recording = 100000)...)
