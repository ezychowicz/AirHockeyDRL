push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using LinearAlgebra
using Distributions 
using DataStructures
using AirHockey 
using Flux
function parse_experience(s, a, r, s_next, d)
    s = AirHockey.state_to_input(env, s)
    a = AirHockey.normalize_action(env.params, a)
    s_next = AirHockey.state_to_input(env, s_next)
    d = d ? 1 : 0
    return Experience(s,a,r,s_next,d)
end

function train(num_episodes)
    σ² = 0.25 # 95% wyników bedzie sie mieścić w [-1,1]
    Σ = diagm([σ², 2])
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
            update_NNs!(policy1.actor, critic1, replay_buff1; samples = 10)
            update_NNs!(policy2.actor, critic2, replay_buff2; samples = 10)
        end
    end
    puck_states, mallet1_states, mallet2_states, times_cumul
end

function update_NNs!(
    actor::NN, critic::NN, replay_buff::CircularBuffer{Experience};
    samples::Int64 = 10, γ = 0.95, ρ = 0.95)

    batch = rand(replay_buff, samples)


    s_a_inputs = hcat([vcat(b.s, b.a) for b in batch]...)  # kolumny to inputy
    ss = hcat([b.s for b in batch]...)
    next_ss = hcat([b.s_next for b in batch]...) # kolumny to stany
    rewards = Float32[b.r for b in batch] 
    dones = Float32[b.d for b in batch]  # będzie 0.0 albo 1.0

    next_actions = actor.target(next_ss)
    critic_input_next =  vcat(next_ss, next_actions)
    target_q = critic.target(critic_input_next)
    target_ys = rewards .+ γ .* (1 .- dones) .* target_q


    loss(nn_out, expected_out) = Flux.Losses.mse(nn_out, expected_out) 
    train_set = [(s_a_inputs, target_ys)]
    Flux.train!(critic.model, train_set, critic.optimizer, (m, x, y) -> loss(m(x), y)) 

    # gradient ascent na wartości oczekiwanej Q stanów s z batcha
    function actor_loss(ins) 
        actor_actions = actor.model(ins)
        # dodaj do wektorów s akcje z polityki (stwórz wejścia do Q)
        critic_inputs = vcat(ins, actor_actions)
        return -mean(critic.model(critic_inputs))
    end
    train_set = [(ss, nothing)]
    Flux.train!(actor.model, train_set, actor.optimizer, (_, x, _) -> actor_loss(x))
end


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
agent1 = Mallet([0.0f0, 0.0f0], [5.0f0, 25.0f0])
agent2 = Mallet([0.0f0, 0.0f0], [95.0f0, 25.0f0])
env = AirHockeyEnv(params, State(agent1, agent2, Puck([0.0f0, 0.0f0], [50.0f0, 25.0f0])), false)
# w sumie to co wyzej bez znaczenia
AirHockey.reset!(env)

train(100000)