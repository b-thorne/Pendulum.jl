using ComponentArrays, Lux, Flux, DiffEqFlux, DifferentialEquations
using OptimizationFlux, Optimization, OptimizationOptimJL
using Random, Plots

# ODE definition
function damped_pendulum!(du, u, p, t)
    du[1] = u[2]
    du[2] = -9.81 * sin(u[1]) - 0.5 * u[2]
end

# Sampling initial conditions and generating data
N = 500  # Number of different initial conditions
rng = Random.default_rng()
datasize = 20
tspan = (0.0f0, 3.0f0)
tsteps = range(tspan[1], tspan[2], length=datasize)

θs = rand(N) .* (2π) .- π
ωs = rand(N) .* 2 .- 1

# Solve all sets of initial conditions
uₜs = map(θs, ωs) do θ, ω
    u₀ = [θ; ω]
    prob = ODEProblem(damped_pendulum!, u₀, tspan)
    Array(solve(prob, Tsit5(), saveat=tsteps))
end

# Plot a couple of the solved trajectories
function plot_trajectories(uₜ, tsteps, indices)
    p1 = plot(legend=:bottomright, xlabel="Time", ylabel="θ", title="θ over Time")
    p2 = plot(legend=:bottomright, xlabel="Time", ylabel="ω", title="ω over Time")
    
    for i in indices
        traj = uₜ[i]
        plot!(p1, tsteps, traj[1, :], label="Initial Cond. $i")
        plot!(p2, tsteps, traj[2, :], label="Initial Cond. $i")
    end

    plot(p1, p2, layout=(2, 1))
end

plot_trajectories(uₜs, tsteps,  [1, 10, 50])

# Convert to an appropriate data structure for training
train_data = map((θ, ω, uₜ) -> ([θ; ω], uₜ), θs, ωs, uₜs)

# Neural ODE problem set up
width = 25
ndim = 2
dudt2 = Lux.Chain(
                  Lux.Dense(ndim, width, relu),
                  Lux.Dense(ndim, width, relu),
                  Lux.Dense(ndim, width, relu),
                  Lux.Dense(width, ndim))
ϕ, st = Lux.setup(rng, dudt2)
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

# du = fᵩ(u,p,t)
function f(ϕ, u₀)
  Array(prob_neuralode(u₀, ϕ, st)[1])
end

# L2 loss
function L2(ϕ)
    map(train_data) do (u₀, uₜ)
        sum(abs2, uₜ .- f(ϕ, u₀))
    end |> sum
end

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, _) -> L2(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(ϕ))

result_neuralode = Optimization.solve(optprob, Adam(0.01), maxiters=300)

result_neuralode

# Generate predictions for a few training examples
training_indices = [1, 10, 50]
training_preds = map(training_indices) do i
    u₀ = [θs[i]; ωs[i]]
    f(result_neuralode.u, u₀)
end

# Generate new test data
N_test = 10
θs_test = rand(N_test) .* (2π) .- π
ωs_test = rand(N_test) .* 2 .- 1

# Generate predictions for a few test examples
test_preds = map(θs_test, ωs_test) do θ, ω
    u₀ = [θ; ω]
    f(result_neuralode.u, u₀)
end

# Function to plot comparisons
function plot_comparisons(uₜs, preds, tsteps, indices, title)
    p1 = plot(legend=:bottomright, xlabel="Time", ylabel="θ", title="θ - $title")
    p2 = plot(legend=:bottomright, xlabel="Time", ylabel="ω", title="ω - $title")

    for i in indices
        plot!(p1, tsteps, uₜs[i][1, :], label="True $i")
        plot!(p1, tsteps, preds[i][1, :], label="Pred $i", linestyle=:dash)
        plot!(p2, tsteps, uₜs[i][2, :], label="True $i")
        plot!(p2, tsteps, preds[i][2, :], label="Pred $i", linestyle=:dash)
    end

    plot(p1, p2, layout=(2, 1))
end

# Plot comparisons for training and test data
plot_comparisons(uₜs, training_preds, tsteps, training_indices, "Training")
plot_comparisons(uₜs[1:N_test], test_preds, tsteps, 1:N_test, "Test")


