using ComponentArrays, Lux, Flux, DiffEqFlux, DifferentialEquations
using OptimizationFlux, Optimization, OptimizationOptimJL
using OptimizationNLopt
using Random, Plots, JLD2

# ODE definition
function damped_pendulum!(du, u, p, t)
    du[1] = u[2]
    du[2] = -9.81 * sin(u[1]) - 0.5 * u[2]
end

# Sampling initial conditions and generating data
N = 500 # Number of different initial conditions
rng = Random.default_rng()
datasize = 30
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

plot_trajectories(uₜs, tsteps,  [1, 2])

# Convert to an appropriate data structure for training
train_data = map((θ, ω, uₜ) -> ([θ; ω], uₜ), θs, ωs, uₜs)

# Neural ODE problem set up
width = 8
ndim = 2
dudt2 = Lux.Chain(
                  Lux.Dense(ndim, width, relu),
                  Lux.Dense(width, width, relu),
                  Lux.Dense(width, width, relu),
                  Lux.Dense(width, ndim))
ϕ, st = Lux.setup(rng, dudt2)
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat=tsteps)

# du = fᵩ(u,p,t)
function f(ϕ, u₀)
  Array(prob_neuralode(u₀, ϕ, st)[1])
end

# L2 loss
function L2(ϕ)
    map(train_data) do (u₀, uₜ)
        sum(abs2, uₜ .- f(ϕ, u₀)) / length(train_data)
    end |> sum
end

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, _) -> L2(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(ϕ))

function callback(p, loss_val)
    println(loss_val)
    false
end


result_neuralode = Optimization.solve(optprob, Adam(0.05), maxiters=500; callback=callback)
result_neuralode

# Generate predictions for a few training examples
training_indices = 1:3
training_preds = map(training_indices) do i
    u₀ = [θs[i]; ωs[i]]
    f(result_neuralode.u, u₀)
end

# Generate new test data
N_test = 3
θs_test = rand(N_test) .* (2π) .- π
ωs_test = rand(N_test) .* 2 .- 1

# Generate predictions for a few test examples
test_preds = map(θs_test, ωs_test) do θ, ω
    u₀ = [θ; ω]
    f(result_neuralode.u, u₀)
end

test_uₜs = map(θs_test, ωs_test) do θ, ω
    u₀ = [θ; ω]
    prob = ODEProblem(damped_pendulum!, u₀, tspan)
    Array(solve(prob, Tsit5(), saveat=tsteps))
end

function plot_comparisons(uₜs, preds, tsteps, indices, title)
    p1 = plot(legend=:topright, xlabel="Time", ylabel="θ", title="θ - $title")
    p2 = plot(legend=false, xlabel="Time", ylabel="ω", title="ω - $title")
    colors = [:blue, :red, :green, :purple]  # Define an array of colors

    for (idx, i) in enumerate(indices)
        color = colors[idx]  # Pick a color for each example
        plot!(p1, tsteps, uₜs[i][1, :], linecolor=color, label="")
        plot!(p1, tsteps, preds[i][1, :], linecolor=color, linestyle=:dash, label="")
        plot!(p2, tsteps, uₜs[i][2, :], linecolor=color, label="")
        plot!(p2, tsteps, preds[i][2, :], linecolor=color, linestyle=:dash, label="")
    end

    # Add proxy plots for the custom legend
    plot!(p1, [], [], linecolor=:black, linestyle=:solid, label="True")
    plot!(p1, [], [], linecolor=:black, linestyle=:dash, label="Predicted", legend=:topright)

    final_plot = plot(p1, p2, layout=(2, 1))
    #savefig(final_plot, "comparison_plot.svg")  # Save the plot to disk
end

# Plot comparisons for training and test data
plot_comparisons(uₜs, training_preds, tsteps, training_indices, "Training")
plot_comparisons(test_uₜs, test_preds, tsteps, 1:N_test, "Test")

@save "node_model.jld2" ϕ_final=result_neuralode.u
@load "node_model.jld2" ϕ_final


# Set up optimization for final position 
# t=3s, θ=π, ω=0

# L2 loss
function final_state_loss(u₀, uₜfinal=[π, 0])
    pred = f(ϕ_final, u₀)
    sum(abs2, pred[:, end] .- uₜfinal)
end

# Initialize an array to store the objective function values
objective_values = Float64[]

# Define a callback function to plot the objective function value at each iteration
function plotting_callback(x, y, z, optim_state)
    push!(objective_values, y)
    plot(objective_values, xlabel="Iterations", ylabel="Objective Value", legend=false)
    display(current())
end

uᵢ = [π/2; 0.1]

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, _) -> final_state_loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, uᵢ)

result_final_state = Optimization.solve(optprob, Adam(0.01), maxiters=100)
print(result_final_state.retcode)


optprob2 = remake(optprob, u0 = result_final_state.u)
result_final_state_2 = Optimization.solve(optprob2, NLopt.LD_LBFGS())

u₀_constrained = result_final_state_2.u

prob = ODEProblem(damped_pendulum!, u₀_constrained, tspan)
uₜ_check = Array(solve(prob, Tsit5(), saveat=tsteps))
uₜ_pred = f(ϕ_final, u₀_constrained)

# Plot a couple of the solved trajectories
function plot_constrained_trajectories(uₜ, uₜ_pred, tsteps)
    p1 = plot(legend=:bottomright, xlabel="Time", ylabel="θ", title="θ over Time")
    p2 = plot(legend=false, xlabel="Time", ylabel="ω", title="ω over Time")
    
    plot!(p1, tsteps, uₜ[1, :], label="Tsit5")
    plot!(p2, tsteps, uₜ[2, :], label="Neural ODE")

    plot!(p1, tsteps, uₜ_pred[1, :])
    plot!(p2, tsteps, uₜ_pred[2, :])

    fig = plot(p1, p2, layout=(2, 1))
    savefig(fig, "constrained_trajectory.png")
end

plot_constrained_trajectories(uₜ_check, uₜ_pred, tsteps)

@save "constrained_trajectory.jld2" u₀_constrained=result_final_state.u
@load "constrained_trajectory.jld2" u₀_constrained

