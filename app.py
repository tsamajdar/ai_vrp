import math
import random
import statistics
from deap import base, creator, tools
import matplotlib.pyplot as plt
import streamlit as st

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="VRP Optimizer", page_icon="🚚", layout="centered"
)

st.title("🚚 Vehicle Routing Problem (VRP) Optimizer")
st.write(
    "Optimize vehicle delivery routes and balance workloads using a"
    " multi-objective Genetic Algorithm (DEAP)."
)

# --- Sidebar Inputs for User Customization ---
st.sidebar.header("Optimization Parameters")
num_locations = st.sidebar.slider(
    "Number of Locations", min_value=5, max_value=25, value=10
)
num_vehicles = st.sidebar.slider(
    "Number of Vehicles", min_value=1, max_value=6, value=3
)
pop_size = st.sidebar.slider(
    "Population Size", min_value=20, max_value=200, value=100, step=10
)
ngen = st.sidebar.slider(
    "Generations", min_value=10, max_value=100, value=40, step=5
)
mut_pb = st.sidebar.slider(
    "Mutation Probability", min_value=0.01, max_value=0.2, value=0.05, step=0.01
)
cx_pb = st.sidebar.slider(
    "Crossover Probability", min_value=0.5, max_value=1.0, value=0.7, step=0.05
)

# Run Button
run_button = st.sidebar.button("🚀 Run Optimization")

if run_button:
  with st.spinner("Running Genetic Algorithm optimization... Please wait."):

    # 1. Generate Locations & Depot dynamically based on user inputs
    random.seed(42)  # For reproducible random locations
    locations = [
        (random.randint(0, 100), random.randint(0, 100))
        for _ in range(num_locations)
    ]
    depot = (50, 50)

    # 2. DEAP Setup
    if "FitnessMin" in creator.__dict__:
      del creator.FitnessMin
    if "Individual" in creator.__dict__:
      del creator.Individual

    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register(
        "indices", random.sample, range(num_locations), num_locations
    )
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, toolbox.indices
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    # 3. Fitness Function
    def evalVRP(individual):
      def get_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

      chunk_size = math.ceil(len(individual) / num_vehicles)
      route_distances = []

      for i in range(num_vehicles):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(individual))
        segment = individual[start:end]

        if not segment:
          route_distances.append(0.0)
          continue

        route = [depot] + [locations[idx] for idx in segment] + [depot]
        vehicle_dist = sum(
            get_distance(route[j], route[j + 1]) for j in range(len(route) - 1)
        )
        route_distances.append(vehicle_dist)

      total_distance = sum(route_distances)
      std_dev = (
          statistics.stdev(route_distances) if len(route_distances) > 1 else 0
      )
      return total_distance, std_dev


    toolbox.register("evaluate", evalVRP)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=mut_pb)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 4. Evolution Loop
    pop = toolbox.population(n=pop_size)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
      ind.fitness.values = fit

    min_fitness_history = []
    avg_fitness_history = []

    for generation in range(ngen):
      offspring = toolbox.select(pop, len(pop))
      offspring = [toolbox.clone(ind) for ind in offspring]

      for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cx_pb:
          toolbox.mate(child1, child2)
          del child1.fitness.values
          del child2.fitness.values

      for mutant in offspring:
        if random.random() < mut_pb:
          toolbox.mutate(mutant)
          del mutant.fitness.values

      invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
      fitnesses = list(map(toolbox.evaluate, invalid_ind))
      for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

      pop[:] = offspring

      fits = [ind.fitness.values[0] for ind in pop]
      min_fitness_history.append(min(fits))
      avg_fitness_history.append(sum(fits) / len(fits))

    # 5. Extract Best Individual
    best_individual = tools.selBest(pop, k=1)[0]
    best_distance, best_std = best_individual.fitness.values

    # --- Display Results in Streamlit ---
    st.success("Optimization Complete!")

    col1, col2 = st.columns(2)
    col1.metric("Best Total Distance", f"{best_distance:.2f} units")
    col2.metric("Route Imbalance (Std Dev)", f"{best_std:.2f}")

    # Plot 1: Convergence Graph
    st.subheader("📈 Convergence Over Generations")
    fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
    ax_conv.plot(
        min_fitness_history, label="Best Fitness (Min Distance)", color="blue"
    )
    ax_conv.plot(
        avg_fitness_history,
        label="Average Fitness",
        color="orange",
        linestyle="--",
    )
    ax_conv.set_xlabel("Generation")
    ax_conv.set_ylabel("Total Distance")
    ax_conv.legend()
    ax_conv.grid(True)
    st.pyplot(fig_conv)
    plt.close(fig_conv)

    # Plot 2: Route Map
    st.subheader("🗺️ Optimized Vehicle Routes")
    fig_route, ax_route = plt.subplots(figsize=(8, 6))
    chunk_size = math.ceil(len(best_individual) / num_vehicles)
    colors = ["r", "g", "b", "c", "m", "y", "k"]

    for i in range(num_vehicles):
      start = i * chunk_size
      end = min((i + 1) * chunk_size, len(best_individual))
      segment = best_individual[start:end]

      if not segment:
        continue

      route_indices = [depot] + [locations[idx] for idx in segment] + [depot]
      rx = [p[0] for p in route_indices]
      ry = [p[1] for p in route_indices]
      color = colors[i % len(colors)]
      ax_route.plot(
          rx,
          ry,
          marker="o",
          linestyle="-",
          color=color,
          linewidth=2,
          label=f"Vehicle {i+1}",
      )

    ax_route.scatter(
        [depot[0]],
        [depot[1]],
        color="black",
        s=200,
        marker="s",
        label="Depot",
        zorder=5,
    )
    all_x = [loc[0] for loc in locations]
    all_y = [loc[1] for loc in locations]
    ax_route.scatter(
        all_x, all_y, color="gray", s=70, marker="o", label="Locations", zorder=3
    )

    ax_route.set_title("Optimized Vehicle Routing Problem (VRP) Routes")
    ax_route.set_xlabel("X Coordinate")
    ax_route.set_ylabel("Y Coordinate")
    ax_route.legend(loc="upper right")
    ax_route.grid(True)
    st.pyplot(fig_route)
    plt.close(fig_route)

else:
  st.info(
      "👈 Configure your parameters in the sidebar and click **Run"
      " Optimization** to start."
  )