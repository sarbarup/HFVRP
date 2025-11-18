"""
Per-vehicle-time workaround by grouping vehicles into speed classes.

What this implements
- Groups vehicles into speed classes (by unique top speed values).
- For each speed class, registers a time transit callback that computes arc time
  using the class' speed (m/min).
- Adds one Time dimension per class using that class-specific transit callback.
  For the vehicle-time capacity parameter of each class-dimension we:
    - set the vehicle's own capacity to its true max time (derived from its range & speed)
    - set vehicles not in that class to a very large capacity (so that the class-dimension
      is non‑binding for those vehicles)
- Keeps Capacity and Range dimensions as before.
- Cost per arc remains per-vehicle and in consistent monetary units (distance $ + labor $).
- Reporting computes true per-vehicle times/costs using each vehicle's true speed (for clarity).

Why this works (practical explanation to present to a panel)
- OR-Tools does not provide a direct per-vehicle time transit callback for a single Time dimension.
- Creating one Time dimension per speed class and making non-member vehicles unconstraining
  (very large capacity) for that dimension makes the dimension representing the vehicle's
  actual speed the binding one. Other class-dimensions still accumulate times but their
  very-large capacity prevents them from being active constraints for vehicles not belonging
  to that class.
- This yields accurate per-vehicle time feasibility for vehicles that are members of a class,
  while remaining practical and efficient when vehicles can be grouped into a small number
  of speed classes.

Notes / caveats
- If every vehicle has a unique speed (15 distinct speeds), this is equivalent to one dimension
  per vehicle, which increases model size but still works.
- The large capacities for non-member vehicles should be large enough to avoid accidentally
  constraining routes, but not so large as to cause integer overflow in the solver.
- Distances are still Haversine (straight-line) and should be replaced with road distances
  for production use.

Usage
- Requires ortools, numpy, pandas installed.
- Run the file as a script, it will build the model, solve (10s time limit) and print a summary.

Author: adapted to implement per-vehicle time via speed classes
"""

import math
import numpy as np
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# -------------------------
# Parameters & constants
# -------------------------
COST_SCALE_FACTOR = 1000  # scale $ to integer units for OR-Tools cost
TIME_SCALE_FACTOR = 100   # scale minutes to integer units for Time dimension
EARTH_RADIUS_M = 6371000

NUM_LOCATIONS = 50
NUM_NODES = NUM_LOCATIONS + 1  # include depot
NUM_VEHICLES = 15

# Monetary conversion parameters (scenario-specific)
FUEL_COST_PER_KM = 0.40  # $ per km (fuel + maintenance)
FUEL_ECONOMY_KM_PER_GALLON = 10.0  # km per gallon (approx; heavy vehicles might be lower)

# Large capacity for non-member vehicles in class dimensions
# Keep within 32-bit signed integer safe range for OR-Tools (use ~1e9)
LARGE_TIME_CAPACITY = int(1e9)


# -------------------------
# Helper math functions
# -------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return int(round(EARTH_RADIUS_M * c))


# -------------------------
# Data model creation
# -------------------------
def create_data_model():
    data = {}
    data["depot"] = 0
    data["num_vehicles"] = NUM_VEHICLES

    data["locations"] = [
        "Cincinnati, OH (Depot)", "Albany, NY", "Albany, OR", "Baton Rouge, LA", "Birmingham, AL",
        "Boston, MA", "Buffalo, NY", "Cambridge, MA", "Charlotte, NC", "Chicago, IL",
        "Colorado Springs, CO", "Columbus, OH", "Dallas, TX", "Davis, CA", "Decatur, IL",
        "Denver, CO", "Detroit, MI", "Forks, WA", "Frankfort, KY", "Grand Rapids, MI",
        "Hilton Head, SC", "Indianapolis, IN", "Key West, FL", "Las Vegas, NV", "Lexington, KY",
        "Lexington, SC", "Lincoln, NE", "Los Angeles, CA", "Louisville, KY", "Memphis, TN",
        "Miami, FL", "Nashville, TN", "New York City, NY", "Newport, KY", "Omaha, NE",
        "Oakland, CA", "Oshkosh, WI", "Philadelphia, PA", "Pittsburgh, PA", "Portland, ME",
        "Portland, OR", "Salem, MA", "Salt Lake City, UT", "San Francisco, CA", "Santa Claus, IN",
        "Seattle, WA", "Spokane, WA", "St Louis, MO", "Tampa, FL", "Toronto, Canada",
        "Washington, DC (1)"
    ]

    COORDINATES = [
        (39.10, -84.51), # 0 Cincinnati, OH (Depot)
        (42.65, -73.76), # 1 Albany, NY
        (44.63, -123.10), # 2 Albany, OR
        (30.45, -91.19), # 3 Baton Rouge, LA
        (33.52, -86.80), # 4 Birmingham, AL
        (42.36, -71.06), # 5 Boston, MA
        (42.89, -78.87), # 6 Buffalo, NY
        (42.37, -71.11), # 7 Cambridge, MA
        (35.23, -80.84), # 8 Charlotte, NC
        (41.88, -87.63), # 9 Chicago, IL
        (38.83, -104.82), # 10 Colorado Springs, CO
        (39.96, -83.00), # 11 Columbus, OH
        (32.78, -96.80), # 12 Dallas, TX
        (38.54, -121.74), # 13 Davis, CA
        (39.84, -89.00), # 14 Decatur, IL
        (39.74, -104.99), # 15 Denver, CO
        (42.33, -83.05), # 16 Detroit, MI
        (47.96, -124.38), # 17 Forks, WA
        (38.20, -84.87), # 18 Frankfort, KY
        (42.96, -85.67), # 19 Grand Rapids, MI
        (32.22, -80.75), # 20 Hilton Head, SC
        (39.77, -86.16), # 21 Indianapolis, IN
        (24.56, -81.78), # 22 Key West, FL
        (36.17, -115.14), # 23 Las Vegas, NV
        (38.00, -84.50), # 24 Lexington, KY
        (34.00, -81.23), # 25 Lexington, SC
        (40.81, -96.70), # 26 Lincoln, NE
        (34.05, -118.24), # 27 Los Angeles, CA
        (38.25, -85.76), # 28 Louisville, KY
        (35.15, -90.05), # 29 Memphis, TN
        (25.76, -80.19), # 30 Miami, FL
        (36.17, -86.78), # 31 Nashville, TN
        (40.71, -74.01), # 32 New York City, NY
        (39.08, -84.48), # 33 Newport, KY
        (41.26, -96.00), # 34 Omaha, NE
        (37.80, -122.27), # 35 Oakland, CA
        (44.02, -88.54), # 36 Oshkosh, WI
        (39.95, -75.17), # 37 Philadelphia, PA
        (40.44, -80.00), # 38 Pittsburgh, PA
        (43.66, -70.26), # 39 Portland, ME
        (45.52, -122.68), # 40 Portland, OR
        (42.52, -70.89), # 41 Salem, MA
        (40.76, -111.89), # 42 Salt Lake City, UT
        (37.77, -122.42), # 43 San Francisco, CA
        (38.12, -86.92), # 44 Santa Claus, IN
        (47.61, -122.33), # 45 Seattle, WA
        (47.66, -117.43), # 46 Spokane, WA
        (38.63, -90.20), # 47 St Louis, MO
        (27.95, -82.46), # 48 Tampa, FL
        (43.65, -79.38), # 49 Toronto, Canada
        (38.90, -77.04)  # 50 Washington, DC (1)
    ]
    data["coordinates"] = COORDINATES

    data["demands"] = [
        0, 3120, 1296, 2707, 1242, 2863, 667, 1577, 1855, 2406,
        1991, 2454, 917, 969, 1562, 3869, 3494, 1264, 95, 1281,
        952, 591, 2034, 441, 2152, 4502, 1999, 2372, 1420, 3715,
        1445, 1044, 2255, 2448, 2525, 2109, 3158, 3290, 1126, 1345,
        4188, 169, 487, 1928, 2109, 1733, 1638, 722, 892, 2893, 798
    ]

    CAPACITIES_KG = [10000, 10000, 10000, 10000, 10000, 7500, 7500, 7500, 7500, 6000, 6000, 6000, 4000, 4000, 4000]
    HOURLY_RATES = [33, 33, 40, 29, 33, 27, 28, 28, 25, 26, 22, 24, 22, 23, 22]
    TOP_SPEEDS_KMH = [80, 80, 80, 80, 80, 90, 90, 90, 90, 90, 90, 90, 105, 105, 105]
    FUEL_TANKS_GALLON = [150, 150, 150, 145, 150, 120, 130, 130, 130, 130, 125, 125, 130, 135, 135]

    # Data integrity checks
    assert len(COORDINATES) == NUM_NODES, "Coordinates length mismatch"
    assert len(data["demands"]) == NUM_NODES, "Demands length mismatch"
    assert len(CAPACITIES_KG) == NUM_VEHICLES
    assert len(HOURLY_RATES) == NUM_VEHICLES
    assert len(TOP_SPEEDS_KMH) == NUM_VEHICLES
    assert len(FUEL_TANKS_GALLON) == NUM_VEHICLES

    data["vehicle_capacities"] = CAPACITIES_KG
    data["driver_hourly_rate"] = HOURLY_RATES
    data["vehicle_speeds_kmh"] = TOP_SPEEDS_KMH
    # speeds in m/min
    data["vehicle_speeds_m_per_min"] = [max(1, int(round(kmh * 1000.0 / 60.0))) for kmh in TOP_SPEEDS_KMH]

    # compute max range (meters) using FUEL_ECONOMY_KM_PER_GALLON
    data["max_range"] = [int(round(g * FUEL_ECONOMY_KM_PER_GALLON * 1000.0)) for g in FUEL_TANKS_GALLON]

    # precompute pairwise distances (meters)
    distance_matrix = []
    for i in range(NUM_NODES):
        row = []
        for j in range(NUM_NODES):
            if i == j:
                row.append(0)
            else:
                lat1, lon1 = COORDINATES[i]
                lat2, lon2 = COORDINATES[j]
                row.append(haversine_distance(lat1, lon1, lat2, lon2))
        distance_matrix.append(row)
    data["distance_matrix"] = distance_matrix

    return data


# -------------------------
# Core solver & model
# -------------------------
def solve_with_time_classes(time_limit_seconds=10):
    data = create_data_model()

    manager = pywrapcp.RoutingIndexManager(len(data["distance_matrix"]), data["num_vehicles"], data["depot"])
    routing = pywrapcp.RoutingModel(manager)

    # --- Basic callbacks: distance and demand ---
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    distance_transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    def demand_callback(from_index):
        return data["demands"][manager.IndexToNode(from_index)]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    # --- Per-vehicle monetary cost (distance $ + labor $) ---
    for vehicle_id in range(data["num_vehicles"]):
        hourly_rate = data["driver_hourly_rate"][vehicle_id]
        speed_m_per_min = data["vehicle_speeds_m_per_min"][vehicle_id]

        def make_cost_callback(rate_per_hr, speed_m_per_min_local):
            def cost_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                dist_m = data["distance_matrix"][from_node][to_node]
                dist_km = dist_m / 1000.0
                # travel hours by that vehicle
                travel_hours = (dist_m / speed_m_per_min_local) / 60.0 if speed_m_per_min_local > 0 else 1e6
                distance_cost = dist_km * FUEL_COST_PER_KM
                labor_cost = travel_hours * rate_per_hr
                total_cost = distance_cost + labor_cost
                return int(round(total_cost * COST_SCALE_FACTOR))
            return cost_callback

        idx = routing.RegisterTransitCallback(make_cost_callback(hourly_rate, speed_m_per_min))
        routing.SetArcCostEvaluatorOfVehicle(idx, vehicle_id)

    # --- Capacity constraint ---
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        data["vehicle_capacities"],
        True,
        "Capacity"
    )

    # --- Range constraint (meters) ---
    routing.AddDimensionWithVehicleCapacity(
        distance_transit_callback_index,
        0,
        data["max_range"],
        True,
        "Range"
    )

    # --- Time dimensions per speed class ---
    # Group vehicles into speed classes by unique top speeds (km/h)
    speed_to_vehicles = {}
    for vid, kmh in enumerate(data["vehicle_speeds_kmh"]):
        speed_to_vehicles.setdefault(kmh, []).append(vid)

    # For stable iteration, sort classes by speed
    speed_classes = sorted(list(speed_to_vehicles.items()), key=lambda x: x[0])  # list of (kmh, [vehicle_ids])

    time_dimension_names = []
    for class_idx, (class_speed_kmh, vehicle_ids) in enumerate(speed_classes):
        class_speed_m_per_min = max(1, int(round(class_speed_kmh * 1000.0 / 60.0)))
        # Register a transit callback for this class that uses the CLASS speed.
        def make_class_time_callback(speed_m_per_min_local):
            def class_time_cb(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                dist_m = data["distance_matrix"][from_node][to_node]
                time_min = dist_m / speed_m_per_min_local
                return int(round(time_min * TIME_SCALE_FACTOR))
            return class_time_cb

        time_cb_index = routing.RegisterTransitCallback(make_class_time_callback(class_speed_m_per_min))

        # Build per-vehicle maximum times for this dimension:
        # - For vehicles in this class: compute true max time (max_range / vehicle_speed)
        # - For vehicles not in this class: assign a very large capacity so that this dimension
        #   does not constrain them.
        vehicle_time_caps_scaled = []
        for v in range(data["num_vehicles"]):
            if v in vehicle_ids:
                # compute vehicle's max time (minutes) from its range and its actual speed (use vehicle speed)
                vehicle_speed = data["vehicle_speeds_m_per_min"][v]
                vehicle_max_range_m = data["max_range"][v]
                max_time_min = vehicle_max_range_m / vehicle_speed if vehicle_speed > 0 else 0
                vehicle_time_caps_scaled.append(int(round(max_time_min * TIME_SCALE_FACTOR)))
            else:
                vehicle_time_caps_scaled.append(LARGE_TIME_CAPACITY)

        dim_name = f"Time_class_{class_idx}_spd_{class_speed_kmh}kmh"
        routing.AddDimensionWithVehicleCapacity(
            time_cb_index,
            0,
            vehicle_time_caps_scaled,
            True,
            dim_name
        )
        time_dimension_names.append(dim_name)

    # --- Search parameters ---
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = time_limit_seconds
    search_parameters.log_search = False

    # Solve
    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        raise RuntimeError("No solution found. Consider increasing time_limit, loosening constraints, or checking data.")

    return data, manager, routing, solution, time_dimension_names


# -------------------------
# Reporting helpers
# -------------------------
def create_solution_tables(data, manager, routing, solution, time_dimension_names):
    """
    Compute route summary (routes_df) and visit details (visits_df).
    True times and costs are computed with each vehicle's actual speed and rates.
    """
    routes_data = []
    visits = []

    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        if solution.Value(routing.NextVar(index)) == routing.End(vehicle_id):
            continue

        route_distance_m = 0
        route_load = 0
        stops = 0
        total_cost_dollars = 0.0
        total_time_minutes_true = 0.0

        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index != data["depot"]:
                stops += 1
                route_load += data["demands"][node_index]
                visits.append({
                    "Vehicle": vehicle_id,
                    "Stop": stops,
                    "Location": data["locations"][node_index],
                    "Demand_kg": data["demands"][node_index],
                    "Cumulative_Load_kg": route_load,
                    "Latitude": data["coordinates"][node_index][0],
                    "Longitude": data["coordinates"][node_index][1]
                })

            prev_index = index
            index = solution.Value(routing.NextVar(index))
            from_node = manager.IndexToNode(prev_index)
            to_node = manager.IndexToNode(index)
            dist_m = data["distance_matrix"][from_node][to_node]
            route_distance_m += dist_m

            # true per-vehicle time and cost
            speed_m_per_min = data["vehicle_speeds_m_per_min"][vehicle_id]
            time_min = dist_m / speed_m_per_min if speed_m_per_min > 0 else 1e6
            time_hr = time_min / 60.0
            dist_km = dist_m / 1000.0
            distance_cost = dist_km * FUEL_COST_PER_KM
            labor_cost = time_hr * data["driver_hourly_rate"][vehicle_id]
            arc_cost = distance_cost + labor_cost

            total_time_minutes_true += time_min
            total_cost_dollars += arc_cost

        routes_data.append({
            "Vehicle_ID": vehicle_id,
            "Capacity_kg": data["vehicle_capacities"][vehicle_id],
            "Speed_km_per_h": data["vehicle_speeds_kmh"][vehicle_id],
            "Hourly_Rate_$": data["driver_hourly_rate"][vehicle_id],
            "Max_Range_km": round(data["max_range"][vehicle_id] / 1000.0, 1),
            "Stops": stops,
            "Total_Load_kg": route_load,
            "Load_Utilization_%": round((route_load / data["vehicle_capacities"][vehicle_id]) * 100.0, 1) if data["vehicle_capacities"][vehicle_id] > 0 else 0.0,
            "Total_Distance_m": route_distance_m,
            "Total_Distance_km": round(route_distance_m / 1000.0, 1),
            "Distance_Utilization_%": round((route_distance_m / data["max_range"][vehicle_id]) * 100.0, 1) if data["max_range"][vehicle_id] > 0 else 0.0,
            "Total_Time_min": round(total_time_minutes_true, 1),
            "Total_Time_hours": round(total_time_minutes_true / 60.0, 2),
            "Total_Cost_$": round(total_cost_dollars, 2)
        })

    routes_df = pd.DataFrame(routes_data)
    visits_df = pd.DataFrame(visits)
    return routes_df, visits_df


def print_summary(routes_df):
    print("\n" + "=" * 80)
    print("SOLUTION SUMMARY (Time classes per speed)")
    print("=" * 80)
    if routes_df.empty:
        print("No routes found.")
        return

    print(f"Vehicles Used: {len(routes_df)} out of {NUM_VEHICLES} available")
    print(f"Total Stops: {routes_df['Stops'].sum()}")
    print(f"Total Distance: {routes_df['Total_Distance_km'].sum():,.1f} km")
    print(f"Total Time: {routes_df['Total_Time_hours'].sum():,.2f} hours")
    print(f"Total Cost: ${routes_df['Total_Cost_$'].sum():,.2f} (fuel+labor)")
    print(f"Total Load Delivered: {routes_df['Total_Load_kg'].sum():,.0f} kg\n")

    print(f"Average Load Utilization: {routes_df['Load_Utilization_%'].mean():.1f}%")
    print(f"Average Range Utilization: {routes_df['Distance_Utilization_%'].mean():.1f}%")
    print(f"Average Stops per Vehicle: {routes_df['Stops'].mean():.1f}")
    print(f"Average Distance per Vehicle: {routes_df['Total_Distance_km'].mean():,.1f} km")
    print(f"Average Cost per Vehicle: ${routes_df['Total_Cost_$'].mean():,.2f}")
    print("=" * 80 + "\n")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    data, manager, routing, solution, time_dims = solve_with_time_classes(time_limit_seconds=10)
    routes_df, visits_df = create_solution_tables(data, manager, routing, solution, time_dims)
    print_summary(routes_df)

    # Example: inspect outputs
    # print(routes_df.head())
    # visits_df.to_csv("visits_time_classes.csv", index=False)