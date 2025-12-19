import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pydeck as pdk

import streamlit as st
from streamlit_folium import st_folium
import folium
import numpy as np
from collections import Counter


class VehicleType(Enum):
	CAR = "car"
	BIKE = "bike"
	TRUCK = "truck"
	BUS = "bus"
	MOTORCYCLE = "motorcycle"

class ObstacleType(Enum):
	POTHOLE = "pothole"
	BARRIER = "barrier"
	CONSTRUCTION = "construction"
	TRAFFIC_LIGHT = "traffic_light"
	SPEED_BUMP = "speed_bump"

@dataclass
class Vehicle:
	vehicle_type: VehicleType
	position: float  # position along road in meters
	velocity: float  # velocity in m/s
	length: float    # vehicle length in meters
	max_speed: float # max speed in m/s
	lane: int = 0    # which lane the vehicle is in
	acceleration: float = 0.0
	color: str = "blue"  # color for visualization

@dataclass
class Obstacle:
	obstacle_type: ObstacleType
	position: float  # position along road in meters
	severity: float  # 0-1, how much it affects traffic
	width: float     # width of obstacle in meters
	lane: int = 0    # which lane the obstacle affects

@dataclass
class Road:
	length: float
	lanes: int = 2
	lane_width: float = 3.5  # meters per lane
	road_type: str = "urban"  # urban, highway, residential

def create_map(center: List[float] = [28.6139, 77.2090], zoom_start: int = 13) -> folium.Map:
	m = folium.Map(location=center, zoom_start=zoom_start, control_scale=True, tiles="cartodbpositron")
	# Enable draw controls for roads, vehicles, and obstacles
	from folium.plugins import Draw
	Draw(
		draw_options={
			"polyline": True,  # For roads
			"polygon": True,   # For construction zones
			"rectangle": True, # For barriers
			"circle": True,    # For potholes, speed bumps
			"circlemarker": True, # For traffic lights
			"marker": True,    # For general obstacles
		},
		edit_options={"edit": True, "remove": True},
	).add_to(m)
	return m


def main() -> None:
	st.set_page_config(page_title="Road Engineering Simulation Tool", layout="wide")
	st.title("🛣️ Road Engineering Simulation Tool")
	st.caption("Design roads, add vehicles & obstacles, analyze traffic flow for engineering solutions")

	# Initialize session state
	if "vehicles" not in st.session_state:
		st.session_state["vehicles"] = []
	if "obstacles" not in st.session_state:
		st.session_state["obstacles"] = []

	with st.sidebar:
		st.header("🗺️ Map Controls")
		default_city = st.selectbox("Quick center", [
			"New Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Kolkata", "Chennai"
		], index=0)
		city_to_latlon = {
			"New Delhi": [28.6139, 77.2090], "Mumbai": [19.0760, 72.8777],
			"Bengaluru": [12.9716, 77.5946], "Hyderabad": [17.3850, 78.4867],
			"Kolkata": [22.5726, 88.3639], "Chennai": [13.0827, 80.2707],
		}
		zoom = st.slider("Zoom", 8, 18, 13)
		reset = st.button("Re-center map")

		st.divider()
		st.header("🛣️ Road Design")
		road_type = st.selectbox("Road Type", ["Urban", "Highway", "Residential"], index=0)
		num_lanes = st.slider("Number of Lanes", 1, 6, 2)
		lane_width = st.slider("Lane Width (m)", 2.5, 4.0, 3.5, 0.1)
		
		if "road_config" not in st.session_state:
			st.session_state["road_config"] = {"type": road_type, "lanes": num_lanes, "width": lane_width}

		st.divider()
		st.header("🚗 Vehicle Fleet")
		st.write("Add vehicles to your simulation:")
		
		vehicle_type = st.selectbox("Vehicle Type", [vt.value for vt in VehicleType])
		num_vehicles = st.slider("Number", 1, 100, 10)
		if st.button("Add Vehicles"):
			add_vehicles_to_simulation(vehicle_type, num_vehicles)
		
		st.write(f"**Current Fleet:** {len(st.session_state['vehicles'])} vehicles")
		if st.button("Clear All Vehicles"):
			st.session_state["vehicles"] = []

		st.divider()
		st.header("🚧 Obstacles")
		st.write("Add obstacles to test road conditions:")
		
		obstacle_type = st.selectbox("Obstacle Type", [ot.value for ot in ObstacleType])
		obstacle_lane = st.slider("Lane", 0, num_lanes-1, 0)
		severity = st.slider("Severity (0-1)", 0.0, 1.0, 0.5, 0.1)
		if st.button("Add Obstacle"):
			add_obstacle_to_simulation(obstacle_type, severity, obstacle_lane)
		
		st.write(f"**Current Obstacles:** {len(st.session_state['obstacles'])}")
		if st.button("Clear All Obstacles"):
			st.session_state["obstacles"] = []

		st.divider()
		st.header("🎮 Simulation Controls")
		sim_speed = st.slider("Simulation Speed", 0.1, 5.0, 1.0, 0.1)
		show_animation = st.checkbox("Show Real-time Animation", value=True)
		show_3d = st.checkbox("Show 3D Deck.gl View", value=False)
		st.session_state["show_3d"] = show_3d
		st.divider()
		st.header("⚡ Performance")
		light_mode = st.checkbox("Lightweight Rendering (smoother)", value=True, help="Reduce redraws to improve smoothness")
		update_every_n_steps = st.slider("Update UI every N steps", 1, 20, 5)
		st.session_state["light_mode"] = light_mode
		st.session_state["update_every_n_steps"] = update_every_n_steps
		if st.button("🚀 Start Traffic Simulation", type="primary"):
			run_visual_traffic_simulation(sim_speed, show_animation)

	if reset or "map_center" not in st.session_state:
		st.session_state["map_center"] = city_to_latlon.get(default_city, [28.6139, 77.2090])

	m = create_map(center=st.session_state["map_center"], zoom_start=zoom)
	map_col, control_col = st.columns([3, 2])

	with map_col:
		returned = st_folium(m, width=None, height=640, returned_objects=["last_active_drawing", "all_drawings", "bounds"])

	with control_col:
		st.subheader("🛣️ Road Design")
		all_drawings: Optional[List[Dict[str, Any]]] = returned.get("all_drawings") if returned else None
		
		# Debug information
		if all_drawings:
			st.write(f"**Debug:** Found {len(all_drawings)} drawn features")
			for i, feature in enumerate(all_drawings):
				geom_type = feature.get("geometry", {}).get("type", "Unknown")
				st.write(f"Feature {i+1}: {geom_type}")
		
		if all_drawings:
			geojson = {
				"type": "FeatureCollection",
				"features": all_drawings,
			}
			st.session_state["geojson"] = geojson
			
			# Parse road - improved parsing
			road_coords, drawn_obstacles = parse_geojson_improved(geojson)
			if road_coords:
				st.success(f"✅ Road: {len(road_coords)} points")
				st.session_state["road_coords"] = road_coords
				
				# Show road stats
				length_m = polyline_length_meters(road_coords)
				st.metric("Road Length", f"{length_m:.1f} m")
				
				# Show current fleet and obstacles
				st.write(f"**Fleet:** {len(st.session_state['vehicles'])} vehicles")
				st.write(f"**Obstacles:** {len(st.session_state['obstacles'])}")
				
				# Show road coordinates for debugging
				if st.checkbox("Show Road Coordinates"):
					st.write("Road coordinates:")
					for i, coord in enumerate(road_coords):
						st.write(f"Point {i+1}: {coord}")
			else:
				st.warning("⚠️ No road detected. Make sure you drew a polyline (line) on the map.")
				st.write("**Tips:**")
				st.write("1. Use the polyline tool (diagonal line with dots)")
				st.write("2. Click multiple points to create a line")
				st.write("3. Double-click to finish the line")
				
				# Manual road creation option
				st.divider()
				st.write("**Alternative: Create Road Manually**")
				if st.button("Create Sample Road"):
					# Create a sample road for testing
					sample_road = [
						(28.6139, 77.2090),  # New Delhi center
						(28.6149, 77.2100),  # 100m east
						(28.6159, 77.2110),  # 200m east
						(28.6169, 77.2120),  # 300m east
					]
					st.session_state["road_coords"] = sample_road
					st.success("✅ Sample road created! You can now add vehicles and run simulation.")
					st.rerun()
		else:
			st.info("📍 Use map tools to draw your road design")

	# Display simulation results
	display_simulation_results()


def add_vehicles_to_simulation(vehicle_type_str: str, num_vehicles: int) -> None:
	"""Add vehicles to the simulation fleet."""
	vehicle_type = VehicleType(vehicle_type_str)
	
	# Vehicle specifications
	vehicle_specs = {
		VehicleType.CAR: {"length": 4.5, "max_speed": 15.0},  # 54 km/h
		VehicleType.BIKE: {"length": 1.8, "max_speed": 8.0},  # 29 km/h
		VehicleType.TRUCK: {"length": 12.0, "max_speed": 12.0},  # 43 km/h
		VehicleType.BUS: {"length": 12.0, "max_speed": 10.0},  # 36 km/h
		VehicleType.MOTORCYCLE: {"length": 2.0, "max_speed": 12.0},  # 43 km/h
	}
	
	spec = vehicle_specs[vehicle_type]
	
	for _ in range(num_vehicles):
		vehicle = Vehicle(
			vehicle_type=vehicle_type,
			position=0.0,  # Will be distributed along road
			velocity=0.0,
			length=spec["length"],
			max_speed=spec["max_speed"]
		)
		st.session_state["vehicles"].append(vehicle)
	
	st.success(f"Added {num_vehicles} {vehicle_type.value}s to simulation")


def add_obstacle_to_simulation(obstacle_type_str: str, severity: float, lane: int = 0) -> None:
	"""Add an obstacle to the simulation."""
	obstacle_type = ObstacleType(obstacle_type_str)
	
	# Obstacle specifications
	obstacle_specs = {
		ObstacleType.POTHOLE: {"width": 2.0, "base_severity": 0.3},
		ObstacleType.BARRIER: {"width": 5.0, "base_severity": 0.8},
		ObstacleType.CONSTRUCTION: {"width": 20.0, "base_severity": 0.9},
		ObstacleType.TRAFFIC_LIGHT: {"width": 1.0, "base_severity": 0.6},
		ObstacleType.SPEED_BUMP: {"width": 3.0, "base_severity": 0.4},
	}
	
	spec = obstacle_specs[obstacle_type]
	obstacle = Obstacle(
		obstacle_type=obstacle_type,
		position=0.0,  # Will be positioned along road
		severity=min(1.0, severity * spec["base_severity"]),
		width=spec["width"],
		lane=lane
	)
	st.session_state["obstacles"].append(obstacle)
	
	st.success(f"Added {obstacle_type.value} obstacle (severity: {obstacle.severity:.1f})")


def run_visual_traffic_simulation(sim_speed: float, show_animation: bool) -> None:
	"""Run visual traffic simulation with real-time animation."""
	road_coords = st.session_state.get("road_coords")
	if not road_coords:
		st.error("❌ No road drawn! Draw a road first.")
		return
	
	vehicles = st.session_state.get("vehicles", [])
	obstacles = st.session_state.get("obstacles", [])
	road_config = st.session_state.get("road_config", {"lanes": 2, "width": 3.5})
	
	if not vehicles:
		st.error("❌ No vehicles in simulation! Add some vehicles first.")
		return
	
	road_length = polyline_length_meters(road_coords)
	
	# Create road object
	road = Road(
		length=road_length,
		lanes=road_config["lanes"],
		lane_width=road_config["width"],
		road_type=road_config["type"].lower()
	)
	
	# Distribute vehicles across lanes
	np.random.seed(42)
	for i, vehicle in enumerate(vehicles):
		vehicle.position = (i * road_length / len(vehicles)) + np.random.uniform(-10, 10)
		vehicle.velocity = np.random.uniform(2, vehicle.max_speed * 0.8)
		vehicle.lane = i % road.lanes
		# Set vehicle colors
		vehicle_colors = {"car": "blue", "truck": "red", "bus": "green", "bike": "orange", "motorcycle": "purple"}
		vehicle.color = vehicle_colors.get(vehicle.vehicle_type.value, "blue")
	
	# Position obstacles along road
	for i, obstacle in enumerate(obstacles):
		obstacle.position = (i + 1) * road_length / (len(obstacles) + 1)
	
	# Precompute and cache cumulative distances for 3D re-use
	if "road_coords" in st.session_state:
		st.session_state["_cum_distances"] = build_cumulative_distances(st.session_state["road_coords"])

	# Run simulation with visualization
	if show_animation:
		run_animated_simulation(road, vehicles, obstacles, sim_speed)
	else:
		run_static_simulation(road, vehicles, obstacles)


def run_animated_simulation(road: Road, vehicles: List[Vehicle], obstacles: List[Obstacle], sim_speed: float) -> None:
	"""Run animated traffic simulation."""
	st.header("🚗 Real-time Traffic Simulation")
	
	# Create progress bar
	progress_bar = st.progress(0)
	status_text = st.empty()
	
	# Placeholders to avoid accumulating charts
	plot_placeholder = st.empty()
	deck_placeholder = st.empty()

	# First frame
	fig = create_traffic_plot(road, vehicles, obstacles)
	
	# Simulation parameters
	dt = 0.2
	T = 60.0
	steps = int(T / dt)
	
	# Store simulation data
	sim_data = {
		"positions": [],
		"velocities": [],
		"times": []
	}
	
	# Run simulation
	light_mode = bool(st.session_state.get("light_mode", True))
	update_every = int(st.session_state.get("update_every_n_steps", 5))

	for step in range(steps):
		# Update vehicles
		update_vehicles(vehicles, road, obstacles, dt)
		
		# Store data
		sim_data["positions"].append([v.position for v in vehicles])
		sim_data["velocities"].append([v.velocity for v in vehicles])
		sim_data["times"].append(step * dt)
		
		# Update progress
		progress = (step + 1) / steps
		progress_bar.progress(progress)
		status_text.text(f"Simulation Step: {step + 1}/{steps} | Time: {step * dt:.1f}s")
		
		# Show current state every few steps
		should_update = (step % (update_every if light_mode else 5) == 0)
		if should_update:
			fig = create_traffic_plot(road, vehicles, obstacles, step * dt)
			plot_placeholder.plotly_chart(fig, use_container_width=True)
			# Optional 3D view (skip during loop if light_mode to avoid heavy redraws)
			if st.session_state.get("show_3d") and not light_mode:
				cum = st.session_state.get("_cum_distances")
				deck = create_deck_gl_view(st.session_state.get("road_coords", []), road, vehicles, obstacles, cum)
				if deck:
					deck_placeholder.pydeck_chart(deck, use_container_width=True)
		
		# Control simulation speed
		time.sleep(0.1 / sim_speed)
	
	# Final results
	st.success("✅ Simulation Complete!")
	# Render final charts and deck once
	plot_placeholder.plotly_chart(create_traffic_plot(road, vehicles, obstacles, T), use_container_width=True)
	if st.session_state.get("show_3d"):
		cum = st.session_state.get("_cum_distances")
		deck = create_deck_gl_view(st.session_state.get("road_coords", []), road, vehicles, obstacles, cum)
		if deck:
			deck_placeholder.pydeck_chart(deck, use_container_width=True)
	# Analysis
	display_simulation_results_visual(road, vehicles, obstacles, sim_data)


def run_static_simulation(road: Road, vehicles: List[Vehicle], obstacles: List[Obstacle]) -> None:
	"""Run static traffic simulation with final results."""
	st.header("📊 Traffic Analysis Results")
	
	# Run simulation
	dt = 0.2
	T = 60.0
	steps = int(T / dt)
	
	for step in range(steps):
		update_vehicles(vehicles, road, obstacles, dt)
	
	# Create visualization
	fig = create_traffic_plot(road, vehicles, obstacles, T)
	st.plotly_chart(fig, use_container_width=True)
	
	# Show metrics
	display_simulation_results_visual(road, vehicles, obstacles, None)


def create_traffic_plot(road: Road, vehicles: List[Vehicle], obstacles: List[Obstacle], time: float = 0.0) -> go.Figure:
	"""Create interactive traffic visualization plot."""
	fig = go.Figure()
	
	# Add road lanes
	for lane in range(road.lanes):
		y_pos = lane * road.lane_width
		fig.add_trace(go.Scatter(
			x=[0, road.length],
			y=[y_pos, y_pos],
			mode='lines',
			line=dict(color='gray', width=2),
			name=f'Lane {lane + 1}',
			showlegend=False
		))
	
	# Add obstacles
	for obstacle in obstacles:
		obstacle_y = obstacle.lane * road.lane_width
		fig.add_trace(go.Scatter(
			x=[obstacle.position - obstacle.width/2, obstacle.position + obstacle.width/2],
			y=[obstacle_y, obstacle_y],
			mode='lines',
			line=dict(color='red', width=8),
			name=f'{obstacle.obstacle_type.value}',
			showlegend=True
		))
	
	# Add vehicles
	for vehicle in vehicles:
		vehicle_y = vehicle.lane * road.lane_width
		fig.add_trace(go.Scatter(
			x=[vehicle.position],
			y=[vehicle_y],
			mode='markers',
			marker=dict(
				size=15,
				color=vehicle.color,
				symbol='square'
			),
			name=f'{vehicle.vehicle_type.value}',
			showlegend=True,
			text=f'Pos: {vehicle.position:.1f}m<br>Speed: {vehicle.velocity:.1f}m/s',
			hovertemplate='%{text}<extra></extra>'
		))
	
	# Update layout
	fig.update_layout(
		title=f"Traffic Simulation - Time: {time:.1f}s",
		xaxis_title="Road Position (meters)",
		yaxis_title="Lane Position (meters)",
		width=800,
		height=400,
		showlegend=True
	)
	
	return fig


def update_vehicles(vehicles: List[Vehicle], road: Road, obstacles: List[Obstacle], dt: float) -> None:
	"""Update vehicle positions and velocities."""
	# Sort vehicles by position
	vehicles.sort(key=lambda v: v.position)
	
	for i, vehicle in enumerate(vehicles):
		# Calculate following distance
		if i < len(vehicles) - 1:
			leader = vehicles[i + 1]
			following_distance = leader.position - vehicle.position
		else:
			following_distance = road.length - vehicle.position
		
		# Calculate obstacle effects
		obstacle_effect = 0.0
		for obstacle in obstacles:
			if obstacle.lane == vehicle.lane:
				distance_to_obstacle = abs(vehicle.position - obstacle.position)
				if distance_to_obstacle < obstacle.width * 2:
					effect = obstacle.severity * np.exp(-distance_to_obstacle / obstacle.width)
					obstacle_effect = max(obstacle_effect, effect)
		
		# Car-following model
		desired_speed = vehicle.max_speed * (1 - obstacle_effect)
		safe_distance = max(2.0, vehicle.velocity * 1.5)
		
		if following_distance < safe_distance:
			vehicle.velocity = max(0, vehicle.velocity - 2.0 * dt)
		else:
			acceleration = 2.0 * (desired_speed - vehicle.velocity) / vehicle.max_speed
			vehicle.velocity = min(desired_speed, vehicle.velocity + acceleration * dt)
		
		# Update position
		vehicle.position = min(road.length, vehicle.position + vehicle.velocity * dt)


def display_simulation_results_visual(road: Road, vehicles: List[Vehicle], obstacles: List[Obstacle], sim_data: Optional[Dict]) -> None:
	"""Display visual simulation results."""
	st.divider()
	st.header("📊 Traffic Analysis Results")
	
	# Key metrics
	col1, col2, col3, col4 = st.columns(4)
	
	avg_speed = np.mean([v.velocity for v in vehicles])
	max_speed = np.max([v.velocity for v in vehicles])
	ideal_speed = np.mean([v.max_speed for v in vehicles])
	congestion_level = max(0, 1 - avg_speed / ideal_speed)
	efficiency = avg_speed / ideal_speed
	
	with col1:
		st.metric("Avg Speed", f"{avg_speed:.1f} m/s", f"{avg_speed*3.6:.1f} km/h")
	with col2:
		st.metric("Max Speed", f"{max_speed:.1f} m/s", f"{max_speed*3.6:.1f} km/h")
	with col3:
		st.metric("Congestion Level", f"{congestion_level:.1%}")
	with col4:
		st.metric("Efficiency", f"{efficiency:.1%}")
	
	# Vehicle distribution
	st.subheader("🚗 Vehicle Distribution")
	vehicle_counts = {}
	for vehicle in vehicles:
		vehicle_counts[vehicle.vehicle_type.value] = vehicle_counts.get(vehicle.vehicle_type.value, 0) + 1
	
	fig_dist = px.bar(
		x=list(vehicle_counts.keys()),
		y=list(vehicle_counts.values()),
		title="Vehicle Count by Type",
		labels={'x': 'Vehicle Type', 'y': 'Count'}
	)
	st.plotly_chart(fig_dist, use_container_width=True)
	
	# Speed analysis
	if sim_data:
		st.subheader("📈 Speed Analysis Over Time")
		fig_speed = go.Figure()
		fig_speed.add_trace(go.Scatter(
			x=sim_data['times'],
			y=[np.mean(v) for v in sim_data['velocities']],
			mode='lines',
			name='Average Speed'
		))
		fig_speed.update_layout(
			title="Average Speed Over Time",
			xaxis_title="Time (seconds)",
			yaxis_title="Speed (m/s)"
		)
		st.plotly_chart(fig_speed, use_container_width=True)
	
	# Engineering insights
	st.subheader("🔧 Engineering Insights")
	
	if congestion_level > 0.7:
		st.error("🚨 **High Congestion Detected!** Consider widening road or adding lanes.")
	elif congestion_level > 0.4:
		st.warning("⚠️ **Moderate Congestion** - Monitor traffic flow and consider optimization.")
	else:
		st.success("✅ **Good Traffic Flow** - Road design appears efficient.")
	
	# Lane analysis
	st.write("**Lane Usage Analysis:**")
	lane_usage = Counter([v.lane for v in vehicles])
	for lane, count in lane_usage.items():
		usage_pct = count / len(vehicles) * 100
		st.write(f"Lane {lane + 1}: {count} vehicles ({usage_pct:.1f}%)")


# -----------------------------
# 3D (Deck.gl) utilities
# -----------------------------
def build_cumulative_distances(coords: List[Tuple[float, float]]) -> List[float]:
	"""Precompute cumulative distances (meters) along a lat/lng polyline."""
	if not coords:
		return []
	cum = [0.0]
	for i in range(1, len(coords)):
		cum.append(cum[-1] + haversine_m(coords[i - 1], coords[i]))
	return cum


def bearing_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
	"""Return bearing from point a to b in degrees (0=N, clockwise)."""
	lat1, lon1 = np.radians(a[0]), np.radians(a[1])
	lat2, lon2 = np.radians(b[0]), np.radians(b[1])
	dlon = lon2 - lon1
	y = np.sin(dlon) * np.cos(lat2)
	x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
	brng = np.degrees(np.arctan2(y, x))
	return (brng + 360.0) % 360.0


def position_to_latlng_heading(pos_m: float, coords: List[Tuple[float, float]], cum: Optional[List[float]] = None) -> Tuple[float, float, float]:
	"""Map a distance along the polyline to lat/lng and local heading (deg)."""
	if not coords:
		return 0.0, 0.0, 0.0
	if cum is None:
		cum = build_cumulative_distances(coords)
	L = cum[-1] if cum else 0.0
	if L <= 0.0:
		lat, lng = coords[0]
		return float(lat), float(lng), 0.0
	# Clamp
	s = float(max(0.0, min(pos_m, L)))
	# Find segment
	for i in range(len(cum) - 1):
		if cum[i] <= s <= cum[i + 1]:
			seg_len = max(1e-6, cum[i + 1] - cum[i])
			alpha = (s - cum[i]) / seg_len
			lat = coords[i][0] + alpha * (coords[i + 1][0] - coords[i][0])
			lng = coords[i][1] + alpha * (coords[i + 1][1] - coords[i][1])
			hdg = bearing_deg(coords[i], coords[i + 1])
			return float(lat), float(lng), float(hdg)
	# Fallback end
	lat, lng = coords[-1]
	hdg = bearing_deg(coords[-2], coords[-1]) if len(coords) >= 2 else 0.0
	return float(lat), float(lng), float(hdg)


def create_deck_gl_view(road_coords: List[Tuple[float, float]], road: Road, vehicles: List[Vehicle], obstacles: List[Obstacle], cum_distances: Optional[List[float]] = None) -> Optional[pdk.Deck]:
	"""Create a Deck.gl scene with road, vehicles, and obstacles in 3D."""
	if not road_coords:
		return None
	# Center view
	center_lat, center_lng = road_coords[0]
	if len(road_coords) >= 2:
		center_lat = float(np.mean([c[0] for c in road_coords]))
		center_lng = float(np.mean([c[1] for c in road_coords]))

	# Build data (reuse if provided)
	cum = cum_distances if cum_distances is not None else build_cumulative_distances(road_coords)
	# Road path as [lng, lat]
	road_path = [[lng, lat] for (lat, lng) in road_coords]

	vehicle_rows = []
	for v in vehicles:
		lat, lng, hdg = position_to_latlng_heading(v.position, road_coords, cum)
		vehicle_rows.append({
			"position": [lng, lat],
			"elevation": max(3.0, v.length),
			"radius": max(1.5, v.length * 0.3),
			"color": {
				"car": [0, 122, 255],
				"truck": [220, 20, 60],
				"bus": [34, 139, 34],
				"bike": [255, 140, 0],
				"motorcycle": [128, 0, 128],
			}.get(v.vehicle_type.value, [0, 122, 255]),
			"bearing": hdg,
			"type": v.vehicle_type.value,
		})

	obstacle_rows = []
	for ob in obstacles:
		lat, lng, _ = position_to_latlng_heading(ob.position, road_coords, cum)
		obstacle_rows.append({
			"position": [lng, lat],
			"elevation": max(2.0, ob.width * 0.8) * (1.0 + ob.severity),
			"radius": max(1.5, ob.width * 0.5),
			"color": [200, 0, 0] if ob.obstacle_type.value in ("barrier", "construction") else [255, 165, 0],
			"type": ob.obstacle_type.value,
		})

	layers: List[pdk.Layer] = []
	# Road
	layers.append(pdk.Layer(
		"PathLayer",
		data=[{"path": road_path}],
		get_path="path",
		get_color=[120, 120, 120],
		width_scale=1,
		width_min_pixels=6,
		get_width=4,
		elevation_scale=1,
		pickable=False,
	))
	# Vehicles as extruded columns
	layers.append(pdk.Layer(
		"ColumnLayer",
		data=vehicle_rows,
		get_position="position",
		get_elevation="elevation",
		get_fill_color="color",
		get_radius="radius",
		elevation_scale=1,
		extruded=True,
		pickable=True,
	))
	# Obstacles as extruded columns
	layers.append(pdk.Layer(
		"ColumnLayer",
		data=obstacle_rows,
		get_position="position",
		get_elevation="elevation",
		get_fill_color="color",
		get_radius="radius",
		elevation_scale=1,
		extruded=True,
		pickable=True,
	))

	deck = pdk.Deck(
		map_provider="carto",
		map_style="light",
		initial_view_state=pdk.ViewState(
			latitude=center_lat,
			longitude=center_lng,
			zoom=15,
			pitch=50,
			bearing=0,
		),
		layers=layers,
		tooltip={
			"html": "<b>{type}</b>",
			"style": {"color": "white"}
		},
	)
	return deck

def run_enhanced_simulation(road_coords: List[Tuple[float, float]], road_length: float) -> None:
	"""Run enhanced traffic simulation with multiple vehicle types and obstacles."""
	vehicles = st.session_state.get("vehicles", [])
	obstacles = st.session_state.get("obstacles", [])
	
	if not vehicles:
		st.error("❌ No vehicles in simulation! Add some vehicles first.")
		return
	
	# Distribute vehicles along road
	np.random.seed(42)
	for i, vehicle in enumerate(vehicles):
		vehicle.position = (i * road_length / len(vehicles)) + np.random.uniform(-10, 10)
		vehicle.velocity = np.random.uniform(2, vehicle.max_speed * 0.8)
	
	# Position obstacles along road
	for i, obstacle in enumerate(obstacles):
		obstacle.position = (i + 1) * road_length / (len(obstacles) + 1)
	
	# Run simulation
	result = run_realistic_traffic_sim(road_length, vehicles, obstacles)
	st.session_state["sim_result"] = result
	st.success("✅ Simulation completed!")


def display_simulation_results() -> None:
	"""Display simulation results with engineering insights."""
	result = st.session_state.get("sim_result")
	if not result:
		return
	
	st.divider()
	st.header("📊 Traffic Analysis Results")
	
	# Key metrics
	col1, col2, col3, col4 = st.columns(4)
	with col1:
		st.metric("Avg Speed", f"{result['avg_speed']:.1f} m/s", f"{result['avg_speed']*3.6:.1f} km/h")
	with col2:
		st.metric("Max Speed", f"{result['max_speed']:.1f} m/s", f"{result['max_speed']*3.6:.1f} km/h")
	with col3:
		st.metric("Congestion Level", f"{result['congestion_level']:.1%}")
	with col4:
		st.metric("Efficiency", f"{result['efficiency']:.1%}")
	
	# Charts
	col1, col2 = st.columns(2)
	with col1:
		st.subheader("Speed Over Time")
		st.line_chart(result['speed_time_series'])
	
	with col2:
		st.subheader("Vehicle Distribution")
		st.bar_chart(result['vehicle_distribution'])
	
	# Engineering insights
	st.subheader("🔧 Engineering Insights")
	
	if result['congestion_level'] > 0.7:
		st.error("🚨 **High Congestion Detected!** Consider widening road or adding lanes.")
	elif result['congestion_level'] > 0.4:
		st.warning("⚠️ **Moderate Congestion** - Monitor traffic flow and consider optimization.")
	else:
		st.success("✅ **Good Traffic Flow** - Road design appears efficient.")
	
	# Obstacle impact analysis
	if result['obstacle_impact']:
		st.write("**Obstacle Impact Analysis:**")
		for obstacle, impact in result['obstacle_impact'].items():
			if impact > 0.5:
				st.error(f"🚧 {obstacle}: High impact ({impact:.1%}) - Consider removal or mitigation")
			elif impact > 0.2:
				st.warning(f"⚠️ {obstacle}: Moderate impact ({impact:.1%})")
			else:
				st.info(f"ℹ️ {obstacle}: Low impact ({impact:.1%})")


def parse_geojson_improved(geojson: Dict[str, Any]) -> Tuple[Optional[List[Tuple[float, float]]], List[Dict[str, Any]]]:
	"""Improved GeoJSON parsing that handles different coordinate formats."""
	road_coords: Optional[List[Tuple[float, float]]] = None
	obstacles: List[Dict[str, Any]] = []
	
	features = geojson.get("features", [])
	
	for feature in features:
		geom = feature.get("geometry") or {}
		geom_type = geom.get("type")
		coords = geom.get("coordinates", [])
		
		# Debug output
		st.write(f"Processing {geom_type} with {len(coords)} coordinate sets")
		
		if geom_type == "LineString" and coords:
			# LineString coordinates are [lng, lat] pairs
			parsed_coords = []
			for coord in coords:
				if len(coord) >= 2:
					lng, lat = float(coord[0]), float(coord[1])
					parsed_coords.append((lat, lng))  # Convert to (lat, lng)
			
			if parsed_coords:
				road_coords = parsed_coords
				st.write(f"✅ Found LineString road with {len(road_coords)} points")
		
		elif geom_type == "Polyline" and coords:
			# Polyline coordinates might be [lat, lng] pairs
			parsed_coords = []
			for coord in coords:
				if len(coord) >= 2:
					lat, lng = float(coord[0]), float(coord[1])
					# Check if coordinates make sense (lat should be -90 to 90)
					if -90 <= lat <= 90 and -180 <= lng <= 180:
						parsed_coords.append((lat, lng))
					else:
						# Try swapping if coordinates seem wrong
						lat, lng = float(coord[1]), float(coord[0])
						parsed_coords.append((lat, lng))
			
			if parsed_coords:
				road_coords = parsed_coords
				st.write(f"✅ Found Polyline road with {len(road_coords)} points")
		
		else:
			# Treat as obstacle
			obstacles.append(feature)
	
	return road_coords, obstacles


def parse_geojson(geojson: Dict[str, Any]) -> Tuple[Optional[List[Tuple[float, float]]], List[Dict[str, Any]]]:
	"""Extract first polyline as road and collect remaining as obstacles.

	Returns:
	- road_coords: list of (lat, lon) tuples for the first LineString/Polyline-like geometry
	- obstacles: list of remaining feature dicts
	"""
	road_coords: Optional[List[Tuple[float, float]]] = None
	obstacles: List[Dict[str, Any]] = []

	features = geojson.get("features", [])
	for feature in features:
		geom = feature.get("geometry") or {}
		geom_type = geom.get("type")
		# Folium Draw returns Leaflet-style coordinates [lat, lng]
		if road_coords is None and geom_type in ("LineString", "Polyline"):
			coords = geom.get("coordinates", [])
			# Normalize to (lat, lon)
			parsed = []
			for c in coords:
				if isinstance(c, (list, tuple)) and len(c) >= 2:
					lat, lon = float(c[1]), float(c[0]) if geom_type == "LineString" else float(c[0]), float(c[1])
					# Heuristic: Leaflet typically stores [lng, lat] for GeoJSON; Polyline may be [lat, lng]
					# Try to detect by range. If abs(lat) > 90, swap.
					if abs(lat) > 90:
						lat, lon = lon, lat
					parsed.append((lat, lon))
			if parsed:
				road_coords = parsed
		else:
			obstacles.append(feature)

	return road_coords, obstacles


def polyline_length_meters(coords: List[Tuple[float, float]]) -> float:
	"""Approximate length using haversine distance sum."""
	R = 6371000.0
	def haversine(p1, p2):
		lat1, lon1 = np.radians(p1[0]), np.radians(p1[1])
		lat2, lon2 = np.radians(p2[0]), np.radians(p2[1])
		dlat = lat2 - lat1
		dlon = lon2 - lon1
		a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
		c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
		return R*c
	return float(np.sum([haversine(coords[i], coords[i+1]) for i in range(len(coords)-1)])) if len(coords) > 1 else 0.0


def obstacle_positions_along_polyline(road_coords: List[Tuple[float, float]], obstacles: List[Dict[str, Any]]) -> List[float]:
	"""Project obstacle features to cumulative distance along the road polyline (meters)."""
	positions_m: List[float] = []
	if not road_coords:
		return positions_m
	for feat in obstacles:
		geom = feat.get("geometry") or {}
		if not geom:
			continue
		gtype = geom.get("type")
		if gtype == "Point":
			lng, lat = geom.get("coordinates", [None, None])
			if lat is None or lng is None:
				continue
			positions_m.append(project_point_onto_polyline_distance((float(lat), float(lng)), road_coords))
		elif gtype in ("Polygon", "MultiPolygon"):
			coords = geom.get("coordinates", [])
			if coords:
				# Take first ring, first vertex as proxy
				candidate = coords[0][0] if gtype == "Polygon" else coords[0][0][0]
				lng, lat = candidate[0], candidate[1]
				positions_m.append(project_point_onto_polyline_distance((float(lat), float(lng)), road_coords))
	return sorted([p for p in positions_m if p is not None])


def project_point_onto_polyline_distance(pt: Tuple[float, float], poly: List[Tuple[float, float]]) -> float:
	"""Return cumulative distance (m) from start to closest point on the polyline."""
	if len(poly) < 2:
		return 0.0
	# Coarse approach: sample vertices only for simplicity
	dists = [segment_path_length(poly, i) + haversine_m(poly[i], pt) for i in range(len(poly))]
	return float(min(dists))


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
	R = 6371000.0
	lat1, lon1 = np.radians(a[0]), np.radians(a[1])
	lat2, lon2 = np.radians(b[0]), np.radians(b[1])
	dlat = lat2 - lat1
	dlon = lon2 - lon1
	ae = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
	c = 2*np.arctan2(np.sqrt(ae), np.sqrt(1-ae))
	return float(R*c)


def segment_path_length(poly: List[Tuple[float, float]], idx: int) -> float:
	"""Cumulative length up to vertex idx."""
	if idx <= 0:
		return 0.0
	return float(np.sum([haversine_m(poly[i], poly[i+1]) for i in range(idx)]))


def run_realistic_traffic_sim(road_length: float, vehicles: List[Vehicle], obstacles: List[Obstacle], dt: float = 0.2, T: float = 60.0) -> Dict[str, Any]:
	"""Run realistic traffic simulation with multiple vehicle types and obstacles."""
	steps = int(T / dt)
	speed_time_series = []
	vehicle_positions = []
	
	# Sort vehicles by position
	vehicles = sorted(vehicles, key=lambda v: v.position)
	
	for step in range(steps):
		# Update each vehicle
		for i, vehicle in enumerate(vehicles):
			# Calculate following distance
			if i < len(vehicles) - 1:
				leader = vehicles[i + 1]
				following_distance = leader.position - vehicle.position
			else:
				following_distance = road_length - vehicle.position
			
			# Calculate obstacle effects
			obstacle_effect = 0.0
			for obstacle in obstacles:
				distance_to_obstacle = abs(vehicle.position - obstacle.position)
				if distance_to_obstacle < obstacle.width * 2:  # Within influence zone
					effect = obstacle.severity * np.exp(-distance_to_obstacle / obstacle.width)
					obstacle_effect = max(obstacle_effect, effect)
			
			# Car-following model with obstacle effects
			desired_speed = vehicle.max_speed * (1 - obstacle_effect)
			safe_distance = max(2.0, vehicle.velocity * 1.5)  # Safe following distance
			
			if following_distance < safe_distance:
				# Too close - slow down
				vehicle.velocity = max(0, vehicle.velocity - 2.0 * dt)
			else:
				# Accelerate towards desired speed
				acceleration = 2.0 * (desired_speed - vehicle.velocity) / vehicle.max_speed
				vehicle.velocity = min(desired_speed, vehicle.velocity + acceleration * dt)
			
			# Update position
			vehicle.position = min(road_length, vehicle.position + vehicle.velocity * dt)
		
		# Record metrics
		avg_speed = np.mean([v.velocity for v in vehicles])
		speed_time_series.append(avg_speed)
		vehicle_positions.append([v.position for v in vehicles])
	
	# Calculate final metrics
	all_speeds = [v.velocity for v in vehicles]
	avg_speed = np.mean(all_speeds)
	max_speed = np.max(all_speeds)
	
	# Congestion level (based on speed reduction)
	ideal_speed = np.mean([v.max_speed for v in vehicles])
	congestion_level = max(0, 1 - avg_speed / ideal_speed)
	
	# Efficiency (based on how close vehicles get to their max speeds)
	efficiency = avg_speed / ideal_speed
	
	# Vehicle distribution by type
	vehicle_distribution = {}
	for vehicle_type in VehicleType:
		count = sum(1 for v in vehicles if v.vehicle_type == vehicle_type)
		if count > 0:
			vehicle_distribution[vehicle_type.value] = count
	
	# Obstacle impact analysis
	obstacle_impact = {}
	for obstacle in obstacles:
		# Calculate how much this obstacle affects traffic
		impact = 0.0
		for vehicle in vehicles:
			distance = abs(vehicle.position - obstacle.position)
			if distance < obstacle.width * 3:  # Within influence zone
				impact += obstacle.severity * np.exp(-distance / obstacle.width)
		obstacle_impact[obstacle.obstacle_type.value] = min(1.0, impact / len(vehicles))
	
	return {
		"avg_speed": avg_speed,
		"max_speed": max_speed,
		"congestion_level": congestion_level,
		"efficiency": efficiency,
		"speed_time_series": speed_time_series,
		"vehicle_distribution": vehicle_distribution,
		"obstacle_impact": obstacle_impact,
		"final_positions": [v.position for v in vehicles],
		"vehicle_types": [v.vehicle_type.value for v in vehicles]
	}


def run_simple_traffic_sim(L: float, N: int, v_des_kmh: float, dt: float, T: float, obst_positions: List[float], seed: int = 0) -> Dict[str, Any]:
	"""Simple 1D traffic simulation along a line with soft obstacles.

	- L: road length (m)
	- N: number of cars
	- v_des_kmh: desired speed in km/h
	- dt: time step (s)
	- T: total time (s)
	- obst_positions: obstacle positions along line in meters

	Cars follow a basic car-following rule: v_{t+1} = v_t + a*(1 - (v_t/v0)^4) - brake_term
	Obstacles impose local speed drops near their positions.
	"""
	np.random.seed(seed)
	steps = int(T / dt)
	v0 = max(0.1, v_des_kmh * 1000.0 / 3600.0)
	positions = np.linspace(0, max(L - 1, 1.0), N)
	# small initial spacing noise
	positions += np.linspace(0, 1.0, N)
	velocities = np.full(N, min(v0, 5.0))
	a_max = 1.2
	b_comf = 1.5
	s0 = 2.0
	T_headway = 1.2
	obst_positions = sorted([p for p in obst_positions if 0 <= p <= L])

	avg_speeds_ts = []
	all_speeds = []
	for _ in range(steps):
		# compute spacing
		delta = np.roll(positions, -1) - positions
		delta[-1] = L - positions[-1]  # last car to end
		delta = np.maximum(delta, 0.1)

		# IDM-like acceleration
		v_rel = velocities - np.roll(velocities, -1)
		s_star = s0 + np.maximum(0.0, velocities * T_headway + (velocities * v_rel) / (2 * np.sqrt(a_max * b_comf)))
		a_idm = a_max * (1 - (velocities / v0) ** 4 - (s_star / delta) ** 2)

		# Obstacle braking: gaussian speed drop zones
		brake = np.zeros(N)
		for ob in obst_positions:
			influence = np.exp(-((positions - ob) ** 2) / (2 * (10.0 ** 2)))
			brake += 1.2 * influence

		acc = a_idm - brake
		velocities = np.maximum(0.0, velocities + acc * dt)
		positions = np.minimum(L, positions + velocities * dt)

		avg_speeds_ts.append(float(np.mean(velocities)))
		all_speeds.append(velocities.copy())

	return {
		"avg_speed_time_series": avg_speeds_ts,
		"final_positions": positions,
		"speeds_over_time": np.concatenate(all_speeds) if all_speeds else np.array([]),
	}


if __name__ == "__main__":
	main()


